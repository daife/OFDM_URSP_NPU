#v2相对于v1的改进,
#1.总子载波64->256,数据子载波48->234
#2.调制方式64-QAM->1024-QAM，并且加入到网络中
#3.ldpc编码，将在下一个版本（v3）中实现

import torch
import torch.nn as nn
import numpy as np

# ----------- 1024-QAM调制模块 -----------
class QAM1024Modulator(nn.Module):
    def __init__(self):
        super().__init__()
        # 生成1024-QAM星座点
        constellation = self._generate_1024qam_constellation()
        self.register_buffer('constellation', constellation)
    
    def _generate_1024qam_constellation(self):
        # 1024-QAM = 32x32网格，每个符号10比特
        points = []
        for i in range(32):
            for q in range(32):
                # 归一化到单位功率
                real = (2*i - 31) / np.sqrt(341.3333)  # 341.3333 = (31^2 + 31^2) * 2/3
                imag = (2*q - 31) / np.sqrt(341.3333)
                points.append([real, imag])
        return torch.tensor(points, dtype=torch.float32)
    
    def forward(self, bits):
        # bits: (batch, num_symbols * 10) - 每个1024-QAM符号10比特
        batch_size = bits.size(0)
        num_symbols = bits.size(1) // 10

        bits = bits.view(batch_size, num_symbols, 10)
        # 用long类型初始化indices，避免类型转换
        indices = torch.zeros(batch_size, num_symbols, dtype=torch.long, device=bits.device)
        for i in range(10):
            indices = indices + bits[:, :, i] * (2 ** (9-i))
        # indices已经是long类型，无需再转换
        modulated = self.constellation[indices]  # (batch, num_symbols, 2)
        return modulated.permute(0, 2, 1)  # (batch, 2, num_symbols)

# ----------- IDFT模块（Conv1d实现，256点） -----------
class ComplexIDFT256(nn.Module):
    def __init__(self, fft_len=256):
        super().__init__()
        self.fft_len = fft_len

        n = np.arange(fft_len).reshape(-1, 1)
        k = np.arange(fft_len)
        cos_kernel = np.cos(2 * np.pi * k * n / fft_len) / fft_len
        sin_kernel = np.sin(2 * np.pi * k * n / fft_len) / fft_len

        real_kernel = torch.tensor(cos_kernel, dtype=torch.float32).unsqueeze(1)
        imag_kernel = torch.tensor(sin_kernel, dtype=torch.float32).unsqueeze(1)

        self.real_conv = nn.Conv1d(1, fft_len, kernel_size=fft_len, bias=False)
        self.imag_conv = nn.Conv1d(1, fft_len, kernel_size=fft_len, bias=False)
        self.real_conv.weight.data = real_kernel
        self.imag_conv.weight.data = imag_kernel
        self.real_conv.weight.requires_grad = False
        self.imag_conv.weight.requires_grad = False

    def forward(self, x):
        # x: (batch, 2, 256)
        x_real = x[:, 0:1, :]  # (batch, 1, 256)
        x_imag = x[:, 1:2, :]
        out_real = self.real_conv(x_real) - self.imag_conv(x_imag)  # (batch, 256, 1)
        out_imag = self.real_conv(x_imag) + self.imag_conv(x_real)  # (batch, 256, 1)
        out_real = out_real.permute(0, 2, 1)  # (batch, 1, 256)
        out_imag = out_imag.permute(0, 2, 1)  # (batch, 1, 256)
        out = torch.cat([out_real, out_imag], dim=1)  # (batch, 2, 256)
        return out  # 时域复数信号 (batch, 2, 256)

# ----------- 导频插入和子载波映射（向量化版本） ----------- 
class SubcarrierMapper(nn.Module):
    def __init__(self):
        super().__init__()
        # 按照结构分段构造
        # 保护子载波: 0~5
        # 导频: 6
        # 数据: 7~58 (52个)
        # 导频: 59
        # 数据: 60~124 (65个)
        # DC: 125~131 (7个)
        # 数据: 132~196 (65个)
        # 导频: 197
        # 数据: 198~249 (52个)
        # 导频: 250
        # 保护子载波: 251~255

        # 数据子载波索引
        data_indices = []
        # 7~58
        data_indices += list(range(7, 59))
        # 60~124
        data_indices += list(range(60, 125))
        # 132~196
        data_indices += list(range(132, 197))
        # 198~249
        data_indices += list(range(198, 250))

        assert len(data_indices) == 234, f"数据子载波数量应为234，实际为{len(data_indices)}"
        # 导频索引
        pilot_indices = torch.tensor([6, 59, 197, 250], dtype=torch.long)
        self.register_buffer('data_indices', torch.tensor(data_indices, dtype=torch.long))
        self.register_buffer('pilot_indices', pilot_indices)

    def forward(self, data_freq):
        # data_freq: (batch, 2, 234)
        batch_size = data_freq.size(0)
        device = data_freq.device
        dtype = data_freq.dtype
        output = torch.zeros(batch_size, 2, 256, dtype=dtype, device=device)
        # 用scatter插入数据子载波
        # shape: (batch, 2, 234) -> (batch, 2, 256)
        data_indices = self.data_indices.unsqueeze(0).unsqueeze(0).expand(batch_size, 2, -1)  # (batch, 2, 234)
        output = output.scatter(2, data_indices, data_freq)
        # 用scatter插入导频
        pilot_value = torch.tensor([1.0, 0.0], dtype=dtype, device=device).view(1, 2, 1).expand(batch_size, 2, self.pilot_indices.size(0))
        pilot_indices = self.pilot_indices.unsqueeze(0).unsqueeze(0).expand(batch_size, 2, -1)  # (batch, 2, 4)
        output = output.scatter(2, pilot_indices, pilot_value)
        return output

# ----------- 主OFDM生成器模块 -----------
class OFDMGeneratorModuleV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.qam_modulator = QAM1024Modulator()
        self.subcarrier_mapper = SubcarrierMapper()
        self.ifft_model = ComplexIDFT256(256)

    def forward(self, bits):
        # bits: (batch, 234*10) - 234个数据子载波，每个1024-QAM符号10比特
        # 1024-QAM调制
        modulated_data = self.qam_modulator(bits)  # (batch, 2, 234)
        
        # 子载波映射和导频插入
        freq_domain = self.subcarrier_mapper(modulated_data)  # (batch, 2, 256)
        
        # IDFT变换到时域
        time_samples = self.ifft_model(freq_domain)  # (batch, 2, 256)
        
        return time_samples

# ----------- 独立函数接口（兼容v1） -----------
def generate_ofdm_symbol_v2(eq_freq):
    # eq_freq: (batch, 2, 234) - 已调制的数据子载波
    mapper = SubcarrierMapper()
    freq_with_pilots = mapper(eq_freq)
    ifft_model = ComplexIDFT256(256)
    time_samples = ifft_model(freq_with_pilots)
    return time_samples

# 导出ONNX模型
if __name__ == "__main__":
    model = OFDMGeneratorModuleV2()
    # dummy_input用int64类型
    dummy_input = torch.randint(0, 2, (1, 234*10), dtype=torch.int64)  # 随机比特流

    torch.onnx.export(
        model,
        (dummy_input,),
        "ofdm_generator_v2.onnx",
        input_names=["bits"],
        output_names=["ofdm_time_samples"],
        opset_version=18,
        dynamic_axes={"bits": {0: "batch"}, "ofdm_time_samples": {0: "batch"}}
    )

    print("OFDM Generator V2 模型已导出到 ofdm_generator_v2.onnx")
    print(f"输入: 比特流 (batch, {234*10})")
    print("输出: OFDM时域信号 (batch, 2, 256)")