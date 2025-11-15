import torch
import torch.nn as nn
import numpy as np

# ----------- IFFT模块（Conv1d实现） -----------
class ComplexIFFT(nn.Module):
    def __init__(self, fft_len):
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
        # x: (batch, 2, 64)
        x_real = x[:, 0:1, :]  # (batch, 1, 64)
        x_imag = x[:, 1:2, :]
        out_real = self.real_conv(x_real) - self.imag_conv(x_imag)  # (batch, 64, 1)
        out_imag = self.real_conv(x_imag) + self.imag_conv(x_real)  # (batch, 64, 1)
        out_real = out_real.permute(0, 2, 1)  # (batch, 1, 64)
        out_imag = out_imag.permute(0, 2, 1)  # (batch, 1, 64)
        out = torch.cat([out_real, out_imag], dim=1)  # (batch, 2, 64)
        return out  # 时域复数信号 (batch, 2, 64)

# ----------- 导频插入 -----------
def insert_pilots(data_freq):
    # data_freq: (batch, 2, 64)
    pilot_indices = torch.tensor([11, 25, 39, 53], dtype=torch.long, device=data_freq.device)
    pilot_value = torch.tensor([1.0, 0.0], dtype=data_freq.dtype, device=data_freq.device)
    pilot_value = pilot_value.unsqueeze(0).unsqueeze(0).expand(data_freq.size(0), pilot_indices.size(0), 2)
    out = data_freq.clone()
    # 交换维度以便 scatter
    out = out.permute(0, 2, 1)  # (batch, 64, 2)
    out.scatter_(1, pilot_indices.unsqueeze(0).unsqueeze(-1).expand(data_freq.size(0), pilot_indices.size(0), 2), pilot_value)
    out = out.permute(0, 2, 1)  # (batch, 2, 64)
    return out

# ----------- 主流程示例 -----------
def generate_ofdm_symbol(eq_freq):
    # eq_freq: (batch, 2, 64)
    freq_with_pilots = insert_pilots(eq_freq.clone())  # 插入导频
    ifft_model = ComplexIFFT(64)
    time_samples = ifft_model(freq_with_pilots)  # (batch, 2, 64)
    return time_samples  # 时域信号（无CP）

class OFDMGeneratorModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.ifft_model = ComplexIFFT(64)

    def forward(self, eq_freq):
        # eq_freq: (batch, 2, 64)
        freq_with_pilots = insert_pilots(eq_freq.clone())
        time_samples = self.ifft_model(freq_with_pilots)
        return time_samples  # (batch, 2, 64)

# 导出ONNX模型
if __name__ == "__main__":
    model = OFDMGeneratorModule()
    dummy_input = torch.randn(1, 2, 64)  # batch=1，频域复数输入
    torch.onnx.export(
        model,
        (dummy_input,),
        "ofdm_generator.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
    )