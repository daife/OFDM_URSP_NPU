import torch
import torch.nn as nn
import numpy as np

# ----------- FFT模块（Conv1d实现） -----------
class OneSidedComplexFFT(nn.Module):
    def __init__(self, fft_len):
        super().__init__()
        self.fft_len = fft_len
        self.fft_bins = fft_len  # 保留全部频率分量（64点）

        n = np.arange(self.fft_len)
        k = np.arange(self.fft_bins).reshape(-1, 1)
        cos_kernel = np.cos(2 * np.pi * k * n / self.fft_len)
        sin_kernel = -np.sin(2 * np.pi * k * n / self.fft_len)

        real_kernel = torch.tensor(cos_kernel, dtype=torch.float32).unsqueeze(1)
        imag_kernel = torch.tensor(sin_kernel, dtype=torch.float32).unsqueeze(1)

        self.real_conv = nn.Conv1d(1, self.fft_bins, kernel_size=self.fft_len, bias=False)
        self.imag_conv = nn.Conv1d(1, self.fft_bins, kernel_size=self.fft_len, bias=False)
        self.real_conv.weight.data = real_kernel
        self.imag_conv.weight.data = imag_kernel
        self.real_conv.weight.requires_grad = False
        self.imag_conv.weight.requires_grad = False

    def forward(self, x):
        # x: (batch, 1, fft_len)
        real_part = self.real_conv(x)
        imag_part = self.imag_conv(x)
        return torch.stack([real_part, imag_part], dim=-1)  # (batch, fft_bins, 2)

# ----------- 导频提取 -----------
def extract_pilots(fft_out):
    # fft_out: (batch, 64, 2)  # 2为实虚部
    pilot_indices = [11, 25, 39, 53]  # 对应-21,-7,+7,+21（FFT输出顺序，需确认）
    pilots = fft_out[:, pilot_indices, :]  # (batch, 4, 2)
    return pilots

# ----------- 信道估计（插值补全） -----------
def channel_estimation(pilots):
    # pilots: (batch, 4, 2)
    # 简单线性插值，实际可用更复杂方法
    batch = pilots.shape[0]
    h_est = torch.zeros((batch, 64, 2), dtype=pilots.dtype)
    pilot_pos = [11, 25, 39, 53]
    for i in range(3):
        start, end = pilot_pos[i], pilot_pos[i+1]
        for b in range(batch):
            h_est[b, start:end+1, :] = torch.linspace(0, 1, end-start+1).unsqueeze(-1) * pilots[b, i+1, :] + \
                                       torch.linspace(1, 0, end-start+1).unsqueeze(-1) * pilots[b, i, :]
    # 两端补齐
    h_est[:, :pilot_pos[0], :] = pilots[:, 0, :].unsqueeze(1)
    h_est[:, pilot_pos[-1]:, :] = pilots[:, -1, :].unsqueeze(1)
    return h_est

# ----------- 信道均衡 -----------
def channel_equalization(fft_out, h_est):
    # fft_out, h_est: (batch, 64, 2)
    # 复数除法：(a+jb)/(c+jd) = [(ac+bd) + j(bc-ad)] / (c^2+d^2)
    a, b = fft_out[..., 0], fft_out[..., 1]
    c, d = h_est[..., 0], h_est[..., 1]
    denom = c**2 + d**2 + 1e-8
    real = (a * c + b * d) / denom
    imag = (b * c - a * d) / denom
    return torch.stack([real, imag], dim=-1)  # (batch, 64, 2)

# ----------- 主流程示例 -----------
def process_ofdm_symbol(ofdm_samples):
    # ofdm_samples: (batch, 64), np.complex64
    x_real = torch.from_numpy(ofdm_samples.real).float().unsqueeze(1)  # (batch, 1, 64)
    x_imag = torch.from_numpy(ofdm_samples.imag).float().unsqueeze(1)
    x = x_real + 1j * x_imag

    # FFT
    fft_model = OneSidedComplexFFT(64)
    fft_out = fft_model(x_real)  # (batch, 64, 2)

    # 导频提取
    pilots = extract_pilots(fft_out)  # (batch, 4, 2)

    # 信道估计
    h_est = channel_estimation(pilots)  # (batch, 64, 2)

    # 信道均衡
    eq_out = channel_equalization(fft_out, h_est)  # (batch, 64, 2)

    return eq_out  # 均衡后的频域复数



class OFDMProcessModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fft_model = OneSidedComplexFFT(64)

    def forward(self, x_real):
        fft_out = self.fft_model(x_real)  # (batch, 64, 2)
        pilots = extract_pilots(fft_out)
        h_est = channel_estimation(pilots)
        eq_out = channel_equalization(fft_out, h_est)
        return eq_out

# 导出ONNX模型
if __name__ == "__main__":
    model = OFDMProcessModule()
    dummy_input = torch.randn(1, 1, 64)  # batch=1，实部输入
    torch.onnx.export(
        model,
        (dummy_input,),  # 注意这里加了括号，变成元组
        "ofdm_process.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=16,
    )