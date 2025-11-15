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
        # x: (batch, 2, 64)
        x_real = x[:, 0:1, :]  # (batch, 1, 64)
        x_imag = x[:, 1:2, :]
        real_part = self.real_conv(x_real) - self.imag_conv(x_imag)  # (batch, 64, 1)
        imag_part = self.real_conv(x_imag) + self.imag_conv(x_real)  # (batch, 64, 1)
        real_part = real_part.permute(0, 2, 1)  # (batch, 1, 64)
        imag_part = imag_part.permute(0, 2, 1)  # (batch, 1, 64)
        out = torch.cat([real_part, imag_part], dim=1)  # (batch, 2, 64)
        return out  # (batch, 2, 64)

# ----------- 导频提取 -----------
def extract_pilots(fft_out):
    # fft_out: (batch, 2, 64)
    pilot_indices = [11, 25, 39, 53]
    pilots = fft_out[:, :, pilot_indices]  # (batch, 2, 4)
    pilots = pilots.permute(0, 2, 1)  # (batch, 4, 2)
    return pilots

# ----------- 信道估计（插值补全） -----------
def channel_estimation(pilots):
    # pilots: (batch, 4, 2)
    batch = pilots.shape[0]
    h_est = torch.zeros((batch, 64, 2), dtype=pilots.dtype, device=pilots.device)
    pilot_pos = [11, 25, 39, 53]

    for i in range(3):
        start, end = pilot_pos[i], pilot_pos[i+1]
        alpha = torch.linspace(0, 1, end - start + 1, device=pilots.device).unsqueeze(0).unsqueeze(-1)
        h_est[:, start:end+1, :] = alpha * pilots[:, i+1, :].unsqueeze(1) + (1 - alpha) * pilots[:, i, :].unsqueeze(1)

    h_est[:, :pilot_pos[0], :] = pilots[:, 0, :].unsqueeze(1)
    h_est[:, pilot_pos[-1]:, :] = pilots[:, -1, :].unsqueeze(1)
    h_est = h_est.permute(0, 2, 1)  # (batch, 2, 64)
    return h_est

# ----------- 信道均衡 -----------
def channel_equalization(fft_out, h_est):
    # fft_out, h_est: (batch, 2, 64)
    a, b = fft_out[:, 0, :], fft_out[:, 1, :]
    c, d = h_est[:, 0, :], h_est[:, 1, :]
    denom = c**2 + d**2 + 1e-8
    real = (a * c + b * d) / denom
    imag = (b * c - a * d) / denom
    eq = torch.stack([real, imag], dim=1)  # (batch, 2, 64)
    return eq

# ----------- 主流程示例 -----------
def process_ofdm_symbol(ofdm_samples):
    # ofdm_samples: (batch, 2, 64), np.float32
    x = torch.from_numpy(ofdm_samples).float()
    fft_model = OneSidedComplexFFT(64)
    fft_out = fft_model(x)  # (batch, 2, 64)

    pilots = extract_pilots(fft_out)  # (batch, 4, 2)
    h_est = channel_estimation(pilots)  # (batch, 2, 64)
    eq_out = channel_equalization(fft_out, h_est)  # (batch, 2, 64)

    return eq_out  # 均衡后的频域复数

class OFDMProcessModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fft_model = OneSidedComplexFFT(64)

    def forward(self, x):
        # x: (batch, 2, 64)
        fft_out = self.fft_model(x)  # (batch, 2, 64)
        pilots = extract_pilots(fft_out)
        h_est = channel_estimation(pilots)
        eq_out = channel_equalization(fft_out, h_est)
        return eq_out  # (batch, 2, 64)

# 导出ONNX模型
if __name__ == "__main__":
    model = OFDMProcessModule()
    dummy_input = torch.randn(1, 2, 64)  # batch=1，实部输入
    torch.onnx.export(
        model,
        (dummy_input,),
        "ofdm_process.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
    )