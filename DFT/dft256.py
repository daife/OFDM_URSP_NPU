import torch
import torch.nn as nn
import numpy as np

class ComplexDFT256(nn.Module):
    def __init__(self):
        super().__init__()
        fft_len = 256
        n = np.arange(fft_len)
        k = np.arange(fft_len).reshape(-1, 1)
        cos_kernel = np.cos(2 * np.pi * k * n / fft_len)
        sin_kernel = -np.sin(2 * np.pi * k * n / fft_len)

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
        x_imag = x[:, 1:2, :]  # (batch, 1, 256)
        real_part = self.real_conv(x_real) - self.imag_conv(x_imag)  # (batch, 256, 1)
        imag_part = self.real_conv(x_imag) + self.imag_conv(x_real)  # (batch, 256, 1)
        # 输出直接为(batch, 2, 256)，无需转置
        out = torch.cat([real_part, imag_part], dim=1)  # (batch, 2, 256)
        return out

# ONNX导出
if __name__ == "__main__":
    model = ComplexDFT256()
    dummy_input = torch.randn(1, 2, 256)
    torch.onnx.export(
        model,
        (dummy_input,),
        "dft256.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
    )