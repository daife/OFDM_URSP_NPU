import torch
import torch.nn as nn
import numpy as np

class FFT256(nn.Module):
    def __init__(self):
        super().__init__()
        self.N = 256

    def forward(self, x):
        # x: (batch, 2, 256)
        x_real = x[:, 0, :]  # (batch, 256)
        x_imag = x[:, 1, :]  # (batch, 256)
        real, imag = self.fft(x_real, x_imag)
        out = torch.stack([real, imag], dim=1)  # (batch, 2, 256)
        return out

    def fft(self, x_real, x_imag):
        # x_real, x_imag: (batch, N)
        N = x_real.shape[-1]
        if N == 1:
            return x_real, x_imag
        # 分治递归
        even_real, even_imag = self.fft(x_real[..., ::2], x_imag[..., ::2])
        odd_real, odd_imag = self.fft(x_real[..., 1::2], x_imag[..., 1::2])
        k = torch.arange(N // 2, device=x_real.device, dtype=torch.float32).reshape(1, -1)
        angle = -2 * torch.pi * k / float(N)
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        # twiddle: (cos - j*sin)
        twiddle_real = cos
        twiddle_imag = sin
        # 旋转因子乘以odd部分
        t_real = twiddle_real * odd_real - twiddle_imag * odd_imag
        t_imag = twiddle_real * odd_imag + twiddle_imag * odd_real
        left_real = even_real + t_real
        left_imag = even_imag + t_imag
        right_real = even_real - t_real
        right_imag = even_imag - t_imag
        real = torch.cat([left_real, right_real], dim=-1)
        imag = torch.cat([left_imag, right_imag], dim=-1)
        return real, imag

if __name__ == "__main__":
    model = FFT256()
    dummy_input = torch.randn(32, 2, 256)
    torch.onnx.export(
        model,
        (dummy_input,),
        "fft256.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
    )
