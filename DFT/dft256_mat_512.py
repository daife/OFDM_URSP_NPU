import torch
import torch.nn as nn
import numpy as np

class MatrixDFT256_512(nn.Module):
    def __init__(self):
        super().__init__()
        N = 256
        n = np.arange(N)
        k = np.arange(N).reshape(-1, 1)
        W_real = np.cos(2 * np.pi * k * n / N)  # (256, 256)
        W_imag = -np.sin(2 * np.pi * k * n / N)  # (256, 256)

        # 构造512x512完整DFT矩阵
        # [W_real  -W_imag]
        # [W_imag   W_real]
        W = np.zeros((2*N, 2*N), dtype=np.float32)
        W[:N, :N] = W_real      # 左上角：W_real
        W[:N, N:] = -W_imag     # 右上角：-W_imag
        W[N:, :N] = W_imag      # 左下角：W_imag
        W[N:, N:] = W_real      # 右下角：W_real

        self.register_buffer("W", torch.tensor(W, dtype=torch.float32))  # (512, 512)

    def forward(self, x):
        # x: (batch, 2, 256)
        batch = x.shape[0]
        # 构造 (batch, 512) 输入
        X = torch.zeros((batch, 2*256), dtype=x.dtype, device=x.device)
        X[:, :256] = x[:, 0, :]  # 实部
        X[:, 256:] = x[:, 1, :]  # 虚部
        # 旋转因子矩阵乘法
        Y = torch.matmul(X, self.W.t())  # (batch, 512)
        # 输出 (batch, 2, 256)
        out = torch.stack([Y[:, :256], Y[:, 256:]], dim=1)  # (batch, 2, 256)
        return out

if __name__ == "__main__":
    model = MatrixDFT256_512()
    dummy_input = torch.randn(32, 2, 256)
    torch.onnx.export(
        model,
        (dummy_input,),
        "dft256_mat_512.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
    )
