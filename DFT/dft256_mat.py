import torch
import torch.nn as nn
import numpy as np

class MatrixDFT256(nn.Module):
    def __init__(self):
        super().__init__()
        N = 256
        n = np.arange(N)
        k = np.arange(N).reshape(-1, 1)
        W_real = np.cos(2 * np.pi * k * n / N)
        W_imag = -np.sin(2 * np.pi * k * n / N)
        self.register_buffer("W_real", torch.tensor(W_real, dtype=torch.float32))
        self.register_buffer("W_imag", torch.tensor(W_imag, dtype=torch.float32))

    def forward(self, x):
        # x: (batch, 2, 256)
        x_real = x[:, 0, :]  # (batch, 256)
        x_imag = x[:, 1, :]  # (batch, 256)
        # DFT: Y = W * x
        real_out = torch.matmul(x_real, self.W_real.t()) - torch.matmul(x_imag, self.W_imag.t())
        imag_out = torch.matmul(x_real, self.W_imag.t()) + torch.matmul(x_imag, self.W_real.t())
        out = torch.stack([real_out, imag_out], dim=1)  # (batch, 2, 256)
        return out

if __name__ == "__main__":
    model = MatrixDFT256()
    dummy_input = torch.randn(1024, 2, 256)  # 修改为1024个batch
    torch.onnx.export(
        model,
        (dummy_input,),
        "dft256_mat_1024.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
    )
