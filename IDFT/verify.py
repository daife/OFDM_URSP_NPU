import numpy as np
import torch
import onnxruntime as ort
import time

def numpy_ifft(x):
    # x: (2, 256), x[0]为实部, x[1]为虚部
    complex_x = x[0] + 1j * x[1]
    ifft_out = np.fft.ifft(complex_x)
    return np.stack([ifft_out.real, ifft_out.imag], axis=0)  # (2, 256)

def run_onnx(model_path, x):
    # x: (batch, 2, 256), float32
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    start = time.time()
    out = sess.run(None, {input_name: x})[0]
    elapsed = time.time() - start
    # 处理输出shape，确保为(batch, 2, 256)
    out = np.squeeze(out)
    if out.shape == (x.shape[0], 2, 256, 1):
        out = out[:, :, :, 0]
    elif out.shape == (x.shape[0], 256, 2):
        out = out.transpose(0, 2, 1)
    elif out.shape == (x.shape[0], 512):
        out = out.reshape(x.shape[0], 2, 256)
    # 其他情况直接返回
    return out, elapsed  # (batch, 2, 256), float

def main():
    np.random.seed(42)
    x = np.random.randn(2, 256).astype(np.float32)
    x_batch = np.stack([x for _ in range(32)], axis=0)  # (32, 2, 256)

    # numpy IFFT
    t0 = time.time()
    ref_out = np.stack([numpy_ifft(xi) for xi in x_batch], axis=0)  # (32, 2, 256)
    t_numpy = time.time() - t0

    # idft256.onnx
    out1, t_onnx1 = run_onnx("idft256_noprefix.onnx", x_batch)

    # idft256_mat.onnx
    out2, t_onnx2 = run_onnx("idft256_mat_noprefix.onnx", x_batch)

    # 误差
    err1 = np.max(np.abs(ref_out - out1), axis=(1,2))  # 每个batch最大误差
    err2 = np.max(np.abs(ref_out - out2), axis=(1,2))

    print(f"numpy IFFT 32个batch总时间: {t_numpy*1000:.2f} ms")
    print(f"idft256.onnx 32个batch总推理时间: {t_onnx1*1000:.2f} ms")
    print(f"idft256_mat.onnx 32个batch总推理时间: {t_onnx2*1000:.2f} ms")
    print(f"idft256.onnx每个batch最大误差: {err1}")
    print(f"idft256_mat.onnx每个batch最大误差: {err2}")

if __name__ == "__main__":
    main()
