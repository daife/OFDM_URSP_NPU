import numpy as np
import torch
import onnxruntime as ort
import time

def numpy_fft(x):
    # x: (2, 256), x[0]为实部, x[1]为虚部
    complex_x = x[0] + 1j * x[1]
    fft_out = np.fft.fft(complex_x)
    return np.stack([fft_out.real, fft_out.imag], axis=0)  # (2, 256)

def run_onnx(model_path, x):
    # x: (1, 2, 256), float32
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    start = time.time()
    out = sess.run(None, {input_name: x})[0]
    elapsed = time.time() - start
    # 处理输出shape，确保为(2, 256)
    out = out[0]  # 去掉batch维
    out = np.squeeze(out)
    if out.shape == (512,):
        out = out.reshape(2, 256)
    elif out.shape == (256, 2):
        out = out.T
    elif out.shape == (2, 256, 1):
        out = out[:, :, 0]
    # 其他情况直接返回
    return out, elapsed  # (2, 256), float

def main():
    np.random.seed(42)
    x = np.random.randn(2, 256).astype(np.float32)
    x_batch = x[np.newaxis, ...]  # (1, 2, 256)

    # numpy FFT
    t0 = time.time()
    ref_out = numpy_fft(x)
    t_numpy = time.time() - t0

    # dft256.onnx
    out1, t_onnx1 = run_onnx("dft256_noprefix.onnx", x_batch)

    # dft256_mat.onnx
    out2, t_onnx2 = run_onnx("dft256_mat_noprefix.onnx", x_batch)

    # 误差
    err1 = np.max(np.abs(ref_out - out1))
    err2 = np.max(np.abs(ref_out - out2))

    print(f"numpy FFT time: {t_numpy*1000:.2f} ms")
    print(f"dft256.onnx推理时间: {t_onnx1*1000:.2f} ms, 最大误差: {err1:.6e}")
    print(f"dft256_mat.onnx推理时间: {t_onnx2*1000:.2f} ms, 最大误差: {err2:.6e}")

if __name__ == "__main__":
    main()
