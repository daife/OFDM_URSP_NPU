import numpy as np
import onnxruntime as ort
import time

def numpy_fft(x):
    complex_x = x[0] + 1j * x[1]
    fft_out = np.fft.fft(complex_x)
    return np.stack([fft_out.real, fft_out.imag], axis=0)

def run_onnx(model_path, x):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    start = time.time()
    out = sess.run(None, {input_name: x})[0]
    elapsed = time.time() - start
    out = np.squeeze(out)
    if out.shape == (x.shape[0], 2, 256, 1):
        out = out[:, :, :, 0]
    elif out.shape == (x.shape[0], 256, 2):
        out = out.transpose(0, 2, 1)
    elif out.shape == (x.shape[0], 512):
        out = out.reshape(x.shape[0], 2, 256)
    return out, elapsed

if __name__ == "__main__":
    np.random.seed(42)
    x = np.random.randn(2, 256).astype(np.float32)
    x_batch = np.stack([x for _ in range(32)], axis=0)

    t0 = time.time()
    ref_out = np.stack([numpy_fft(xi) for xi in x_batch], axis=0)
    t_numpy = time.time() - t0

    out, t_onnx = run_onnx("fft256.onnx", x_batch)
    err = np.max(np.abs(ref_out - out), axis=(1, 2))

    print(f"numpy FFT 32个batch总时间: {t_numpy*1000:.2f} ms")
    print(f"fft256.onnx 32个batch总推理时间: {t_onnx*1000:.2f} ms")
    print(f"fft256.onnx每个batch最大误差: {err}")
