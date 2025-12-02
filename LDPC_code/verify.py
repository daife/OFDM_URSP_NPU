import numpy as np
import torch
import onnxruntime as ort
import time

def numpy_ldpc_decode(x):
    # x: (1944,) 伪码字，简单硬判决作为参考
    # 实际应调用LDPC解码库，这里仅做示例
    return (x > 0.5).astype(np.float32)

def run_onnx(model_path, x):
    # x: (batch, 1944), float32
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    start = time.time()
    out = sess.run(None, {input_name: x})[0]
    elapsed = time.time() - start
    # 输出shape处理
    out = np.squeeze(out)
    return out, elapsed

def main():
    np.random.seed(42)
    batch_size = 32
    K = 1920
    CRC = 24
    N = 22 * 384
    x = np.random.randint(0, 2, (K + CRC)).astype(np.float32)
    x_batch = np.stack([x for _ in range(batch_size)], axis=0)  # (32, 1944)

    # numpy LDPC decode
    t0 = time.time()
    ref_out = np.stack([numpy_ldpc_decode(xi) for xi in x_batch], axis=0)  # (32, 1944)
    t_numpy = time.time() - t0

    # ldpc_bg1.onnx
    out1, t_onnx1 = run_onnx("ldpc_bg1.onnx", x_batch)

    # 误差
    err1 = np.max(np.abs(ref_out - out1), axis=1)  # 每个batch最大误差

    print(f"numpy LDPC 32个batch总时间: {t_numpy*1000:.2f} ms")
    print(f"ldpc_bg1.onnx 32个batch总推理时间: {t_onnx1*1000:.2f} ms")
    print(f"ldpc_bg1.onnx每个batch最大误差: {err1}")

if __name__ == "__main__":
    main()
