import numpy as np
import onnxruntime as ort

def qam64_mod(bits):
    bits = bits.reshape(-1, 6)
    mapping = np.array([-7, -5, -3, -1, +1, +3, +5, +7])
    def bits2int(b): return b[0]*32 + b[1]*16 + b[2]*8 + b[3]*4 + b[4]*2 + b[5]
    ints = np.array([bits2int(b) for b in bits])
    real = mapping[(ints >> 3) & 0b111]
    imag = mapping[ints & 0b111]
    symbols = real + 1j * imag
    return symbols

batch = 1
n_data = 48
n_total = 64
pilot_indices = [11, 25, 39, 53]
data_indices = [i for i in range(n_total) if i not in pilot_indices][:n_data]

bits = np.random.randint(0, 2, (batch, n_data * 6), dtype=np.uint8)
mod_data = qam64_mod(bits[0])  # (48,)

# 构造频域数据，插入导频
freq = np.zeros((n_total,), dtype=np.complex64)
freq[data_indices] = mod_data
freq[pilot_indices] = 1.0 + 0j  # 导频

# numpy IFFT
ifft_np = np.fft.ifft(freq, n=n_total)
ifft_np_real = np.real(ifft_np).reshape(1, 1, n_total).astype(np.float32)  # (1, 1, 64)

# ONNX模型推理
freq_onnx = np.zeros((1, 2, n_total), dtype=np.float32)
freq_onnx[0, 0, :] = np.real(freq)
freq_onnx[0, 1, :] = np.imag(freq)

gen_sess = ort.InferenceSession("ofdm_generator.onnx", providers=['CPUExecutionProvider'])
ofdm_time_onnx = gen_sess.run(None, {"input": freq_onnx})[0]  # (1, 1, 64)

# 对比
mse = np.mean((ifft_np_real.flatten() - ofdm_time_onnx.flatten())**2)
print(f"均方误差: {mse:.8f}")
print("numpy IFFT输出（前10）:", ifft_np_real.flatten()[:10])
print("ONNX模型输出（前10）:", ofdm_time_onnx.flatten()[:10])