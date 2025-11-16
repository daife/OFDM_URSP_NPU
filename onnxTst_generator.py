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

# OFDM标准数据子载波区间
data_subcarrier_ranges = [
    range(-26, -21),  # -26 ~ -22
    range(-20, -7),   # -20 ~ -8
    range(-6, 0),     # -6 ~ -1
    range(1, 7),      # +1 ~ +6
    range(8, 21),     # +8 ~ +20
    range(22, 27),    # +22 ~ +26
]
# 将编号转换为索引（0为DC，编号-32~+31对应索引0~63）
def subcarrier_num_to_index(num):
    return num + 32

data_indices = []
for r in data_subcarrier_ranges:
    data_indices.extend([subcarrier_num_to_index(i) for i in r])
# 排除导频和DC
data_indices = [i for i in data_indices if i not in pilot_indices and i != 32]

bits = np.random.randint(0, 2, (batch, len(data_indices) * 6), dtype=np.uint8)
mod_data = qam64_mod(bits[0])  # (48,)

freq = np.zeros((n_total,), dtype=np.complex64)
freq[data_indices] = mod_data
freq[pilot_indices] = 1.0 + 0j  # 导频

# numpy IFFT
ifft_np = np.fft.ifft(freq, n=n_total)
ifft_np_real = np.real(ifft_np).reshape(1, n_total)
ifft_np_imag = np.imag(ifft_np).reshape(1, n_total)
ifft_np_full = np.stack([ifft_np_real, ifft_np_imag], axis=1).astype(np.float32)  # (1, 2, 64)

# ONNX模型推理
freq_onnx = np.zeros((1, 2, n_total), dtype=np.float32)
freq_onnx[0, 0, :] = np.real(freq)
freq_onnx[0, 1, :] = np.imag(freq)

gen_sess = ort.InferenceSession("ofdm_generator_noprefix_sim.onnx", providers=['CPUExecutionProvider'])
ofdm_time_onnx = gen_sess.run(None, {"input": freq_onnx})[0]  # (1, 2, 64)

# 对比
mse = np.mean((ifft_np_full.flatten() - ofdm_time_onnx.flatten())**2)
print(f"均方误差: {mse:.8f}")
print("numpy IFFT输出（前10）:", ifft_np_full.flatten()[:10])
print("ONNX模型输出（前10）:", ofdm_time_onnx.flatten()[:10])