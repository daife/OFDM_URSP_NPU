import numpy as np
import acl
import time

# 常量定义（如无constant.py可直接补充）
ACL_MEMCPY_HOST_TO_DEVICE = 0
ACL_MEMCPY_DEVICE_TO_HOST = 1
ACL_MEM_MALLOC_NORMAL_ONLY = 0
ACL_SUCCESS = 0

def check_ret(message, ret):
    if ret != ACL_SUCCESS:
        raise Exception(f"{message} failed ret={ret}")

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
data_subcarrier_ranges = [
    range(-26, -21),
    range(-20, -7),
    range(-6, 0),
    range(1, 7),
    range(8, 21),
    range(22, 27),
]
def subcarrier_num_to_index(num):
    return num + 32

data_indices = []
for r in data_subcarrier_ranges:
    data_indices.extend([subcarrier_num_to_index(i) for i in r])
data_indices = [i for i in data_indices if i not in pilot_indices and i != 32]

bits = np.random.randint(0, 2, (batch, len(data_indices) * 6), dtype=np.uint8)
mod_data = qam64_mod(bits[0])  # (48,)

freq = np.zeros((batch, 2, n_total), dtype=np.float32)
freq[0, 0, data_indices] = np.real(mod_data)
freq[0, 1, data_indices] = np.imag(mod_data)

# 初始化ACL
ret = acl.init()
check_ret("acl.init", ret)
dev_id = 0
ret = acl.rt.set_device(dev_id)
check_ret("acl.rt.set_device", ret)
context, ret = acl.rt.create_context(dev_id)
check_ret("acl.rt.create_context", ret)
stream, ret = acl.rt.create_stream()
check_ret("acl.rt.create_stream", ret)

# 加载模型
def load_model(model_path):
    model_id, ret = acl.mdl.load_from_file(model_path)
    check_ret("acl.mdl.load_from_file", ret)
    model_desc = acl.mdl.create_desc()
    ret = acl.mdl.get_desc(model_desc, model_id)
    check_ret("acl.mdl.get_desc", ret)
    input_num = acl.mdl.get_num_inputs(model_desc)
    output_num = acl.mdl.get_num_outputs(model_desc)
    return model_id, model_desc, input_num, output_num

def run_model(model_id, model_desc, input_num, output_num, input_data):
    # 输入数据集
    input_dataset = acl.mdl.create_dataset()
    input_size = input_data.size * input_data.itemsize
    input_device, ret = acl.rt.malloc(input_size, ACL_MEM_MALLOC_NORMAL_ONLY)
    check_ret("acl.rt.malloc", ret)
    # host to device
    input_bytes = input_data.tobytes()
    input_ptr = acl.util.bytes_to_ptr(input_bytes)
    ret = acl.rt.memcpy(input_device, input_size, input_ptr, input_size, ACL_MEMCPY_HOST_TO_DEVICE)
    check_ret("acl.rt.memcpy", ret)
    input_data_buffer = acl.create_data_buffer(input_device, input_size)
    _, ret = acl.mdl.add_dataset_buffer(input_dataset, input_data_buffer)
    check_ret("acl.mdl.add_dataset_buffer", ret)

    # 输出数据集
    output_dataset = acl.mdl.create_dataset()
    output_buffers = []
    for i in range(output_num):
        output_size = acl.mdl.get_output_size_by_index(model_desc, i)
        output_device, ret = acl.rt.malloc(output_size, ACL_MEM_MALLOC_NORMAL_ONLY)
        check_ret("acl.rt.malloc", ret)
        output_data_buffer = acl.create_data_buffer(output_device, output_size)
        _, ret = acl.mdl.add_dataset_buffer(output_dataset, output_data_buffer)
        check_ret("acl.mdl.add_dataset_buffer", ret)
        output_buffers.append((output_device, output_size, output_data_buffer))

    # 推理
    ret = acl.mdl.execute(model_id, input_dataset, output_dataset)
    check_ret("acl.mdl.execute", ret)

    # 获取输出
    output_host = []
    for output_device, output_size, output_data_buffer in output_buffers:
        output_host_bytes = bytes(output_size)
        # 直接使用 bytes 类型
        output_host_ptr = acl.util.bytes_to_ptr(output_host_bytes)
        ret = acl.rt.memcpy(output_host_ptr, output_size, output_device, output_size, ACL_MEMCPY_DEVICE_TO_HOST)
        check_ret("acl.rt.memcpy", ret)
        output_host.append(np.frombuffer(output_host_bytes, dtype=np.float32))

    # 释放device内存和数据集
    ret = acl.rt.free(input_device)
    check_ret("acl.rt.free", ret)
    for output_device, _, output_data_buffer in output_buffers:
        ret = acl.rt.free(output_device)
        check_ret("acl.rt.free", ret)
        ret = acl.destroy_data_buffer(output_data_buffer)
        check_ret("acl.destroy_data_buffer", ret)
    ret = acl.mdl.destroy_dataset(input_dataset)
    check_ret("acl.mdl.destroy_dataset", ret)
    ret = acl.mdl.destroy_dataset(output_dataset)
    check_ret("acl.mdl.destroy_dataset", ret)
    ret = acl.destroy_data_buffer(input_data_buffer)
    check_ret("acl.destroy_data_buffer", ret)
    return output_host[0] if len(output_host) == 1 else output_host

# 推理
gen_model_id, gen_model_desc, gen_input_num, gen_output_num = load_model("ofdm_generator.om")
proc_model_id, proc_model_desc, proc_input_num, proc_output_num = load_model("ofdm_process.om")

start_gen = time.time()
ofdm_time = run_model(gen_model_id, gen_model_desc, gen_input_num, gen_output_num, freq)  # (1, 2, 64)
end_gen = time.time()
print(f"生成器模型推理耗时: {end_gen - start_gen:.6f} 秒")

ofdm_time_input = ofdm_time.astype(np.float32)
start_proc = time.time()
eq_out = run_model(proc_model_id, proc_model_desc, proc_input_num, proc_output_num, ofdm_time_input)  # (1, 2, 64)
end_proc = time.time()
print(f"Process模型推理耗时: {end_proc - start_proc:.6f} 秒")

# 修复 eq_out 维度问题
eq_out = np.array(eq_out).reshape((batch, 2, n_total))
eq_data = eq_out[:, :, data_indices]  # (1, 2, 48)
eq_data_complex = eq_data[0, 0, :] + 1j * eq_data[0, 1, :]

mod_data_flat = mod_data.flatten()
eq_data_flat = eq_data_complex.flatten()
mse = np.mean(np.abs(mod_data_flat - eq_data_flat)**2)
print(f"均方误差: {mse:.6f}")

print("原始调制数据（前48）:", mod_data_flat[:48])
print("解调输出（前48）:", eq_data_flat[:48])

# 释放资源
ret = acl.mdl.unload(gen_model_id)
check_ret("acl.mdl.unload", ret)
ret = acl.mdl.destroy_desc(gen_model_desc)
check_ret("acl.mdl.destroy_desc", ret)
ret = acl.mdl.unload(proc_model_id)
check_ret("acl.mdl.unload", ret)
ret = acl.mdl.destroy_desc(proc_model_desc)
check_ret("acl.mdl.destroy_desc", ret)
ret = acl.rt.destroy_stream(stream)
check_ret("acl.rt.destroy_stream", ret)
ret = acl.rt.destroy_context(context)
check_ret("acl.rt.destroy_context", ret)
ret = acl.rt.reset_device(dev_id)
check_ret("acl.rt.reset_device", ret)
ret = acl.finalize()
check_ret("acl.finalize", ret)