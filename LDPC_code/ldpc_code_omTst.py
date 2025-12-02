import numpy as np
import acl
import time

ACL_MEMCPY_HOST_TO_DEVICE = 0
ACL_MEMCPY_DEVICE_TO_HOST = 1
ACL_MEM_MALLOC_NORMAL_ONLY = 0
ACL_SUCCESS = 0

def check_ret(msg, ret):
    if ret != ACL_SUCCESS:
        raise Exception(f"{msg} failed ret={ret}")

def run_acl_model(model_id, model_desc, input_data):
    # 输入数据
    input_size = input_data.size * input_data.itemsize
    input_device, ret = acl.rt.malloc(input_size, ACL_MEM_MALLOC_NORMAL_ONLY)
    check_ret("acl.rt.malloc", ret)
    input_bytes = input_data.tobytes()
    input_ptr = acl.util.bytes_to_ptr(input_bytes)
    ret = acl.rt.memcpy(input_device, input_size, input_ptr, input_size, ACL_MEMCPY_HOST_TO_DEVICE)
    check_ret("acl.rt.memcpy", ret)
    input_data_buffer = acl.create_data_buffer(input_device, input_size)
    input_dataset = acl.mdl.create_dataset()
    _, ret = acl.mdl.add_dataset_buffer(input_dataset, input_data_buffer)
    check_ret("acl.mdl.add_dataset_buffer", ret)

    # 输出数据
    output_num = acl.mdl.get_num_outputs(model_desc)
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

    # 推理计时
    start = time.time()
    ret = acl.mdl.execute(model_id, input_dataset, output_dataset)
    check_ret("acl.mdl.execute", ret)
    elapsed = time.time() - start

    # 释放资源
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
    return elapsed

if __name__ == "__main__":
    np.random.seed(42)
    # BG1, Z=384, N=22*384=8448, K=8448, 输出为25344
    batch_size = 32
    K = 8448
    output_len = 25344
    # 输入为INT32，shape为(batch_size, K)
    x = np.random.randint(0, 2, (K,)).astype(np.int32)
    x_batch = np.stack([x for _ in range(batch_size)], axis=0)  # (32, 8448)

    # 初始化ACL环境和设备
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
    model_id, ret = acl.mdl.load_from_file("ldpc_bg1.om")
    check_ret("acl.mdl.load_from_file ldpc_bg1.om", ret)
    model_desc = acl.mdl.create_desc()
    ret = acl.mdl.get_desc(model_desc, model_id)
    check_ret("acl.mdl.get_desc ldpc_bg1.om", ret)

    # 推理
    t1 = run_acl_model(model_id, model_desc, x_batch)
    print(f"ldpc_bg1.om 32个batch总推理时间: {t1*1000:.2f} ms")
    print(f"输入shape: {x_batch.shape}, dtype: {x_batch.dtype}")
    print(f"输出应为shape: ({batch_size}, {output_len}), dtype: np.int32")

    # 卸载模型和销毁desc
    ret = acl.mdl.unload(model_id)
    check_ret("acl.mdl.unload ldpc_bg1.om", ret)
    ret = acl.mdl.destroy_desc(model_desc)
    check_ret("acl.mdl.destroy_desc ldpc_bg1.om", ret)

    # 统一释放ACL环境和设备
    ret = acl.rt.destroy_stream(stream)
    check_ret("acl.rt.destroy_stream", ret)
    ret = acl.rt.destroy_context(context)
    check_ret("acl.rt.destroy_context", ret)
    ret = acl.rt.reset_device(dev_id)
    check_ret("acl.rt.reset_device", ret)
    ret = acl.finalize()
    check_ret("acl.finalize", ret)