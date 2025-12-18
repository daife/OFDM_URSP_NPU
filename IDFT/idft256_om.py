import numpy as np
import acl
import time

ACL_MEMCPY_HOST_TO_DEVICE = 0
ACL_MEMCPY_DEVICE_TO_HOST = 1
ACL_MEM_MALLOC_NORMAL_ONLY = 0
ACL_SUCCESS = 0
BATCH_SIZE = 32  # 指定 .om 模型的 batch 数

def check_ret(msg, ret):
    if ret != ACL_SUCCESS:
        raise Exception(f"{msg} failed ret={ret}")

def run_acl_model(model_id, model_desc, input_data):
    # 输入数据如何得到：此处演示用随机浮点数据，shape=(BATCH_SIZE, 2, 256)，实部+虚部
    input_size = input_data.size * input_data.itemsize
    input_device, ret = acl.rt.malloc(input_size, ACL_MEM_MALLOC_NORMAL_ONLY); check_ret("acl.rt.malloc", ret)
    ret = acl.rt.memcpy(input_device, input_size, acl.util.numpy_to_ptr(input_data), input_size, ACL_MEMCPY_HOST_TO_DEVICE); check_ret("acl.rt.memcpy", ret)
    input_data_buffer = acl.create_data_buffer(input_device, input_size)
    input_dataset = acl.mdl.create_dataset(); _, ret = acl.mdl.add_dataset_buffer(input_dataset, input_data_buffer); check_ret("acl.mdl.add_dataset_buffer", ret)

    # 输出数据是怎样的：模型输出为 (batch, 2, 256) 的浮点数组，实部与虚部分别在 dim=1
    output_num = acl.mdl.get_num_outputs(model_desc)
    output_dataset = acl.mdl.create_dataset()
    output_buffers = []
    host_outputs = []
    for i in range(output_num):
        output_size = acl.mdl.get_output_size_by_index(model_desc, i)
        output_device, ret = acl.rt.malloc(output_size, ACL_MEM_MALLOC_NORMAL_ONLY); check_ret("acl.rt.malloc", ret)
        output_data_buffer = acl.create_data_buffer(output_device, output_size)
        _, ret = acl.mdl.add_dataset_buffer(output_dataset, output_data_buffer); check_ret("acl.mdl.add_dataset_buffer", ret)
        output_buffers.append((output_device, output_size, output_data_buffer))
        host_outputs.append(np.empty((BATCH_SIZE, 2, 256), dtype=np.float32))

    # 如何执行模型推理：使用 acl.mdl.execute 将输入/输出 dataset 传入模型执行
    start = time.time()
    ret = acl.mdl.execute(model_id, input_dataset, output_dataset); check_ret("acl.mdl.execute", ret)
    elapsed = time.time() - start

    # 拷回输出
    for i, (output_device, output_size, _) in enumerate(output_buffers):
        ret = acl.rt.memcpy(acl.util.numpy_to_ptr(host_outputs[i]), output_size, output_device, output_size, ACL_MEMCPY_DEVICE_TO_HOST); check_ret("acl.rt.memcpy D2H", ret)
    out = host_outputs[0] if output_num == 1 else np.stack(host_outputs, axis=0)

    # 释放资源
    ret = acl.rt.free(input_device); check_ret("acl.rt.free input", ret)
    for output_device, _, output_data_buffer in output_buffers:
        ret = acl.rt.free(output_device); check_ret("acl.rt.free output", ret)
        ret = acl.destroy_data_buffer(output_data_buffer); check_ret("acl.destroy_data_buffer out", ret)
    ret = acl.mdl.destroy_dataset(input_dataset); check_ret("acl.mdl.destroy_dataset in", ret)
    ret = acl.mdl.destroy_dataset(output_dataset); check_ret("acl.mdl.destroy_dataset out", ret)
    ret = acl.destroy_data_buffer(input_data_buffer); check_ret("acl.destroy_data_buffer in", ret)
    return out, elapsed

if __name__ == "__main__":
    np.random.seed(42)
    x = np.random.randn(2, 256).astype(np.float32)
    x_batch = np.stack([x for _ in range(BATCH_SIZE)], axis=0)

    ret = acl.init(); check_ret("acl.init", ret)
    dev_id = 0
    ret = acl.rt.set_device(dev_id); check_ret("acl.rt.set_device", ret)
    context, ret = acl.rt.create_context(dev_id); check_ret("acl.rt.create_context", ret)
    stream, ret = acl.rt.create_stream(); check_ret("acl.rt.create_stream", ret)

    model_id, ret = acl.mdl.load_from_file("idft256.om"); check_ret("acl.mdl.load_from_file", ret)
    model_desc = acl.mdl.create_desc(); ret = acl.mdl.get_desc(model_desc, model_id); check_ret("acl.mdl.get_desc", ret)

    out, t = run_acl_model(model_id, model_desc, x_batch)
    print(f"idft256.om {BATCH_SIZE} 个 batch 总推理时间: {t*1000:.2f} ms")
    print("输出形状:", out.shape)

    ret = acl.mdl.unload(model_id); check_ret("acl.mdl.unload", ret)
    ret = acl.mdl.destroy_desc(model_desc); check_ret("acl.mdl.destroy_desc", ret)
    ret = acl.rt.destroy_stream(stream); check_ret("acl.rt.destroy_stream", ret)
    ret = acl.rt.destroy_context(context); check_ret("acl.rt.destroy_context", ret)
    ret = acl.rt.reset_device(dev_id); check_ret("acl.rt.reset_device", ret)
    ret = acl.finalize(); check_ret("acl.finalize", ret)
