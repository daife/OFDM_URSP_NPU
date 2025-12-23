import numpy as np
import acl
import time
import ctypes

# 宏定义：.om 模型编译时的 batch 数,注意这个是编译静态图时就已经确定了，该数字要换模型文件
MODEL_BATCH = 1192
ACL_MEM_MALLOC_NORMAL_ONLY = 0
ACL_MEMCPY_HOST_TO_DEVICE = 0
ACL_MEMCPY_DEVICE_TO_HOST = 1
ACL_SUCCESS = 0

def check_ret(msg, ret):
    if ret != ACL_SUCCESS:
        raise RuntimeError(f"{msg} failed ret={ret}")

def create_io_resources(model_desc, input_shape):
    input_size = int(np.prod(input_shape) * np.dtype(np.float32).itemsize)
    input_device, ret = acl.rt.malloc(input_size, ACL_MEM_MALLOC_NORMAL_ONLY); check_ret("acl.rt.malloc input", ret)
    input_buf = acl.create_data_buffer(input_device, input_size)
    input_ds = acl.mdl.create_dataset()
    _, ret = acl.mdl.add_dataset_buffer(input_ds, input_buf); check_ret("acl.mdl.add_dataset_buffer input", ret)

    out_num = acl.mdl.get_num_outputs(model_desc)
    output_ds = acl.mdl.create_dataset()
    out_dev, out_sizes, out_bufs = [], [], []
    for i in range(out_num):
        size_i = acl.mdl.get_output_size_by_index(model_desc, i)
        dev_i, ret = acl.rt.malloc(size_i, ACL_MEM_MALLOC_NORMAL_ONLY); check_ret("acl.rt.malloc output", ret)
        buf_i = acl.create_data_buffer(dev_i, size_i)
        _, ret = acl.mdl.add_dataset_buffer(output_ds, buf_i); check_ret("acl.mdl.add_dataset_buffer output", ret)
        out_dev.append(dev_i); out_sizes.append(size_i); out_bufs.append(buf_i)

    return {
        "input_device": input_device,
        "input_size": input_size,
        "input_buf": input_buf,
        "input_ds": input_ds,
        "out_dev": out_dev,
        "out_sizes": out_sizes,
        "out_bufs": out_bufs,
        "output_ds": output_ds,
    }

def destroy_io_resources(io_res):
    ret = acl.rt.free(io_res["input_device"]); check_ret("acl.rt.free input", ret)
    ret = acl.destroy_data_buffer(io_res["input_buf"]); check_ret("acl.destroy_data_buffer input", ret)
    ret = acl.mdl.destroy_dataset(io_res["input_ds"]); check_ret("acl.mdl.destroy_dataset input", ret)
    for dev, buf in zip(io_res["out_dev"], io_res["out_bufs"]):
        ret = acl.rt.free(dev); check_ret("acl.rt.free output", ret)
        ret = acl.destroy_data_buffer(buf); check_ret("acl.destroy_data_buffer output", ret)
    ret = acl.mdl.destroy_dataset(io_res["output_ds"]); check_ret("acl.mdl.destroy_dataset output", ret)

def run_acl_model(model_id, model_desc, host_x, io_res):
    start = time.time()
    input_bytes = host_x.tobytes()
    input_ptr = acl.util.bytes_to_ptr(input_bytes)
    ret = acl.rt.memcpy(io_res["input_device"], io_res["input_size"], input_ptr,
                        io_res["input_size"], ACL_MEMCPY_HOST_TO_DEVICE); check_ret("acl.rt.memcpy H2D", ret)

    
    ret = acl.mdl.execute(model_id, io_res["input_ds"], io_res["output_ds"]); check_ret("acl.mdl.execute", ret)
    

    out_size = io_res["out_sizes"][0]
    host_out_buf = ctypes.create_string_buffer(out_size)
    host_out_ptr = ctypes.addressof(host_out_buf)
    ret = acl.rt.memcpy(host_out_ptr, out_size, io_res["out_dev"][0],
                        out_size, ACL_MEMCPY_DEVICE_TO_HOST); check_ret("acl.rt.memcpy D2H", ret)
    host_out_bytes = acl.util.ptr_to_bytes(host_out_ptr, out_size)
    host_out = np.frombuffer(host_out_bytes, dtype=host_x.dtype).reshape(host_x.shape)
    elapsed = time.time() - start
    return host_out, elapsed

if __name__ == "__main__":
    # 随机模拟输入，x_batch就是输入
    np.random.seed(42)
    x_batch = np.random.randn(MODEL_BATCH, 2, 256).astype(np.float32)

    ret = acl.init(); check_ret("acl.init", ret)
    dev_id = 0
    ret = acl.rt.set_device(dev_id); check_ret("acl.rt.set_device", ret)
    ctx, ret = acl.rt.create_context(dev_id); check_ret("acl.rt.create_context", ret)
    stream, ret = acl.rt.create_stream(); check_ret("acl.rt.create_stream", ret)

    # 根据实际选择对应的 .om 模型文件，idft就用idft的模型，反正输入输出是一样的
    model_id, ret = acl.mdl.load_from_file("dft256_mat_1192.om"); check_ret("acl.mdl.load_from_file", ret)
    model_desc = acl.mdl.create_desc()
    ret = acl.mdl.get_desc(model_desc, model_id); check_ret("acl.mdl.get_desc", ret)

    io_res = create_io_resources(model_desc, x_batch.shape)
#####下面是推理，只要循环运行下面这个就行，不需要每次都load和unload模型
    out, t = run_acl_model(model_id, model_desc, x_batch, io_res)
    print(f"idft256_mat.om batch={MODEL_BATCH} 输出形状 {out.shape}, 总耗时 {t*1000:.2f} ms")
#####上面是推理
    destroy_io_resources(io_res)
    ret = acl.mdl.unload(model_id); check_ret("acl.mdl.unload", ret)
    ret = acl.mdl.destroy_desc(model_desc); check_ret("acl.mdl.destroy_desc", ret)
    ret = acl.rt.destroy_stream(stream); check_ret("acl.rt.destroy_stream", ret)
    ret = acl.rt.destroy_context(ctx); check_ret("acl.rt.destroy_context", ret)
    ret = acl.rt.reset_device(dev_id); check_ret("acl.rt.reset_device", ret)
    ret = acl.finalize(); check_ret("acl.finalize", ret)
