import numpy as np
import acl
import time
import cv2
import ctypes

MODEL_PATH = "./person_yolo11n.om"
IMAGE_PATH = "./YOLO/f000012.jpg"
SAVE_PATH = "./YOLO/person_out.png"
#nms的两个参数
CONF_THRESH=0.25 #置信度
IOU_THRESH=0.45 #防止重合，调一下试一下就知道了
MODEL_BATCH = 1
ACL_MEM_MALLOC_NORMAL_ONLY = 0
ACL_MEMCPY_HOST_TO_DEVICE = 0
ACL_MEMCPY_DEVICE_TO_HOST = 1
ACL_SUCCESS = 0

def check_ret(msg, ret):
    if ret != ACL_SUCCESS:
        raise RuntimeError(f"{msg} failed ret={ret}")

def preprocess(img_np):
    # 输入为BGR格式，转RGB归一化并转为模型输入格式nchw
    img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) #将opencv默认的BGR格式转换为RGB格式，如果解码出来就是RGB那就不用这一行
    img = img.transpose(2,0,1)[np.newaxis].astype(np.float32)/255
    return img

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
    ret = acl.rt.memcpy(io_res["input_device"], io_res["input_size"],
                        acl.util.bytes_to_ptr(host_x.tobytes()),
                        io_res["input_size"], ACL_MEMCPY_HOST_TO_DEVICE); check_ret("acl.rt.memcpy H2D", ret)

    start = time.time()
    ret = acl.mdl.execute(model_id, io_res["input_ds"], io_res["output_ds"]); check_ret("acl.mdl.execute", ret)
    infer_time = time.time() - start

    out_size = io_res["out_sizes"][0]
    host_out_buf = ctypes.create_string_buffer(out_size)
    host_out_ptr = ctypes.addressof(host_out_buf)
    ret = acl.rt.memcpy(host_out_ptr, out_size, io_res["out_dev"][0],
                        out_size, ACL_MEMCPY_DEVICE_TO_HOST); check_ret("acl.rt.memcpy D2H", ret)
    host_out_bytes = acl.util.ptr_to_bytes(host_out_ptr, out_size)
    elem_cnt = out_size // np.dtype(np.float32).itemsize
    # 输出shape为(5, 8400)或(1, 5, 8400)
    if elem_cnt == 5 * 8400:
        host_out = np.frombuffer(host_out_bytes, dtype=np.float32).reshape(1, 5, 8400)
    else:
        host_out = np.frombuffer(host_out_bytes, dtype=np.float32).reshape(1, elem_cnt)
    return host_out, infer_time

def postprocess(pred, conf_thresh=CONF_THRESH, iou_thresh=IOU_THRESH):
    arr = pred[0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
    if arr.shape == (5, 8400):
        arr = arr.transpose(1, 0)  # (8400, 5)
    elif arr.shape == (8400, 5):
        pass
    else:
        raise ValueError(f"Unexpected pred shape: {arr.shape}")
    conf_mask = arr[:, 4] > conf_thresh
    detections = []
    for i in range(arr.shape[0]):
        if not conf_mask[i]:
            continue
        cx, cy, w, h = arr[i, :4]
        conf = arr[i, 4]
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        detections.append([x1, y1, x2, y2, float(conf), 0])  # 只检测person
    boxes = [d[:4] for d in detections]
    scores = [d[4] for d in detections]
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
    if len(indices) == 0:
        return []
    if isinstance(indices, np.ndarray):
        indices = indices.flatten()
    else:
        indices = [i[0] if isinstance(i, (list, tuple, np.ndarray)) else i for i in indices]
    return [detections[i] for i in indices]

def draw_boxes(img, detections):
    for det in detections:
        x1, y1, x2, y2, score, cls_id = det
        label = f"person {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, label, (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return img

def main(mode="print"):  # mode: "print" or "save"
    # 前处理计时
    img_np = cv2.imread(IMAGE_PATH) #这里是用opencv从图片读取（bgr），换成你h265解码后的图像就行
    t0 = time.time()
    img_input = preprocess(img_np)
    t1 = time.time()

    # ACL初始化与模型加载
    ret = acl.init(); check_ret("acl.init", ret)
    dev_id = 0
    ret = acl.rt.set_device(dev_id); check_ret("acl.rt.set_device", ret)
    ctx, ret = acl.rt.create_context(dev_id); check_ret("acl.rt.create_context", ret)
    stream, ret = acl.rt.create_stream(); check_ret("acl.rt.create_stream", ret)
    model_id, ret = acl.mdl.load_from_file(MODEL_PATH); check_ret("acl.mdl.load_from_file", ret)
    model_desc = acl.mdl.create_desc()
    ret = acl.mdl.get_desc(model_desc, model_id); check_ret("acl.mdl.get_desc", ret)

    io_res = create_io_resources(model_desc, img_input.shape)

    # 推理
    t2 = time.time()
    out, infer_time = run_acl_model(model_id, model_desc, img_input, io_res)
    t3 = time.time()

    # 后处理
    detections = postprocess(out)
    t4 = time.time()

    print(f"前处理耗时: {(t1-t0)*1000:.2f} ms, 推理耗时: {(t3-t2)*1000:.2f} ms, 后处理耗时: {(t4-t3)*1000:.2f} ms")
    print("检测结果:", detections)

    if mode == "save":
        img_draw = draw_boxes(img_np.copy(), detections)#这一步就是在原图（的拷贝）上绘制框
        cv2.imwrite(SAVE_PATH, img_draw)#这个是保存
        print(f"结果已保存到: {SAVE_PATH}")

    if mode == "show":#要显示在屏幕上就是以下代码
        img_draw = draw_boxes(img_np.copy(), detections)
        cv2.imshow("Detections", img_draw)#这一步是显示
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 释放资源
    destroy_io_resources(io_res)
    ret = acl.mdl.unload(model_id); check_ret("acl.mdl.unload", ret)
    ret = acl.mdl.destroy_desc(model_desc); check_ret("acl.mdl.destroy_desc", ret)
    ret = acl.rt.destroy_stream(stream); check_ret("acl.rt.destroy_stream", ret)
    ret = acl.rt.destroy_context(ctx); check_ret("acl.rt.destroy_context", ret)
    ret = acl.rt.reset_device(dev_id); check_ret("acl.rt.reset_device", ret)
    ret = acl.finalize(); check_ret("acl.finalize", ret)

if __name__ == "__main__":
    # main("print")
    main("save")
    # main("show")
