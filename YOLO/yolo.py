import os
import time
import numpy as np
import acl
from PIL import Image
import cv2

# 配置参数
MODEL_PATH = "./yolo11n.om"
IMAGE_PATH = "./YOLO/tst.jpg"
SAVE_OUTPUT = True  # False: 仅打印; True: 打印并保存带框结果
OUTPUT_IMAGE_PATH = "./YOLO/tst_out.jpg"
ACL_MEM_MALLOC_NORMAL_ONLY = 0
ACL_MEMCPY_HOST_TO_DEVICE = 0
ACL_MEMCPY_DEVICE_TO_HOST = 1
ACL_SUCCESS = 0
CONF_THRESH = 0.9
NMS_THRESH = 0.45
YOLO_OUTPUT_SHAPE = (1, 8400, 84)  # (batch, anchors, 4+80)

COCO80 = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli",
    "carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table",
    "toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster",
    "sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

def check_ret(msg, ret):
    if ret != ACL_SUCCESS:
        raise RuntimeError(f"{msg} failed ret={ret}")

def init_acl():
    ret = acl.init(); check_ret("acl.init", ret)
    dev_id = 0
    ret = acl.rt.set_device(dev_id); check_ret("acl.rt.set_device", ret)
    ctx, ret = acl.rt.create_context(dev_id); check_ret("acl.rt.create_context", ret)
    stream, ret = acl.rt.create_stream(); check_ret("acl.rt.create_stream", ret)
    return dev_id, ctx, stream

def finalize_acl(dev_id, ctx, stream):
    ret = acl.rt.destroy_stream(stream); check_ret("acl.rt.destroy_stream", ret)
    ret = acl.rt.destroy_context(ctx); check_ret("acl.rt.destroy_context", ret)
    ret = acl.rt.reset_device(dev_id); check_ret("acl.rt.reset_device", ret)
    ret = acl.finalize(); check_ret("acl.finalize", ret)

def load_model(model_path):
    model_id, ret = acl.mdl.load_from_file(model_path); check_ret("acl.mdl.load_from_file", ret)
    model_desc = acl.mdl.create_desc()
    ret = acl.mdl.get_desc(model_desc, model_id); check_ret("acl.mdl.get_desc", ret)
    return model_id, model_desc

def unload_model(model_id, model_desc):
    ret = acl.mdl.unload(model_id); check_ret("acl.mdl.unload", ret)
    ret = acl.mdl.destroy_desc(model_desc); check_ret("acl.mdl.destroy_desc", ret)

def preprocess(image_path):
    t0 = time.time()
    img = Image.open(image_path).convert("RGB")
    np_img = np.array(img, dtype=np.float32) / 255.0  # HWC, RGB
    np_img = np_img.transpose(2, 0, 1)[np.newaxis, ...]  # NCHW
    return np_img, time.time() - t0, img.size  # (W, H) from PIL

def run_inference(model_id, model_desc, host_x):
    input_size = host_x.size * host_x.itemsize
    input_dev, ret = acl.rt.malloc(input_size, ACL_MEM_MALLOC_NORMAL_ONLY); check_ret("acl.rt.malloc input", ret)
    ret = acl.rt.memcpy(input_dev, input_size, acl.util.bytes_to_ptr(host_x.tobytes()),
                        input_size, ACL_MEMCPY_HOST_TO_DEVICE); check_ret("acl.rt.memcpy H2D", ret)
    input_buf = acl.create_data_buffer(input_dev, input_size)
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

    t0 = time.time()
    ret = acl.mdl.execute(model_id, input_ds, output_ds); check_ret("acl.mdl.execute", ret)
    infer_time = time.time() - t0

    # 只取第一个输出
    host_out = np.empty(out_sizes[0] // 4, dtype=np.float32)
    ret = acl.rt.memcpy(acl.util.bytes_to_ptr(host_out.tobytes()), out_sizes[0],
                        out_dev[0], out_sizes[0], ACL_MEMCPY_DEVICE_TO_HOST); check_ret("acl.rt.memcpy D2H", ret)
    # 释放资源
    ret = acl.rt.free(input_dev); check_ret("acl.rt.free input", ret)
    for dev, buf in zip(out_dev, out_bufs):
        ret = acl.rt.free(dev); check_ret("acl.rt.free output", ret)
        ret = acl.destroy_data_buffer(buf); check_ret("acl.destroy_data_buffer output", ret)
    ret = acl.mdl.destroy_dataset(input_ds); check_ret("acl.mdl.destroy_dataset input", ret)
    ret = acl.mdl.destroy_dataset(output_ds); check_ret("acl.mdl.destroy_dataset output", ret)
    ret = acl.destroy_data_buffer(input_buf); check_ret("acl.destroy_data_buffer input", ret)
    return host_out, infer_time

def iou(box1, box2):
    x11, y11, x12, y12 = box1[:4]
    x21, y21, x22, y22 = box2[:4]
    xi1, yi1 = max(x11, x21), max(y11, y21)
    xi2, yi2 = min(x12, x22), min(y12, y22)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = max(0, x12 - x11) * max(0, y12 - y11)
    area2 = max(0, x22 - x21) * max(0, y22 - y21)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def nms(dets, iou_thresh):
    kept = []
    by_cls = {}
    for d in dets:
        by_cls.setdefault(d[5], []).append(d)
    for cls_dets in by_cls.values():
        cls_dets = sorted(cls_dets, key=lambda x: x[4], reverse=True)
        while cls_dets:
            best = cls_dets.pop(0)
            kept.append(best)
            cls_dets = [d for d in cls_dets if iou(best, d) < iou_thresh]
    return kept

def postprocess(raw_out, orig_size, input_size):
    t0 = time.time()
    in_h, in_w = input_size
    orig_w, orig_h = orig_size
    scale_x = orig_w / in_w
    scale_y = orig_h / in_h
    arr = raw_out.reshape(YOLO_OUTPUT_SHAPE).squeeze(0)  # (8400, 84)
    detections = []
    for i in range(arr.shape[0]):
        cx, cy, w, h = arr[i, :4]
        obj_conf = arr[i, 4]
        cls_scores = arr[i, 5:]
        cls_id = int(np.argmax(cls_scores))
        cls_conf = cls_scores[cls_id]
        conf = cls_conf * obj_conf
        if conf < CONF_THRESH:
            continue
        x1 = max(0, min(int((cx - w / 2) * scale_x), orig_w - 1))
        y1 = max(0, min(int((cy - h / 2) * scale_y), orig_h - 1))
        x2 = max(0, min(int((cx + w / 2) * scale_x), orig_w - 1))
        y2 = max(0, min(int((cy + h / 2) * scale_y), orig_h - 1))
        detections.append((x1, y1, x2, y2, conf, cls_id))
    detections = nms(detections, NMS_THRESH)
    post_time = time.time() - t0
    return detections, post_time

def summarize_and_optionally_save(dets, image_path, image_size, save_output):
    counts = {}
    for _, _, _, _, conf, cls_id in dets:
        name = COCO80[cls_id] if cls_id < len(COCO80) else str(cls_id)
        counts[name] = counts.get(name, 0) + 1
    print("检测结果：", {k: v for k, v in counts.items()})
    if not save_output:
        return
    img = Image.open(image_path).convert("RGB")
    np_img = np.array(img)
    bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    for x1, y1, x2, y2, conf, cls_id in dets:
        name = COCO80[cls_id] if cls_id < len(COCO80) else str(cls_id)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(bgr, f"{name} {conf:.2f}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(OUTPUT_IMAGE_PATH, bgr)
    print(f"已保存结果到 {OUTPUT_IMAGE_PATH}")

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"模型文件不存在: {MODEL_PATH}")
        return
    if not os.path.exists(IMAGE_PATH):
        print(f"图片不存在: {IMAGE_PATH}")
        return

    dev_id, ctx, stream = init_acl()
    try:
        model_id, model_desc = load_model(MODEL_PATH)
        try:
            img, t_pre, orig_wh = preprocess(IMAGE_PATH)
            out, t_inf = run_inference(model_id, model_desc, img)
            dets, t_post = postprocess(out, orig_wh, img.shape[-2:])
            summarize_and_optionally_save(dets, IMAGE_PATH, orig_wh, SAVE_OUTPUT)
            print(f"耗时: 预处理 {t_pre*1000:.2f} ms, 推理 {t_inf*1000:.2f} ms, 后处理 {t_post*1000:.2f} ms")
        finally:
            unload_model(model_id, model_desc)
    finally:
        finalize_acl(dev_id, ctx, stream)

if __name__ == "__main__":
    main()