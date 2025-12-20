import numpy as np
import acl
import time
from PIL import Image
import cv2

MODEL_PATH = "./yolo11n.om"
IMAGE_PATH = "./YOLO/tst.jpg"
SAVE_PATH = "./YOLO/tst_out.jpg"
MODEL_BATCH = 1
ACL_MEM_MALLOC_NORMAL_ONLY = 0
ACL_MEMCPY_HOST_TO_DEVICE = 0
ACL_MEMCPY_DEVICE_TO_HOST = 1
ACL_SUCCESS = 0

COCO_LABELS = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

def check_ret(msg, ret):
    if ret != ACL_SUCCESS:
        raise RuntimeError(f"{msg} failed ret={ret}")

def preprocess(img_np):
    # 输入为RGB格式，直接归一化并转为模型输入格式
    img = img_np.transpose(2,0,1)[np.newaxis].astype(np.float32)/255
    return img

def run_acl_model(model_id, model_desc, host_x):
    input_size = host_x.size * host_x.itemsize
    input_device, ret = acl.rt.malloc(input_size, ACL_MEM_MALLOC_NORMAL_ONLY); check_ret("acl.rt.malloc", ret)
    ret = acl.rt.memcpy(input_device, input_size, acl.util.bytes_to_ptr(host_x.tobytes()),
                        input_size, ACL_MEMCPY_HOST_TO_DEVICE); check_ret("acl.rt.memcpy", ret)
    input_buf = acl.create_data_buffer(input_device, input_size)
    input_ds = acl.mdl.create_dataset()
    _, ret = acl.mdl.add_dataset_buffer(input_ds, input_buf); check_ret("acl.mdl.add_dataset_buffer", ret)

    out_num = acl.mdl.get_num_outputs(model_desc)
    output_ds = acl.mdl.create_dataset()
    out_dev, out_sizes, out_bufs = [], [], []
    for i in range(out_num):
        size_i = acl.mdl.get_output_size_by_index(model_desc, i)
        dev_i, ret = acl.rt.malloc(size_i, ACL_MEM_MALLOC_NORMAL_ONLY); check_ret("acl.rt.malloc", ret)
        buf_i = acl.create_data_buffer(dev_i, size_i)
        _, ret = acl.mdl.add_dataset_buffer(output_ds, buf_i); check_ret("acl.mdl.add_dataset_buffer", ret)
        out_dev.append(dev_i); out_sizes.append(size_i); out_bufs.append(buf_i)

    start = time.time()
    ret = acl.mdl.execute(model_id, input_ds, output_ds); check_ret("acl.mdl.execute", ret)
    infer_time = time.time() - start

    # 输出 shape: (1, 8400, 84)
    host_out = np.empty((1, 8400, 84), dtype=np.float32)
    ret = acl.rt.memcpy(acl.util.bytes_to_ptr(host_out.tobytes()), out_sizes[0],
                        out_dev[0], out_sizes[0], ACL_MEMCPY_DEVICE_TO_HOST); check_ret("acl.rt.memcpy D2H", ret)

    # 释放资源
    ret = acl.rt.free(input_device); check_ret("acl.rt.free input", ret)
    for dev, buf in zip(out_dev, out_bufs):
        ret = acl.rt.free(dev); check_ret("acl.rt.free output", ret)
        ret = acl.destroy_data_buffer(buf); check_ret("acl.destroy_data_buffer output", ret)
    ret = acl.mdl.destroy_dataset(input_ds); check_ret("acl.mdl.destroy_dataset input", ret)
    ret = acl.mdl.destroy_dataset(output_ds); check_ret("acl.mdl.destroy_dataset output", ret)
    ret = acl.destroy_data_buffer(input_buf); check_ret("acl.destroy_data_buffer input", ret)
    return host_out, infer_time

def postprocess(pred, print_only=True, orig_img=None):
    CONF_THRESH = 0.01
    IOU_THRESH = 0.3
    arr = pred[0]
    conf_mask = arr[:, 4] > CONF_THRESH
    detections = []
    for i in range(arr.shape[0]):
        if not conf_mask[i]:
            continue
        cx, cy, w, h = arr[i, :4]
        conf = arr[i, 4]
        cls_scores = arr[i, 5:]
        class_id = np.argmax(cls_scores)
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        detections.append([x1, y1, x2, y2, conf, class_id])
    # NMS
    boxes = [d[:4] for d in detections]
    confs = [d[4] for d in detections]
    indices = cv2.dnn.NMSBoxes(boxes, confs, CONF_THRESH, IOU_THRESH)
    if len(indices) == 0:
        result = []
    else:
        if isinstance(indices, np.ndarray):
            indices = indices.flatten()
        else:
            indices = [i[0] if isinstance(i, (list, tuple, np.ndarray)) else i for i in indices]
        result = [detections[i] for i in indices]
    # 打印
    label_count = {}
    for det in result:
        cls_id = det[5]
        label = COCO_LABELS[cls_id]
        label_count[label] = label_count.get(label, 0) + 1
    print("识别到的标签及数量:", label_count)
    print("总数:", len(result))
    if not print_only and orig_img is not None:
        img_draw = orig_img.copy()
        h, w = img_draw.shape[:2]
        for det in result:
            x1, y1, x2, y2, conf, cls_id = det
            # 参考 example.py，确保坐标为 int 且在图片范围内
            x1 = max(0, min(int(round(x1)), w - 1))
            y1 = max(0, min(int(round(y1)), h - 1))
            x2 = max(0, min(int(round(x2)), w - 1))
            y2 = max(0, min(int(round(y2)), h - 1))
            label = f"{COCO_LABELS[cls_id]} {conf:.2f}"
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0,255,0), 2)
            text_y = max(y1 - 10, 0)
            cv2.putText(img_draw, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imwrite(SAVE_PATH, cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR))
        print(f"已保存结果图片到 {SAVE_PATH}")

def main(print_only=True):
    # 计时：前处理
    t0 = time.time()
    img_pil = Image.open(IMAGE_PATH).convert("RGB")
    img_np = np.array(img_pil)
    img_input = preprocess(img_np)
    t1 = time.time()
    print(f"前处理耗时: {(t1-t0)*1000:.2f} ms")

    # ACL初始化与模型加载
    ret = acl.init(); check_ret("acl.init", ret)
    dev_id = 0
    ret = acl.rt.set_device(dev_id); check_ret("acl.rt.set_device", ret)
    ctx, ret = acl.rt.create_context(dev_id); check_ret("acl.rt.create_context", ret)
    stream, ret = acl.rt.create_stream(); check_ret("acl.rt.create_stream", ret)
    model_id, ret = acl.mdl.load_from_file(MODEL_PATH); check_ret("acl.mdl.load_from_file", ret)
    model_desc = acl.mdl.create_desc()
    ret = acl.mdl.get_desc(model_desc, model_id); check_ret("acl.mdl.get_desc", ret)

    # 推理
    t2 = time.time()
    out, infer_time = run_acl_model(model_id, model_desc, img_input)
    t3 = time.time()
    print(f"推理耗时: {infer_time*1000:.2f} ms")

    # 后处理
    t4 = time.time()
    postprocess(out, print_only=print_only, orig_img=img_np)
    t5 = time.time()
    print(f"后处理耗时: {(t5-t4)*1000:.2f} ms")

    # 释放资源
    ret = acl.mdl.unload(model_id); check_ret("acl.mdl.unload", ret)
    ret = acl.mdl.destroy_desc(model_desc); check_ret("acl.mdl.destroy_desc", ret)
    ret = acl.rt.destroy_stream(stream); check_ret("acl.rt.destroy_stream", ret)
    ret = acl.rt.destroy_context(ctx); check_ret("acl.rt.destroy_context", ret)
    ret = acl.rt.reset_device(dev_id); check_ret("acl.rt.reset_device", ret)
    ret = acl.finalize(); check_ret("acl.finalize", ret)

if __name__ == "__main__":
    # print_only=True 只打印，False则打印并保存图片
    main(print_only=True)
    # main(print_only=False)
