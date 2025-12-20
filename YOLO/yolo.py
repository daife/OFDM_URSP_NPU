import os
import time
import numpy as np
import acl
import cv2
from PIL import Image

MODEL_PATH = "./yolo11n.om"
IMAGE_PATH = "./YOLO/tst.jpg"
SAVE_RESULT = True  # False=仅打印; True=打印并保存标注图
OUTPUT_PATH = "./YOLO/tst_out.jpg"
INPUT_SIZE = 640

ACL_MEM_MALLOC_NORMAL_ONLY = 0
ACL_MEMCPY_HOST_TO_DEVICE = 0
ACL_MEMCPY_DEVICE_TO_HOST = 1
ACL_SUCCESS = 0

def check_ret(msg, ret):
    if ret != ACL_SUCCESS:
        raise RuntimeError(f"{msg} failed ret={ret}")

class YOLO11NRunner:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_id = None
        self.model_desc = None

    def init_acl(self, device_id=0):
        ret = acl.init(); check_ret("acl.init", ret)
        ret = acl.rt.set_device(device_id); check_ret("acl.rt.set_device", ret)
        self.ctx, ret = acl.rt.create_context(device_id); check_ret("acl.rt.create_context", ret)
        self.stream, ret = acl.rt.create_stream(); check_ret("acl.rt.create_stream", ret)

    def load_model(self):
        self.model_id, ret = acl.mdl.load_from_file(self.model_path); check_ret("acl.mdl.load_from_file", ret)
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id); check_ret("acl.mdl.get_desc", ret)

    def unload(self):
        if self.model_id is not None:
            ret = acl.mdl.unload(self.model_id); check_ret("acl.mdl.unload", ret)
        if self.model_desc is not None:
            ret = acl.mdl.destroy_desc(self.model_desc); check_ret("acl.mdl.destroy_desc", ret)
        if hasattr(self, "stream"):
            ret = acl.rt.destroy_stream(self.stream); check_ret("acl.rt.destroy_stream", ret)
        if hasattr(self, "ctx"):
            ret = acl.rt.destroy_context(self.ctx); check_ret("acl.rt.destroy_context", ret)
        ret = acl.rt.reset_device(0); check_ret("acl.rt.reset_device", ret)
        ret = acl.finalize(); check_ret("acl.finalize", ret)

    def _create_dataset(self, buf_ptr, size):
        ds = acl.mdl.create_dataset()
        buf = acl.create_data_buffer(buf_ptr, size)
        _, ret = acl.mdl.add_dataset_buffer(ds, buf); check_ret("acl.mdl.add_dataset_buffer", ret)
        return ds, buf

    def infer(self, input_np):
        input_size = input_np.size * input_np.itemsize
        input_dev, ret = acl.rt.malloc(input_size, ACL_MEM_MALLOC_NORMAL_ONLY); check_ret("acl.rt.malloc input", ret)
        ret = acl.rt.memcpy(input_dev, input_size, acl.util.bytes_to_ptr(input_np.tobytes()),
                            input_size, ACL_MEMCPY_HOST_TO_DEVICE); check_ret("acl.rt.memcpy H2D", ret)
        input_ds, input_buf = self._create_dataset(input_dev, input_size)

        out_count = acl.mdl.get_num_outputs(self.model_desc)
        output_ds = acl.mdl.create_dataset()
        out_dev, out_sizes, out_bufs = [], [], []
        for i in range(out_count):
            size_i = acl.mdl.get_output_size_by_index(self.model_desc, i)
            dev_i, ret = acl.rt.malloc(size_i, ACL_MEM_MALLOC_NORMAL_ONLY); check_ret("acl.rt.malloc output", ret)
            buf_i = acl.create_data_buffer(dev_i, size_i)
            _, ret = acl.mdl.add_dataset_buffer(output_ds, buf_i); check_ret("acl.mdl.add_dataset_buffer", ret)
            out_dev.append(dev_i); out_sizes.append(size_i); out_bufs.append(buf_i)

        start = time.time()
        ret = acl.mdl.execute(self.model_id, input_ds, output_ds); check_ret("acl.mdl.execute", ret)
        infer_time = (time.time() - start) * 1000

        host_out = np.empty(out_sizes[0] // 4, dtype=np.float32)
        ret = acl.rt.memcpy(acl.util.bytes_to_ptr(host_out.tobytes()), out_sizes[0],
                            out_dev[0], out_sizes[0], ACL_MEMCPY_DEVICE_TO_HOST); check_ret("acl.rt.memcpy D2H", ret)
        print(f"模型输出shape(reshape前): {host_out.shape}")
        # 不再 reshape，保持扁平，提高性能

        ret = acl.rt.free(input_dev); check_ret("acl.rt.free input", ret)
        ret = acl.destroy_data_buffer(input_buf); check_ret("acl.destroy_data_buffer input", ret)
        ret = acl.mdl.destroy_dataset(input_ds); check_ret("acl.mdl.destroy_dataset input", ret)
        for dev_i, buf_i in zip(out_dev, out_bufs):
            ret = acl.rt.free(dev_i); check_ret("acl.rt.free output", ret)
            ret = acl.destroy_data_buffer(buf_i); check_ret("acl.destroy_data_buffer output", ret)
        ret = acl.mdl.destroy_dataset(output_ds); check_ret("acl.mdl.destroy_dataset output", ret)
        return host_out, infer_time

def preprocess(image_rgb, input_size=INPUT_SIZE):
    """保持宽高比缩放 + 填充灰边，返回模型输入、原始尺寸和填充信息"""
    h, w = image_rgb.shape[:2]
    scale = min(input_size / w, input_size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(image_rgb, (nw, nh))
    top = (input_size - nh) // 2
    bottom = input_size - nh - top
    left = (input_size - nw) // 2
    right = input_size - nw - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    inp = padded.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
    return inp, (h, w), (top, left, scale)

def postprocess(pred, orig_shape, pad_info):
    """每类保留最高分框，坐标反归一到原图"""
    CONF_THRESH = 0.1
    arr = np.asarray(pred)
    if arr.ndim == 1:
        if arr.size % 8 != 0:
            raise ValueError(f"Unexpected pred size {arr.size}, cannot reshape to Nx8")
        arr = arr.reshape(1, -1, 8)
    if arr.ndim == 2:
        if arr.shape[1] != 8:
            raise ValueError(f"Unexpected pred shape {arr.shape}, expect (*,8)")
        arr = arr[np.newaxis, ...]
    arr = arr[0]
    detections = []
    top, left, scale = pad_info
    for i in range(arr.shape[0]):
        conf = arr[i, 4]
        if conf < CONF_THRESH:
            continue
        cls_scores = arr[i, 5:]
        if cls_scores.size == 0:
            continue
        class_id = int(np.argmax(cls_scores))
        score = cls_scores[class_id]
        if score < CONF_THRESH or not np.isfinite(score):
            continue
        cx, cy, w, h = arr[i, :4]
        if not np.all(np.isfinite([cx, cy, w, h])):
            continue
        cx = (cx - left) / scale
        cy = (cy - top) / scale
        w = w / scale
        h = h / scale
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        detections.append([x1, y1, x2, y2, float(score), class_id])
    best = {}
    for det in detections:
        cls_id = det[5]
        if (cls_id not in best) or (det[4] > best[cls_id][4]):
            best[cls_id] = det
    return list(best.values())

def draw_and_save(image_rgb, detections, path):
    if image_rgb is None or image_rgb.size == 0:
        return
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    for det in detections:
        if len(det) < 6:
            continue
        x1, y1, x2, y2, score, cls_id = det
        x1, y1, x2, y2 = [int(round(v)) for v in (x1, y1, x2, y2)]
        x1 = np.clip(x1, 0, w - 1)
        y1 = np.clip(y1, 0, h - 1)
        x2 = np.clip(x2, 0, w - 1)
        y2 = np.clip(y2, 0, h - 1)
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{cls_id} {score:.2f}"
        cv2.putText(img_bgr, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(path, img_bgr)

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"模型不存在: {os.path.abspath(MODEL_PATH)}"); return
    if not os.path.exists(IMAGE_PATH):
        print(f"图片不存在: {os.path.abspath(IMAGE_PATH)}"); return

    runner = YOLO11NRunner(MODEL_PATH)
    runner.init_acl()
    runner.load_model()

    try:
        t0 = time.time()
        img = Image.open(IMAGE_PATH).convert("RGB")
        np_img_uint8 = np.array(img, dtype=np.uint8)
        input_tensor, orig_shape, pad_info = preprocess(np_img_uint8)
        t_pre = (time.time() - t0) * 1000

        pred, t_inf = runner.infer(input_tensor)

        t1 = time.time()
        detections = postprocess(pred, orig_shape, pad_info)
        t_post = (time.time() - t1) * 1000

        counts = {}
        for _, _, _, _, _, cid in detections:
            counts[cid] = counts.get(cid, 0) + 1
        print(f"检测结果: {counts}, 预处理 {t_pre:.2f}ms, 推理 {t_inf:.2f}ms, 后处理 {t_post:.2f}ms")

        if SAVE_RESULT:
            draw_and_save(np_img_uint8.copy(), detections, OUTPUT_PATH)
            print(f"结果已保存: {os.path.abspath(OUTPUT_PATH)}")
    finally:
        runner.unload()

if __name__ == "__main__":
    main()
