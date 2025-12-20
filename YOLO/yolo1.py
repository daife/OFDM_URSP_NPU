import numpy as np
import time
from PIL import Image
import cv2
from acllite_model import AclLiteModel

import acl
from acllite_resource import resource_list

class AclLiteResource:
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.context = None
        self.stream = None

    def init(self):
        acl.init()
        acl.rt.set_device(self.device_id)
        self.context, _ = acl.rt.create_context(self.device_id)
        self.stream, _ = acl.rt.create_stream()

    def __del__(self):
        resource_list.destroy()
        if self.stream:
            acl.rt.destroy_stream(self.stream)
        if self.context:
            acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)
        acl.finalize()

# COCO 80类标签
COCO_CLASSES = [
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

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    # 转为NCHW格式，float32
    img_np = img_np.transpose(2, 0, 1)[np.newaxis].astype(np.float32)/255
    return img_np, np.array(img)

def postprocess(pred, conf_thresh=0.00001, iou_thresh=0.9):
    arr = pred[0]
    # 只处理(1, 84, 8400)情况
    if arr.shape == (1, 84, 8400):
        arr = arr.transpose(0, 2, 1)  # (1, 8400, 84)
        arr = arr[0]
    else:
        raise ValueError(f"Unexpected pred shape: {arr.shape}")
    conf_mask = arr[:, 4] > conf_thresh
    detections = []
    for i in range(arr.shape[0]):
        if not conf_mask[i]:
            continue
        cx, cy, w, h = arr[i, :4]
        conf = arr[i, 4]
        cls_scores = arr[i, 5:]
        class_id = np.argmax(cls_scores)
        score = conf * cls_scores[class_id]
        if score < conf_thresh:
            continue
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        detections.append([x1, y1, x2, y2, float(score), int(class_id)])
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
        label = f"{COCO_CLASSES[cls_id]} {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, label, (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return img

def main(mode="print"):  # mode: "print" or "save"
    MODEL_PATH = "./yolo11n.om"
    IMAGE_PATH = "./YOLO/tst.jpg"

    # ACL初始化
    acl_resource = AclLiteResource()
    acl_resource.init()

    # 1. 读取图片
    t0 = time.time()
    img, img_for_draw = load_image(IMAGE_PATH)
    t1 = time.time()

    # 2. 加载模型
    model = AclLiteModel(MODEL_PATH)

    # 3. 推理
    t2 = time.time()
    pred = model.execute([img])
    t3 = time.time()

    # 4. 后处理
    detections = postprocess(pred)
    t4 = time.time()

    # 5. 统计标签
    label_count = {}
    for det in detections:
        cls_id = det[5]
        label = COCO_CLASSES[cls_id]
        label_count[label] = label_count.get(label, 0) + 1

    # 6. 打印
    print("识别到的标签及数量:", label_count)
    print(f"前处理耗时: {(t1-t0)*1000:.2f} ms, 推理耗时: {(t3-t2)*1000:.2f} ms, 后处理耗时: {(t4-t3)*1000:.2f} ms")

    if mode == "save":
        img_draw = draw_boxes(img_for_draw.copy(), detections)
        save_path = "./YOLO/tst_result.jpg"
        cv2.imwrite(save_path, cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR))
        print(f"结果已保存到: {save_path}")
    # 资源释放
    del acl_resource

if __name__ == "__main__":
    # 仅打印: main("print")
    # 打印并保存: main("save")
    # main("print")
    main("save")
