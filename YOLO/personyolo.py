import os
import cv2
import numpy as np
import time
import acl
from acllite_utils import *
from acllite_imageproc import AclLiteImageProc
from acllite_model import AclLiteModel
from acllite_resource import resource_list
from constants import *

# 宏定义
MODEL_PATH = "./person_yolo11n.om"
IMAGE_PATH = "./YOLO/person.png"
OUTPUT_PATH = "person_out.png"
SHOW_RESULT = False  # 若需imshow显示，改为True

# 与示例一致的ACL资源管理
class AclLiteResource:
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.context = None
        self.stream = None

    def init(self):
        ret = acl.init()
        ret = acl.rt.set_device(self.device_id)
        self.context, ret = acl.rt.create_context(self.device_id)
        self.stream, ret = acl.rt.create_stream()
        return const.SUCCESS

    def __del__(self):
        resource_list.destroy()
        if self.stream:
            acl.rt.destroy_stream(self.stream)
        if self.context:
            acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)
        acl.finalize()

class PersonYOLO:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def init(self):
        self.model = AclLiteModel(self.model_path)
        return const.SUCCESS

    def preprocess(self, frame):
        # 假定frame为640x640，直接归一化和通道变换
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255
        return img

    def postprocess(self, pred):
        CONF_THRESH = 0.25
        IOU_THRESH = 0.45
        arr = pred[0]
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        if arr.shape == (5, 8400):
            arr = arr.transpose(1, 0)  # (8400, 5)
        elif arr.shape == (8400, 5):
            pass
        else:
            raise ValueError(f"Unexpected pred shape: {arr.shape}")
        conf_mask = arr[:, 4] > CONF_THRESH
        detections = []
        for i in range(arr.shape[0]):
            if not conf_mask[i]:
                continue
            cx, cy, w, h = arr[i, :4]
            conf = arr[i, 4]
            # 直接还原bbox
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            detections.append([x1, y1, x2, y2, conf, 0])  # 单类person
        boxes = [d[:4] for d in detections]
        confs = [d[4] for d in detections]
        indices = cv2.dnn.NMSBoxes(boxes, confs, CONF_THRESH, IOU_THRESH)
        if len(indices) == 0:
            return []
        if isinstance(indices, np.ndarray):
            indices = indices.flatten()
        else:
            indices = [i[0] if isinstance(i, (list, tuple, np.ndarray)) else i for i in indices]
        return [detections[i] for i in indices]

def draw_and_save(img, detections, save_path):
    for x1, y1, x2, y2, conf, cls_id in detections:
        label = f"person {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(save_path, img)

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"模型不存在: {os.path.abspath(MODEL_PATH)}")
        return
    if not os.path.exists(IMAGE_PATH):
        print(f"图片不存在: {os.path.abspath(IMAGE_PATH)}")
        return

    acl_res = AclLiteResource()
    acl_res.init()

    yolo = PersonYOLO(MODEL_PATH)
    yolo.init()

    frame = cv2.imread(IMAGE_PATH)

    t0 = time.time()
    img = yolo.preprocess(frame)
    t1 = time.time()
    pred = yolo.model.execute([img])
    t2 = time.time()
    detections = yolo.postprocess(pred)
    t3 = time.time()

    print(f"前处理耗时: {(t1-t0)*1000:.2f} ms")
    print(f"推理耗时: {(t2-t1)*1000:.2f} ms")
    print(f"后处理耗时: {(t3-t2)*1000:.2f} ms")

    print("检测结果:", detections)
    draw_and_save(frame.copy(), detections, OUTPUT_PATH)
    if SHOW_RESULT:
        vis = cv2.imread(OUTPUT_PATH)
        cv2.imshow("person", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
