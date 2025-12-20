import cv2
import numpy as np
import acl
import os
import pyudev  # 新增
from acllite_utils import *
from constants import *
from acllite_imageproc import AclLiteImageProc
from acllite_model import AclLiteModel
from acllite_resource import resource_list

# Camera intrinsic parameters
CAMERA_MATRIX = np.array([
    [465.13093,   0.     , 324.81802],
    [  0.     , 466.33628, 242.54136],
    [  0.     ,   0.     ,   1.     ]
])
DISTORTION_COEFFS = np.array([-0.374992, 0.133505, 0.002906, -0.002975, 0.000000])
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# 宏定义摄像头Vendor/Model ID
CAMERA_VENDOR_ID = "0c45"
CAMERA_MODEL_ID = "6368"

class AclLiteResource:
    """ACL资源管理类"""
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

class YOLO11s:
    """YOLO11s模型处理类"""
    def __init__(self, model_path, input_size=640, correct_distortion=True):
        self.model_path = model_path
        self.input_size = input_size
        self.model = None
        self.dvpp = None
        self.correct_distortion = correct_distortion
        self.camera_matrix = CAMERA_MATRIX
        self.dist_coeffs = DISTORTION_COEFFS

    def init(self):
        """初始化模型和图像处理器"""
        self.dvpp = AclLiteImageProc()
        self.model = AclLiteModel(self.model_path)
        return const.SUCCESS

    def preprocess(self, frame):
        """图像预处理"""
        # 畸变校正
        undistorted_frame = None
        if self.correct_distortion:
            undistorted_frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
            frame = undistorted_frame
        
        # 保持宽高比缩放
        h, w = frame.shape[:2]
        scale = min(self.input_size/w, self.input_size/h)
        nh, nw = int(h*scale), int(w*scale)
        img = cv2.resize(frame, (nw, nh))
        
        # 填充灰边
        top = (self.input_size - nh) // 2
        bottom = self.input_size - nh - top
        left = (self.input_size - nw) // 2
        right = self.input_size - nw - left
        img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                                cv2.BORDER_CONSTANT, value=(114,114,114))
        
        # 归一化并转换格式
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2,0,1)[np.newaxis].astype(np.float32)/255
        return img, (h, w), (top, left, scale), undistorted_frame

    def postprocess(self, pred, orig_shape, pad_info):
        """后处理：每个类别只保留置信度最高的一个检测框"""
        CONF_THRESH = 0.1
        IOU_THRESH = 0.9
        arr = pred[0]
        # 只保留(8400, 8)格式
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        # 只考虑(8400, 8)格式
        assert arr.shape == (8400, 8), f"Unexpected pred shape: {arr.shape}"
        detections = []
        for i in range(arr.shape[0]):
            conf = arr[i, 4]
            if conf < CONF_THRESH:
                continue
            cx, cy, w, h = arr[i, :4]
            cls_scores = arr[i, 5:9]
            class_id = np.argmax(cls_scores)
            score = cls_scores[class_id]
            if score < CONF_THRESH:
                continue
            cx = (cx - pad_info[1]) / pad_info[2]
            cy = (cy - pad_info[0]) / pad_info[2]
            w = w / pad_info[2]
            h = h / pad_info[2]
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            detections.append([x1, y1, x2, y2, score, class_id])
        # 对每个类别保留分数最高的一个
        best_per_class = {}
        for det in detections:
            x1, y1, x2, y2, score, class_id = det
            if (class_id not in best_per_class) or (score > best_per_class[class_id][4]):
                best_per_class[class_id] = det
        return list(best_per_class.values())

def find_camera_by_vid_pid(vendor_id, product_id):
    context = pyudev.Context()
    for device in context.list_devices(subsystem='video4linux'):
        if device.device_node and device.device_node.startswith('/dev/video'):
            vid = device.get('ID_VENDOR_ID', '').lower()
            pid = device.get('ID_MODEL_ID', '').lower()
            if vid == vendor_id.lower() and pid == product_id.lower():
                return device.device_node
    return None

def main():
    # 配置参数
    #MODEL_PATH = "yolo11s16.om"
    MODEL_PATH = "/home/HwHiAiUser/yolo_test/yolo11n-new.om"
    
    # 添加调试信息：检查模型文件是否存在
    if os.path.exists(MODEL_PATH):
        print(f"模型文件存在: {os.path.abspath(MODEL_PATH)}")
    else:
        print(f"错误: 模型文件不存在: {os.path.abspath(MODEL_PATH)}")
        return

    # 查找指定Vendor/Model ID的摄像头
    camera_dev = find_camera_by_vid_pid(CAMERA_VENDOR_ID, CAMERA_MODEL_ID)
    if camera_dev is None:
        print("未找到指定摄像头设备")
        return
    print(f"使用摄像头设备: {camera_dev}")

    # 初始化ACL资源
    acl_resource = AclLiteResource()
    acl_resource.init()

    # 初始化YOLO11s模型
    yolo11s = YOLO11s(MODEL_PATH, correct_distortion=True)
    
    try:
        print("正在初始化模型...")
        yolo11s.init()
        print("模型初始化成功!")
        
        # 打开指定摄像头并设置分辨率为640x480
        cap = cv2.VideoCapture(camera_dev)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 预处理
                img, orig_shape, pad_info, undistorted_frame = yolo11s.preprocess(frame)
                
                # 执行推理
                pred = yolo11s.model.execute([img])
                
                # 后处理
                detections = yolo11s.postprocess(pred, orig_shape, pad_info)
                
                # 使用校正后的图像来显示
                display_frame = undistorted_frame if undistorted_frame is not None else frame
                
                # 绘制结果
                for det in detections:
                    x1, y1, x2, y2, conf, cls_id = det
                    label = f"{cls_id} {conf:.2f}"  # 类别用0,1,2,3
                    cv2.rectangle(display_frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(display_frame, label, (x1,y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                
                cv2.imshow("YOLO11s Detection", display_frame)
                if cv2.waitKey(1) == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
    except Exception as e:
        print(f"模型初始化失败: {e}")

if __name__ == "__main__":
    main()