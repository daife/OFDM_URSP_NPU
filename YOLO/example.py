import cv2
import numpy as np
import acl
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
        """后处理"""
        CONF_THRESH = 0.01
        IOU_THRESH = 0.3
        # 打印实际 shape
        print("pred[0] shape:", pred[0].shape)
        arr = pred[0]
        # 兼容不同格式输出
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
            
        # 处理不同输出格式
        if arr.shape[0] == 84 and arr.shape[1] == 8400:
            arr = arr.transpose(1, 0)  # (8400, 84)
        elif arr.shape[0] == 8400 and arr.shape[1] == 84:
            pass  # already correct
        elif arr.shape[0] == 8 and arr.shape[1] == 8400:
            arr = arr.transpose(1, 0)  # (8400, 8)
        elif arr.shape[0] == 8400 and arr.shape[1] == 8:
            pass  # already correct
        else:
            raise ValueError(f"Unexpected pred shape: {arr.shape}")
            
        conf_mask = arr[:, 4] > CONF_THRESH
        detections = []
        for i in range(arr.shape[0]):
            if not conf_mask[i]:
                continue
            cx, cy, w, h = arr[i, :4]
            conf = arr[i, 4]
            # 根据实际输出维度获取类别分数
            cls_scores = arr[i, 5:] if arr.shape[1] > 5 else np.array([1.0])  # 如果没有类别分数，设为1.0
            
            # 正确还原到原图坐标
            cx = (cx - pad_info[1]) / pad_info[2]
            cy = (cy - pad_info[0]) / pad_info[2]
            w = w / pad_info[2]
            h = h / pad_info[2]
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            class_id = np.argmax(cls_scores)
            detections.append([x1, y1, x2, y2, conf, class_id])
        # NMS处理
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

def main():
    # 配置参数
    MODEL_PATH = "yolo11sss.om"
    
    # 添加调试信息：检查模型文件是否存在
    import os
    if os.path.exists(MODEL_PATH):
        print(f"模型文件存在: {os.path.abspath(MODEL_PATH)}")
    else:
        print(f"错误: 模型文件不存在: {os.path.abspath(MODEL_PATH)}")
        return

    # 初始化ACL资源
    acl_resource = AclLiteResource()
    acl_resource.init()

    # 初始化YOLO11s模型
    yolo11s = YOLO11s(MODEL_PATH, correct_distortion=True)
    
    try:
        print("正在初始化模型...")
        yolo11s.init()
        print("模型初始化成功!")
        
        # 打开摄像头并设置分辨率为640x480
        cap = cv2.VideoCapture(0)
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
                    label = f"class{cls_id} {conf:.2f}"
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