"""
视频处理模块 - 负责视频读取、唇部检测和帧提取
"""
import cv2
import numpy as np
import dlib
from typing import Optional, Tuple, List, Generator
from pathlib import Path


class LipDetector:
    """
    唇部检测器，使用dlib进行人脸和唇部关键点检测
    """
    
    def __init__(self, predictor_path: str = "shape_predictor_68_face_landmarks.dat"):
        """
        初始化唇部检测器
        
        Args:
            predictor_path: dlib关键点预测器文件路径
        """
        self.detector = dlib.get_frontal_face_detector()
        
        # 检查预测器文件是否存在
        if not Path(predictor_path).exists():
            raise FileNotFoundError(
                f"未找到关键点预测器文件: {predictor_path}\n"
                "请从 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 下载"
            )
        
        self.predictor = dlib.shape_predictor(predictor_path)
        print(f"✓ 唇部检测器初始化成功")
    
    def detect_lip_region(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        检测帧中的唇部区域
        
        Args:
            frame: 输入图像帧
            
        Returns:
            唇部区域 (x, y, w, h) 或 None
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)
        
        if len(faces) == 0:
            return None
        
        # 使用第一个检测到的人脸
        shape = self.predictor(gray, faces[0])
        
        # 提取唇部关键点 (48-67为唇部)
        lip_points = []
        for i in range(48, 68):
            lip_points.append((shape.part(i).x, shape.part(i).y))
        
        # 计算唇部边界框
        lip_points = np.array(lip_points)
        x, y, w, h = cv2.boundingRect(lip_points)
        
        # 扩展边界框，增加一些边距
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(frame.shape[1] - x, w + 2 * margin)
        h = min(frame.shape[0] - y, h + 2 * margin)
        
        return (x, y, w, h)
    
    def extract_lip(self, frame: np.ndarray, target_size: Tuple[int, int] = (128, 64)) -> Optional[np.ndarray]:
        """
        从帧中提取并调整唇部图像
        
        Args:
            frame: 输入图像帧
            target_size: 目标尺寸 (width, height)
            
        Returns:
            调整后的唇部图像或None
        """
        lip_region = self.detect_lip_region(frame)
        
        if lip_region is None:
            return None
        
        x, y, w, h = lip_region
        lip_img = frame[y:y+h, x:x+w]
        
        # 调整到目标尺寸
        lip_img = cv2.resize(lip_img, target_size)
        
        return lip_img


class VideoProcessor:
    """
    视频处理器基类
    """
    
    def __init__(self, lip_detector: Optional[LipDetector] = None):
        """
        初始化视频处理器
        
        Args:
            lip_detector: 唇部检测器实例
        """
        self.lip_detector = lip_detector or LipDetector()
        self.cap = None
    
    def open(self) -> bool:
        """打开视频源"""
        raise NotImplementedError
    
    def close(self):
        """关闭视频源"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def read_frame(self) -> Optional[np.ndarray]:
        """读取一帧"""
        raise NotImplementedError
    
    def get_fps(self) -> float:
        """获取帧率"""
        raise NotImplementedError
    
    def get_frame_count(self) -> int:
        """获取总帧数"""
        raise NotImplementedError
    
    def extract_lip_frames(self, num_frames: int = 75) -> Optional[np.ndarray]:
        """
        提取指定数量的唇部帧
        
        Args:
            num_frames: 需要提取的帧数
            
        Returns:
            唇部帧序列 (num_frames, height, width, channels) 或 None
        """
        lip_frames = []
        frame_count = 0
        
        while len(lip_frames) < num_frames:
            frame = self.read_frame()
            
            if frame is None:
                break
            
            lip_img = self.lip_detector.extract_lip(frame)
            
            if lip_img is not None:
                lip_frames.append(lip_img)
            
            frame_count += 1
        
        if len(lip_frames) == 0:
            return None
        
        # 如果帧数不足，复制最后一帧
        while len(lip_frames) < num_frames:
            lip_frames.append(lip_frames[-1].copy())
        
        return np.array(lip_frames[:num_frames])
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class MP4VideoProcessor(VideoProcessor):
    """
    MP4视频文件处理器
    """
    
    def __init__(self, video_path: str, lip_detector: Optional[LipDetector] = None):
        """
        初始化MP4视频处理器
        
        Args:
            video_path: MP4视频文件路径
            lip_detector: 唇部检测器实例
        """
        super().__init__(lip_detector)
        self.video_path = video_path
        
        if not Path(video_path).exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    def open(self) -> bool:
        """打开视频文件"""
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            print(f"✗ 无法打开视频文件: {self.video_path}")
            return False
        
        print(f"✓ 成功打开视频文件: {self.video_path}")
        print(f"  分辨率: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"  帧率: {self.get_fps():.2f} FPS")
        print(f"  总帧数: {self.get_frame_count()}")
        return True
    
    def read_frame(self) -> Optional[np.ndarray]:
        """读取一帧"""
        if self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def get_fps(self) -> float:
        """获取帧率"""
        if self.cap is None:
            return 0.0
        return self.cap.get(cv2.CAP_PROP_FPS)
    
    def get_frame_count(self) -> int:
        """获取总帧数"""
        if self.cap is None:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))


class CameraVideoProcessor(VideoProcessor):
    """
    摄像头视频处理器
    """
    
    def __init__(self, camera_id: int = 0, lip_detector: Optional[LipDetector] = None):
        """
        初始化摄像头处理器
        
        Args:
            camera_id: 摄像头设备ID，默认为0
            lip_detector: 唇部检测器实例
        """
        super().__init__(lip_detector)
        self.camera_id = camera_id
    
    def open(self) -> bool:
        """打开摄像头"""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"✗ 无法打开摄像头 {self.camera_id}")
            return False
        
        # 设置摄像头分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"✓ 成功打开摄像头 {self.camera_id}")
        print(f"  分辨率: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        return True
    
    def read_frame(self) -> Optional[np.ndarray]:
        """读取一帧"""
        if self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def get_fps(self) -> float:
        """获取帧率"""
        if self.cap is None:
            return 30.0  # 默认30fps
        return self.cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    def get_frame_count(self) -> int:
        """摄像头没有固定帧数"""
        return -1  # 表示无限


if __name__ == "__main__":
    # 测试视频处理器
    print("测试视频处理器")
    
    # 测试摄像头
    print("\n测试摄像头:")
    try:
        camera = CameraVideoProcessor(camera_id=0)
        if camera.open():
            print("摄像头已打开，按ESC键退出...")
            
            while True:
                frame = camera.read_frame()
                if frame is not None:
                    cv2.imshow('Camera', frame)
                    
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC键
                        break
                
            camera.close()
            cv2.destroyAllWindows()
    except Exception as e:
        print(f"摄像头测试失败: {e}")
