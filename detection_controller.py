"""
检测控制器模块 - 管理检测流程、帧缓冲区和推理调度
"""
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from video_processor import LipDetector
from model_wrapper import LipNetModel


@dataclass
class DetectionResult:
    """检测结果数据类"""
    text: Optional[str] = None
    confidence: float = 0.0
    lip_region: Optional[Tuple[int, int, int, int]] = None
    frame_id: int = 0
    timestamp: float = 0.0


@dataclass
class FrameBuffer:
    """
    帧缓冲区数据类
    """
    frames: List[np.ndarray] = field(default_factory=list)
    max_size: int = 75
    
    def add(self, frame: np.ndarray):
        """
        添加帧到缓冲区
        
        Args:
            frame: 唇部帧图像
        """
        self.frames.append(frame.copy())
        # 如果超过最大大小，移除最旧的帧
        if len(self.frames) > self.max_size:
            self.frames.pop(0)
    
    def is_full(self) -> bool:
        """
        检查缓冲区是否已满
        
        Returns:
            bool: 是否已满
        """
        return len(self.frames) >= self.max_size
    
    def get_frames(self) -> Optional[np.ndarray]:
        """
        获取缓冲区中的所有帧
        
        Returns:
            np.ndarray: 帧序列或None
        """
        if len(self.frames) == 0:
            return None
        return np.array(self.frames)
    
    def clear(self):
        """清空缓冲区"""
        self.frames.clear()
    
    def clear_half(self):
        """清空一半的帧（滑动窗口）"""
        half = len(self.frames) // 2
        self.frames = self.frames[half:]
    
    def get_size(self) -> int:
        """
        获取当前缓冲区大小
        
        Returns:
            int: 帧数量
        """
        return len(self.frames)
    
    def get_progress(self) -> float:
        """
        获取缓冲区填充进度
        
        Returns:
            float: 0.0 到 1.0
        """
        return len(self.frames) / self.max_size


class DetectionController:
    """
    检测控制器，管理唇部检测和模型推理
    """
    
    def __init__(
        self,
        model: LipNetModel,
        lip_detector: LipDetector,
        num_frames: int = 75
    ):
        """
        初始化检测控制器
        
        Args:
            model: LipNet模型实例
            lip_detector: 唇部检测器实例
            num_frames: 推理所需的帧数，默认75
        """
        self.model = model
        self.lip_detector = lip_detector
        self.num_frames = num_frames
        
        # 帧缓冲区
        self.buffer = FrameBuffer(max_size=num_frames)
        
        # 状态
        self.is_paused = False
        self.frame_count = 0
        self.last_result: Optional[DetectionResult] = None
        
        # 统计信息
        self.total_detections = 0
        self.successful_detections = 0
    
    def process_frame(
        self,
        frame: np.ndarray
    ) -> Tuple[Optional[str], float, Optional[Tuple[int, int, int, int]]]:
        """
        处理单帧图像
        
        Args:
            frame: 输入图像帧
            
        Returns:
            Tuple[Optional[str], float, Optional[Tuple]]: 
                (识别文本, 置信度, 唇部区域)
        """
        if self.is_paused:
            # 暂停时返回上一次的结果
            if self.last_result:
                return (
                    self.last_result.text,
                    self.last_result.confidence,
                    self.last_result.lip_region
                )
            return (None, 0.0, None)
        
        self.frame_count += 1
        
        # 检测唇部区域
        lip_region = self.lip_detector.detect_lip_region(frame)
        
        if lip_region is None:
            # 未检测到人脸，返回空结果
            return (None, 0.0, None)
        
        # 提取唇部图像
        lip_img = self.lip_detector.extract_lip(frame)
        
        if lip_img is None:
            return (None, 0.0, lip_region)
        
        # 添加到缓冲区
        self.buffer.add(lip_img)
        
        # 检查是否需要推理
        text = None
        confidence = 0.0
        
        if self.buffer.is_full():
            # 执行推理
            frames = self.buffer.get_frames()
            
            if frames is not None:
                try:
                    # 推理
                    pred_text, output = self.model.infer(frames)
                    confidence = self.model.get_confidence(output)
                    
                    text = pred_text
                    self.total_detections += 1
                    
                    if text and len(text) > 0:
                        self.successful_detections += 1
                    
                    # 保存结果
                    self.last_result = DetectionResult(
                        text=text,
                        confidence=confidence,
                        lip_region=lip_region,
                        frame_id=self.frame_count,
                        timestamp=time.time()
                    )
                    
                    # 清空一半缓冲区（滑动窗口）
                    self.buffer.clear_half()
                    
                except Exception as e:
                    print(f"推理错误: {e}")
        
        # 如果有上次的结果，继续显示
        if text is None and self.last_result:
            text = self.last_result.text
            confidence = self.last_result.confidence
        
        return (text, confidence, lip_region)
    
    def toggle_pause(self) -> bool:
        """
        切换暂停状态
        
        Returns:
            bool: 新的暂停状态
        """
        self.is_paused = not self.is_paused
        return self.is_paused
    
    def pause(self):
        """暂停检测"""
        self.is_paused = True
    
    def resume(self):
        """继续检测"""
        self.is_paused = False
    
    def reset(self):
        """重置控制器状态"""
        self.buffer.clear()
        self.frame_count = 0
        self.last_result = None
        self.is_paused = False
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """
        获取缓冲区状态
        
        Returns:
            Dict: 缓冲区状态信息
        """
        return {
            'size': self.buffer.get_size(),
            'max_size': self.buffer.max_size,
            'progress': self.buffer.get_progress(),
            'is_full': self.buffer.is_full()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            Dict: 统计信息
        """
        return {
            'frame_count': self.frame_count,
            'total_detections': self.total_detections,
            'successful_detections': self.successful_detections,
            'success_rate': (
                self.successful_detections / self.total_detections 
                if self.total_detections > 0 else 0.0
            )
        }
    
    def __repr__(self) -> str:
        return (
            f"DetectionController("
            f"frames={self.buffer.get_size()}/{self.num_frames}, "
            f"paused={self.is_paused})"
        )


if __name__ == "__main__":
    # 测试检测控制器
    print("=" * 50)
    print("测试检测控制器")
    print("=" * 50)
    
    # 测试FrameBuffer
    print("\n测试 FrameBuffer:")
    buffer = FrameBuffer(max_size=10)
    
    # 添加帧
    for i in range(15):
        frame = np.random.randint(0, 255, (64, 128, 3), dtype=np.uint8)
        buffer.add(frame)
        print(f"  添加帧 {i}: 大小={buffer.get_size()}, 进度={buffer.get_progress():.2f}")
    
    print(f"\n缓冲区是否已满: {buffer.is_full()}")
    print(f"清空一半后...")
    buffer.clear_half()
    print(f"当前大小: {buffer.get_size()}")
    
    # 测试DetectionResult
    print("\n测试 DetectionResult:")
    result = DetectionResult(
        text="hello",
        confidence=0.85,
        lip_region=(100, 100, 50, 30),
        frame_id=100,
        timestamp=time.time()
    )
    print(f"  结果: {result}")
