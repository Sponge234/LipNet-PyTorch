"""
摄像头采集模块 - 封装OpenCV摄像头操作，提供帧采集接口和资源管理功能
"""
import cv2
import numpy as np
from typing import Optional, Tuple


class CameraError(Exception):
    """摄像头相关错误"""
    pass


class CameraCapture:
    """
    摄像头采集器，封装OpenCV摄像头操作
    """
    
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        """
        初始化摄像头采集器
        
        Args:
            camera_id: 摄像头设备ID，默认为0
            width: 目标宽度，默认640
            height: 目标高度，默认480
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self._is_opened = False
    
    def open(self) -> bool:
        """
        打开摄像头
        
        Returns:
            bool: 是否成功打开
            
        Raises:
            CameraError: 摄像头打开失败时抛出
        """
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            self.cap = None
            raise CameraError(
                f"无法打开摄像头 {self.camera_id}\n"
                "可能的原因:\n"
                "  1. 摄像头设备不存在或被其他程序占用\n"
                "  2. 摄像头权限不足\n"
                "  3. 摄像头驱动未正确安装\n"
                "请检查摄像头连接并重试"
            )
        
        # 设置摄像头分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # 验证实际分辨率
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self._is_opened = True
        print(f"✓ 摄像头 {self.camera_id} 已打开")
        print(f"  请求分辨率: {self.width}x{self.height}")
        print(f"  实际分辨率: {actual_width}x{actual_height}")
        
        return True
    
    def read(self) -> Optional[np.ndarray]:
        """
        读取一帧
        
        Returns:
            np.ndarray: 视频帧，读取失败返回None
        """
        if self.cap is None or not self._is_opened:
            return None
        
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def close(self):
        """关闭摄像头并释放资源"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._is_opened = False
        print("✓ 摄像头已关闭")
    
    def is_opened(self) -> bool:
        """
        检查摄像头是否已打开
        
        Returns:
            bool: 是否已打开
        """
        return self._is_opened and self.cap is not None and self.cap.isOpened()
    
    def get_fps(self) -> float:
        """
        获取摄像头帧率
        
        Returns:
            float: 帧率
        """
        if self.cap is None:
            return 30.0
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else 30.0
    
    def get_resolution(self) -> Tuple[int, int]:
        """
        获取当前分辨率
        
        Returns:
            Tuple[int, int]: (宽度, 高度)
        """
        if self.cap is None:
            return (self.width, self.height)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)
    
    def __enter__(self) -> 'CameraCapture':
        """上下文管理器入口"""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
    
    def __repr__(self) -> str:
        return f"CameraCapture(camera_id={self.camera_id}, resolution={self.width}x{self.height})"


if __name__ == "__main__":
    # 测试摄像头采集器
    print("=" * 50)
    print("测试摄像头采集器")
    print("=" * 50)
    
    try:
        with CameraCapture(camera_id=0) as camera:
            print(f"\n摄像头信息:")
            print(f"  帧率: {camera.get_fps():.2f} FPS")
            print(f"  分辨率: {camera.get_resolution()}")
            
            print("\n按ESC键退出，按空格键截图...")
            
            frame_count = 0
            while True:
                frame = camera.read()
                if frame is not None:
                    frame_count += 1
                    
                    # 显示帧计数
                    cv2.putText(frame, f"Frame: {frame_count}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 255, 0), 2)
                    
                    cv2.imshow('Camera Test', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        break
                    elif key == 32:  # 空格
                        filename = f"capture_{frame_count}.jpg"
                        cv2.imwrite(filename, frame)
                        print(f"截图已保存: {filename}")
            
            print(f"\n总共读取 {frame_count} 帧")
    
    except CameraError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"未知错误: {e}")
    finally:
        cv2.destroyAllWindows()
