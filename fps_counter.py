"""
FPS计数器模块 - 实现实时FPS计算功能
"""
import time
from typing import Optional


class FPSCounter:
    """
    FPS计数器，计算实时帧率
    """
    
    def __init__(self, update_interval: float = 1.0):
        """
        初始化FPS计数器
        
        Args:
            update_interval: FPS更新间隔（秒），默认1.0秒更新一次
        """
        self.update_interval = update_interval
        self._start_time: Optional[float] = None
        self._last_update_time: Optional[float] = None
        self._frame_count = 0
        self._fps = 0.0
    
    def start(self) -> 'FPSCounter':
        """
        启动计数器
        
        Returns:
            FPSCounter: self，支持链式调用
        """
        self._start_time = time.time()
        self._last_update_time = self._start_time
        self._frame_count = 0
        self._fps = 0.0
        return self
    
    def tick(self) -> float:
        """
        记录一帧并返回当前FPS
        
        Returns:
            float: 当前FPS
        """
        if self._start_time is None:
            self.start()
        
        self._frame_count += 1
        current_time = time.time()
        elapsed = current_time - self._last_update_time
        
        # 每隔update_interval更新一次FPS
        if elapsed >= self.update_interval:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_update_time = current_time
        
        return self._fps
    
    def get_fps(self) -> float:
        """
        获取当前FPS
        
        Returns:
            float: 当前FPS
        """
        return self._fps
    
    def get_elapsed_time(self) -> float:
        """
        获取从启动到现在经过的时间
        
        Returns:
            float: 经过的秒数
        """
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time
    
    def reset(self):
        """重置计数器"""
        self._start_time = None
        self._last_update_time = None
        self._frame_count = 0
        self._fps = 0.0
    
    def __repr__(self) -> str:
        return f"FPSCounter(fps={self._fps:.2f}, update_interval={self.update_interval}s)"


class AverageFPSCounter:
    """
    平均FPS计数器，计算一段时间内的平均帧率
    """
    
    def __init__(self, window_size: int = 30):
        """
        初始化平均FPS计数器
        
        Args:
            window_size: 滑动窗口大小，默认30帧
        """
        self.window_size = window_size
        self._timestamps: list = []
        self._fps = 0.0
    
    def tick(self) -> float:
        """
        记录一帧并返回当前平均FPS
        
        Returns:
            float: 当前平均FPS
        """
        current_time = time.time()
        self._timestamps.append(current_time)
        
        # 保持窗口大小
        if len(self._timestamps) > self.window_size:
            self._timestamps.pop(0)
        
        # 计算FPS
        if len(self._timestamps) >= 2:
            elapsed = self._timestamps[-1] - self._timestamps[0]
            if elapsed > 0:
                self._fps = (len(self._timestamps) - 1) / elapsed
        
        return self._fps
    
    def get_fps(self) -> float:
        """
        获取当前平均FPS
        
        Returns:
            float: 当前平均FPS
        """
        return self._fps
    
    def reset(self):
        """重置计数器"""
        self._timestamps = []
        self._fps = 0.0


if __name__ == "__main__":
    # 测试FPS计数器
    print("=" * 50)
    print("测试FPS计数器")
    print("=" * 50)
    
    import random
    
    # 测试FPSCounter
    print("\n测试 FPSCounter:")
    counter = FPSCounter(update_interval=1.0).start()
    
    for i in range(100):
        fps = counter.tick()
        # 模拟处理时间
        time.sleep(random.uniform(0.01, 0.05))
        
        if i % 10 == 0:
            print(f"  帧 {i}: FPS = {fps:.2f}")
    
    print(f"\n最终FPS: {counter.get_fps():.2f}")
    print(f"运行时间: {counter.get_elapsed_time():.2f}秒")
    
    # 测试AverageFPSCounter
    print("\n测试 AverageFPSCounter:")
    avg_counter = AverageFPSCounter(window_size=30)
    
    for i in range(100):
        fps = avg_counter.tick()
        time.sleep(random.uniform(0.01, 0.05))
        
        if i % 10 == 0:
            print(f"  帧 {i}: 平均FPS = {fps:.2f}")
    
    print(f"\n最终平均FPS: {avg_counter.get_fps():.2f}")
