"""
设备管理模块 - 负责GPU/CPU设备的选择和管理
"""
import torch
from typing import Optional, Tuple


class DeviceManager:
    """
    设备管理器，负责GPU/CPU设备的选择、查询和切换
    """
    
    def __init__(self, use_gpu: bool = True, device_id: int = 0):
        """
        初始化设备管理器
        
        Args:
            use_gpu: 是否使用GPU，默认为True
            device_id: GPU设备ID，默认为0
        """
        self.use_gpu = use_gpu
        self.device_id = device_id
        self.device = self._select_device()
        
    def _select_device(self) -> torch.device:
        """
        选择并返回计算设备
        
        Returns:
            torch.device: 选中的设备对象
        """
        if self.use_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.device_id}')
            device_name = torch.cuda.get_device_name(device)
            print(f"✓ 使用GPU设备: {device_name}")
        else:
            device = torch.device('cpu')
            if self.use_gpu:
                print("⚠ GPU不可用，自动降级到CPU")
            else:
                print("✓ 使用CPU设备")
        return device
    
    def get_device(self) -> torch.device:
        """
        获取当前设备对象
        
        Returns:
            torch.device: 当前设备对象
        """
        return self.device
    
    def get_device_name(self) -> str:
        """
        获取当前设备名称
        
        Returns:
            str: 设备名称
        """
        if self.device.type == 'cuda':
            return torch.cuda.get_device_name(self.device)
        return 'CPU'
    
    def get_device_info(self) -> dict:
        """
        获取设备详细信息
        
        Returns:
            dict: 设备信息字典
        """
        info = {
            'device_type': self.device.type,
            'device_name': self.get_device_name(),
        }
        
        if self.device.type == 'cuda':
            info.update({
                'device_id': self.device_id,
                'total_memory': torch.cuda.get_device_properties(self.device).total_memory / 1024**3,
                'allocated_memory': torch.cuda.memory_allocated(self.device) / 1024**3,
                'cached_memory': torch.cuda.memory_reserved(self.device) / 1024**3,
            })
        
        return info
    
    def print_device_info(self):
        """打印设备信息"""
        info = self.get_device_info()
        print("\n" + "="*50)
        print("设备信息")
        print("="*50)
        print(f"设备类型: {info['device_type'].upper()}")
        print(f"设备名称: {info['device_name']}")
        
        if self.device.type == 'cuda':
            print(f"设备ID: {info['device_id']}")
            print(f"总显存: {info['total_memory']:.2f} GB")
            print(f"已分配显存: {info['allocated_memory']:.2f} GB")
            print(f"缓存显存: {info['cached_memory']:.2f} GB")
        print("="*50 + "\n")
    
    def to_device(self, model_or_tensor):
        """
        将模型或张量移动到当前设备
        
        Args:
            model_or_tensor: PyTorch模型或张量
            
        Returns:
            移动到当前设备后的对象
        """
        return model_or_tensor.to(self.device)
    
    def clear_cache(self):
        """清理GPU缓存"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            print("✓ GPU缓存已清理")
    
    @staticmethod
    def is_gpu_available() -> bool:
        """
        检查GPU是否可用
        
        Returns:
            bool: GPU是否可用
        """
        return torch.cuda.is_available()
    
    @staticmethod
    def get_available_gpus() -> int:
        """
        获取可用GPU数量
        
        Returns:
            int: 可用GPU数量
        """
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    @staticmethod
    def list_all_gpus():
        """列出所有可用的GPU设备"""
        if not torch.cuda.is_available():
            print("未检测到可用的GPU设备")
            return
        
        gpu_count = torch.cuda.device_count()
        print(f"\n检测到 {gpu_count} 个GPU设备:")
        print("-" * 50)
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  显存: {props.total_memory / 1024**3:.2f} GB")
            print(f"  计算能力: {props.major}.{props.minor}")
        print("-" * 50)


if __name__ == "__main__":
    # 测试设备管理器
    print("测试设备管理器")
    DeviceManager.list_all_gpus()
    
    # 测试GPU模式
    print("\n测试GPU模式:")
    dm_gpu = DeviceManager(use_gpu=True)
    dm_gpu.print_device_info()
    
    # 测试CPU模式
    print("\n测试CPU模式:")
    dm_cpu = DeviceManager(use_gpu=False)
    dm_cpu.print_device_info()
