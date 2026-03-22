"""
模型加载和推理模块
"""
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from model import LipNet
from device_manager import DeviceManager


class LipNetModel:
    """
    LipNet模型加载和推理封装
    """
    
    # 字符映射表 (26个字母 + 空格 + blank)
    CHAR_MAP = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 
                'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 
                't', 'u', 'v', 'w', 'x', 'y', 'z']
    
    def __init__(self, model_path: str, device_manager: Optional[DeviceManager] = None):
        """
        初始化LipNet模型
        
        Args:
            model_path: 模型权重文件路径
            device_manager: 设备管理器实例
        """
        self.model_path = model_path
        self.device_manager = device_manager or DeviceManager()
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 创建模型
        self.model = LipNet()
        
        # 加载权重
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        # 移动到设备
        self.model = self.device_manager.to_device(self.model)
        
        # 设置为评估模式
        self.model.eval()
        
        print(f"✓ 成功加载模型: {self.model_path}")
    
    def preprocess(self, lip_frames: np.ndarray) -> torch.Tensor:
        """
        预处理唇部帧序列
        
        Args:
            lip_frames: 唇部帧序列 (num_frames, height, width, channels)
            
        Returns:
            预处理后的张量 (1, channels, num_frames, height, width)
        """
        # 归一化到 [0, 1]
        frames = lip_frames.astype(np.float32) / 255.0
        
        # 转换为张量 (num_frames, height, width, channels) -> (channels, num_frames, height, width)
        frames = np.transpose(frames, (3, 0, 1, 2))
        
        # 添加batch维度 (1, channels, num_frames, height, width)
        frames = np.expand_dims(frames, axis=0)
        
        return torch.from_numpy(frames)
    
    def infer(self, lip_frames: np.ndarray) -> Tuple[str, np.ndarray]:
        """
        执行推理
        
        Args:
            lip_frames: 唇部帧序列 (num_frames, height, width, channels)
            
        Returns:
            (预测文本, 概率序列)
        """
        # 预处理
        input_tensor = self.preprocess(lip_frames)
        input_tensor = self.device_manager.to_device(input_tensor)
        
        # 推理
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # 转移到CPU
        output = output.cpu().numpy()
        
        # 解码
        text = self.decode(output[0])
        
        return text, output
    
    def decode(self, output: np.ndarray) -> str:
        """
        CTC贪婪解码
        
        Args:
            output: 模型输出 (time_steps, num_classes)
            
        Returns:
            解码后的文本
        """
        # 获取每个时间步的最大概率索引
        pred_indices = np.argmax(output, axis=1)
        
        # CTC解码：移除重复和blank
        decoded = []
        prev_idx = -1
        
        for idx in pred_indices:
            # 跳过blank (索引0) 和重复
            if idx != 0 and idx != prev_idx:
                if 0 < idx < len(self.CHAR_MAP):
                    decoded.append(self.CHAR_MAP[idx])
            prev_idx = idx
        
        return ''.join(decoded)
    
    def get_confidence(self, output: np.ndarray) -> float:
        """
        计算预测置信度
        
        Args:
            output: 模型输出 (time_steps, num_classes)
            
        Returns:
            平均置信度
        """
        # 计算每个时间步的最大概率
        probs = np.exp(output - np.max(output, axis=1, keepdims=True))
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        max_probs = np.max(probs, axis=1)
        
        return float(np.mean(max_probs))


class TextDecoder:
    """
    文本解码器，支持多种解码方式
    """
    
    @staticmethod
    def ctc_greedy_decode(output: np.ndarray, char_map: list) -> str:
        """
        CTC贪婪解码
        
        Args:
            output: 模型输出 (time_steps, num_classes)
            char_map: 字符映射表
            
        Returns:
            解码后的文本
        """
        pred_indices = np.argmax(output, axis=1)
        
        decoded = []
        prev_idx = -1
        
        for idx in pred_indices:
            if idx != 0 and idx != prev_idx:
                if 0 < idx < len(char_map):
                    decoded.append(char_map[idx])
            prev_idx = idx
        
        return ''.join(decoded)
    
    @staticmethod
    def ctc_beam_search_decode(output: np.ndarray, char_map: list, beam_width: int = 10) -> str:
        """
        CTC束搜索解码（简化版）
        
        Args:
            output: 模型输出 (time_steps, num_classes)
            char_map: 字符映射表
            beam_width: 束宽度
            
        Returns:
            解码后的文本
        """
        # 这里实现简化版，实际应用中可以使用更复杂的束搜索
        return TextDecoder.ctc_greedy_decode(output, char_map)


if __name__ == "__main__":
    # 测试模型加载
    print("测试模型加载")
    
    # 检查是否有可用的模型文件
    model_paths = [
        "models/LipNet_405.pkl",
        "models/LipNet_407.pkl"
    ]
    
    for model_path in model_paths:
        if Path(model_path).exists():
            print(f"\n加载模型: {model_path}")
            try:
                model = LipNetModel(model_path)
                print("模型加载成功!")
                break
            except Exception as e:
                print(f"模型加载失败: {e}")
    else:
        print("\n未找到模型文件，请先下载预训练模型")
