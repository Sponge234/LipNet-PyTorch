"""
实时唇语检测GUI模块 - 实现摄像头实时检测和结果显示
"""
import cv2
import numpy as np
import torch
from typing import Optional, Tuple
from pathlib import Path

from camera_capture import CameraCapture, CameraError
from fps_counter import FPSCounter
from detection_controller import DetectionController
from video_processor import LipDetector
from model_wrapper import LipNetModel
from device_manager import DeviceManager


class RealtimeLipNetGUI:
    """
    实时唇语检测GUI应用
    """
    
    # 窗口标题
    WINDOW_TITLE = "LipNet 实时唇语识别"
    
    # 颜色定义 (BGR格式)
    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (0, 0, 255)
    COLOR_YELLOW = (0, 255, 255)
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)
    COLOR_BLUE = (255, 0, 0)
    COLOR_CYAN = (255, 255, 0)
    
    def __init__(
        self,
        model_path: str,
        camera_id: int = 0,
        use_gpu: bool = True,
        predictor_path: str = "shape_predictor_68_face_landmarks.dat",
        num_frames: int = 75
    ):
        """
        初始化实时唇语检测GUI
        
        Args:
            model_path: 模型权重文件路径
            camera_id: 摄像头设备ID
            use_gpu: 是否使用GPU
            predictor_path: dlib关键点预测器文件路径
            num_frames: 推理所需的帧数
        """
        self.model_path = model_path
        self.camera_id = camera_id
        self.use_gpu = use_gpu
        self.predictor_path = predictor_path
        self.num_frames = num_frames
        
        # 组件
        self.device_manager: Optional[DeviceManager] = None
        self.lip_detector: Optional[LipDetector] = None
        self.model: Optional[LipNetModel] = None
        self.camera: Optional[CameraCapture] = None
        self.detection_controller: Optional[DetectionController] = None
        self.fps_counter: Optional[FPSCounter] = None
        
        # 状态
        self.is_running = False
        self.is_paused = False
        
        # 初始化组件
        self._init_components()
    
    def _init_components(self):
        """初始化所有组件"""
        print("\n" + "=" * 50)
        print("初始化 LipNet 实时唇语识别系统")
        print("=" * 50)
        
        # 1. 初始化设备管理器
        print("\n[1/5] 初始化设备管理器...")
        self.device_manager = DeviceManager(use_gpu=self.use_gpu)
        
        # 2. 初始化唇部检测器
        print("\n[2/5] 初始化唇部检测器...")
        if not Path(self.predictor_path).exists():
            raise FileNotFoundError(
                f"未找到关键点预测器文件: {self.predictor_path}\n"
                "请从 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 下载"
            )
        self.lip_detector = LipDetector(predictor_path=self.predictor_path)
        
        # 3. 初始化模型
        print("\n[3/5] 加载LipNet模型...")
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"未找到模型文件: {self.model_path}")
        self.model = LipNetModel(
            model_path=self.model_path,
            device_manager=self.device_manager
        )
        
        # 4. 初始化摄像头
        print("\n[4/5] 初始化摄像头...")
        self.camera = CameraCapture(camera_id=self.camera_id)
        
        # 5. 初始化检测控制器
        print("\n[5/5] 初始化检测控制器...")
        self.detection_controller = DetectionController(
            model=self.model,
            lip_detector=self.lip_detector,
            num_frames=self.num_frames
        )
        
        # 6. 初始化FPS计数器
        self.fps_counter = FPSCounter(update_interval=1.0)
        
        print("\n" + "=" * 50)
        print("✓ 所有组件初始化完成")
        print("=" * 50)
    
    def _overlay_info(
        self,
        frame: np.ndarray,
        text: Optional[str],
        confidence: float,
        fps: float,
        lip_region: Optional[Tuple[int, int, int, int]],
        is_paused: bool,
        buffer_progress: float
    ) -> np.ndarray:
        """
        在帧上叠加信息
        
        Args:
            frame: 输入帧
            text: 识别文本
            confidence: 置信度
            fps: 当前FPS
            lip_region: 唇部区域
            is_paused: 是否暂停
            buffer_progress: 缓冲区进度
            
        Returns:
            np.ndarray: 叠加信息后的帧
        """
        # 复制帧避免修改原图
        result = frame.copy()
        h, w = result.shape[:2]
        
        # 1. 绘制唇部标注框
        if lip_region is not None:
            x, y, rw, rh = lip_region
            cv2.rectangle(result, (x, y), (x + rw, y + rh), self.COLOR_GREEN, 2)
        
        # 2. 绘制识别文本（顶部）
        if text:
            # 文本背景
            text_str = f"识别结果: {text}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(text_str, font, font_scale, thickness)
            
            # 绘制背景矩形
            cv2.rectangle(
                result,
                (10, 10),
                (20 + text_w, 20 + text_h + baseline),
                self.COLOR_BLACK,
                -1
            )
            # 绘制文本
            cv2.putText(
                result, text_str,
                (15, 15 + text_h),
                font, font_scale, self.COLOR_WHITE, thickness
            )
            
            # 3. 绘制置信度
            conf_str = f"置信度: {confidence:.2%}"
            cv2.putText(
                result, conf_str,
                (15, 45 + text_h),
                font, 0.7, self.COLOR_YELLOW, 2
            )
        
        # 4. 绘制FPS（左下角）
        fps_str = f"FPS: {fps:.1f}"
        cv2.putText(
            result, fps_str,
            (10, h - 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_CYAN, 2
        )
        
        # 5. 绘制设备信息
        device_name = self.device_manager.get_device_name()
        if len(device_name) > 20:
            device_name = device_name[:20] + "..."
        device_str = f"设备: {device_name}"
        cv2.putText(
            result, device_str,
            (10, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_WHITE, 1
        )
        
        # 6. 绘制缓冲区进度
        buffer_str = f"缓冲区: {buffer_progress:.0%}"
        cv2.putText(
            result, buffer_str,
            (10, h - 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_WHITE, 1
        )
        
        # 7. 绘制操作提示（右下角）
        hints = [
            "空格: 暂停/继续",
            "ESC/q: 退出"
        ]
        for i, hint in enumerate(hints):
            (hint_w, hint_h), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(
                result, hint,
                (w - hint_w - 10, h - 60 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1
            )
        
        # 8. 暂停提示
        if is_paused:
            pause_str = "已暂停"
            (pause_w, pause_h), _ = cv2.getTextSize(pause_str, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
            # 居中显示
            cx = (w - pause_w) // 2
            cy = (h + pause_h) // 2
            # 绘制背景
            cv2.rectangle(
                result,
                (cx - 10, cy - pause_h - 10),
                (cx + pause_w + 10, cy + 10),
                self.COLOR_BLACK,
                -1
            )
            cv2.putText(
                result, pause_str,
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.COLOR_RED, 3
            )
        
        return result
    
    def _handle_key(self, key: int) -> bool:
        """
        处理按键事件
        
        Args:
            key: 按键码
            
        Returns:
            bool: 是否继续运行
        """
        if key == -1:  # 无按键
            return True
        
        key = key & 0xFF
        
        if key == 32:  # 空格键
            self.is_paused = self.detection_controller.toggle_pause()
            status = "暂停" if self.is_paused else "继续"
            print(f"检测已{status}")
        
        elif key == 27 or key == ord('q'):  # ESC或q键
            return False
        
        return True
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        处理单帧
        
        Args:
            frame: 输入帧
            
        Returns:
            np.ndarray: 处理后的帧
        """
        # 更新FPS
        fps = self.fps_counter.tick()
        
        # 执行检测
        text, confidence, lip_region = self.detection_controller.process_frame(frame)
        
        # 获取缓冲区状态
        buffer_status = self.detection_controller.get_buffer_status()
        
        # 叠加信息
        result = self._overlay_info(
            frame=frame,
            text=text,
            confidence=confidence,
            fps=fps,
            lip_region=lip_region,
            is_paused=self.is_paused,
            buffer_progress=buffer_status['progress']
        )
        
        return result
    
    def _cleanup(self):
        """清理所有资源"""
        print("\n正在清理资源...")
        
        # 关闭摄像头
        if self.camera:
            self.camera.close()
        
        # 销毁窗口
        cv2.destroyAllWindows()
        
        # 清理GPU缓存
        if self.device_manager:
            self.device_manager.clear_cache()
        
        # 打印统计信息
        if self.detection_controller:
            stats = self.detection_controller.get_statistics()
            print("\n统计信息:")
            print(f"  处理帧数: {stats['frame_count']}")
            print(f"  推理次数: {stats['total_detections']}")
            print(f"  成功识别: {stats['successful_detections']}")
            if stats['total_detections'] > 0:
                print(f"  成功率: {stats['success_rate']:.2%}")
        
        print("✓ 资源清理完成")
    
    def run(self):
        """
        运行GUI主循环
        """
        print("\n" + "=" * 50)
        print("启动实时唇语识别")
        print("=" * 50)
        print("操作说明:")
        print("  - 空格键: 暂停/继续检测")
        print("  - ESC/q: 退出程序")
        print("=" * 50 + "\n")
        
        try:
            # 打开摄像头
            self.camera.open()
            
            # 创建窗口
            cv2.namedWindow(self.WINDOW_TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.WINDOW_TITLE, 800, 600)
            
            # 启动FPS计数器
            self.fps_counter.start()
            
            self.is_running = True
            
            # 主循环
            while self.is_running:
                # 读取帧
                frame = self.camera.read()
                
                if frame is None:
                    print("警告: 无法读取帧")
                    continue
                
                # 处理帧
                result_frame = self._process_frame(frame)
                
                # 显示帧
                cv2.imshow(self.WINDOW_TITLE, result_frame)
                
                # 处理按键
                key = cv2.waitKey(1)
                if not self._handle_key(key):
                    break
        
        except CameraError as e:
            print(f"\n错误: {e}")
        
        except KeyboardInterrupt:
            print("\n用户中断")
        
        except Exception as e:
            print(f"\n运行错误: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.is_running = False
            self._cleanup()


def main():
    """主函数（用于测试）"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LipNet 实时唇语识别')
    parser.add_argument('--model', type=str, 
                       default='pretrain/LipNet_unseen_loss_0.44562849402427673_wer_0.1332580699113564_cer_0.06796452465503355.pt',
                       help='模型权重路径')
    parser.add_argument('--camera-id', type=int, default=0, help='摄像头ID')
    parser.add_argument('--device', choices=['gpu', 'cpu'], default='gpu', help='计算设备')
    parser.add_argument('--predictor', type=str, 
                       default='shape_predictor_68_face_landmarks.dat',
                       help='关键点预测器路径')
    
    args = parser.parse_args()
    
    # 创建GUI实例
    gui = RealtimeLipNetGUI(
        model_path=args.model,
        camera_id=args.camera_id,
        use_gpu=(args.device == 'gpu'),
        predictor_path=args.predictor
    )
    
    # 运行
    gui.run()


if __name__ == "__main__":
    main()
