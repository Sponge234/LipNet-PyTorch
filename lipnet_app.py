"""
应用服务模块 - 整合所有功能的主应用类（云服务器版本）
"""
import cv2
import numpy as np
import time
from typing import Optional, Tuple
from pathlib import Path

from device_manager import DeviceManager
from video_processor import VideoProcessor, MP4VideoProcessor, CameraVideoProcessor, LipDetector
from model_wrapper import LipNetModel


class LipNetApp:
    """
    LipNet应用主类，整合视频处理、模型推理和结果保存
    专为云服务器设计，无GUI依赖
    """
    
    def __init__(
        self,
        model_path: str,
        use_gpu: bool = True,
        device_id: int = 0,
        predictor_path: str = "shape_predictor_68_face_landmarks.dat"
    ):
        """
        初始化LipNet应用
        
        Args:
            model_path: 模型权重文件路径
            use_gpu: 是否使用GPU
            device_id: GPU设备ID
            predictor_path: dlib关键点预测器文件路径
        """
        # 初始化设备管理器
        self.device_manager = DeviceManager(use_gpu=use_gpu, device_id=device_id)
        
        # 初始化唇部检测器
        self.lip_detector = LipDetector(predictor_path)
        
        # 加载模型
        self.model = LipNetModel(model_path, self.device_manager)
        
        # 视频处理器
        self.video_processor = None
        
        print("✓ LipNet应用初始化完成")
    
    def process_mp4(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        num_frames: int = 75,
        save_json: bool = True
    ) -> dict:
        """
        处理MP4视频文件，将识别结果叠加到视频并保存
        
        Args:
            video_path: MP4视频文件路径
            output_path: 输出视频文件路径（如果为None，自动生成）
            num_frames: 提取的帧数
            save_json: 是否保存JSON格式的结果
            
        Returns:
            结果字典
        """
        print(f"\n开始处理MP4文件: {video_path}")
        
        # 生成输出路径
        if output_path is None:
            video_path_obj = Path(video_path)
            output_path = str(video_path_obj.parent / f"{video_path_obj.stem}_result{video_path_obj.suffix}")
        
        # 创建视频处理器
        self.video_processor = MP4VideoProcessor(video_path, self.lip_detector)
        
        results = {
            'video_path': video_path,
            'output_path': output_path,
            'predictions': [],
            'total_frames': 0,
            'success': False
        }
        
        try:
            # 打开输入视频
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频文件: {video_path}")
            
            # 获取视频属性
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"视频信息: {width}x{height}, {fps}FPS, 总帧数: {total_frames}")
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # 处理视频
            frame_count = 0
            lip_frames_buffer = []
            current_text = ""
            current_confidence = 0.0
            
            print("正在处理视频帧...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 检测唇部
                lip_frame = self.lip_detector.extract_lip(frame)
                
                if lip_frame is not None:
                    lip_frames_buffer.append(lip_frame)
                
                # 当缓冲区满时进行推理
                if len(lip_frames_buffer) >= num_frames:
                    lip_frames = np.array(lip_frames_buffer[-num_frames:])
                    current_text, output = self.model.infer(lip_frames)
                    current_confidence = self.model.get_confidence(output)
                    
                    # 清空部分缓冲区
                    lip_frames_buffer = lip_frames_buffer[-num_frames//2:]
                    
                    # 记录结果
                    results['predictions'].append({
                        'text': current_text,
                        'confidence': float(current_confidence),
                        'frame': frame_count
                    })
                
                # 在帧上叠加识别结果
                display_frame = frame.copy()
                
                # 绘制唇部区域框
                lip_region = self.lip_detector.detect_lip_region(frame)
                if lip_region:
                    x, y, w, h = lip_region
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # 叠加识别文本
                if current_text:
                    # 文本背景
                    text_size = cv2.getTextSize(current_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
                    cv2.rectangle(display_frame, (10, 10), (20 + text_size[0], 50), (0, 0, 0), -1)
                    
                    # 绘制文本
                    cv2.putText(display_frame, f"Text: {current_text}", (15, 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                    
                    # 绘制置信度
                    cv2.putText(display_frame, f"Conf: {current_confidence:.3f}", (15, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # 写入输出视频
                out.write(display_frame)
                
                frame_count += 1
                
                # 显示进度
                if frame_count % 100 == 0:
                    progress = frame_count / total_frames * 100
                    print(f"  进度: {progress:.1f}% ({frame_count}/{total_frames})")
            
            # 释放资源
            cap.release()
            out.release()
            
            results['total_frames'] = frame_count
            results['success'] = True
            
            print(f"\n✓ 视频处理完成!")
            print(f"  输出文件: {output_path}")
            print(f"  总帧数: {frame_count}")
            print(f"  识别次数: {len(results['predictions'])}")
            
            # 保存JSON结果
            if save_json:
                json_path = str(Path(output_path).with_suffix('.json'))
                self._save_json(json_path, results)
                print(f"  结果文件: {json_path}")
            
            # 打印最终识别结果
            if results['predictions']:
                print(f"\n最终识别结果:")
                for i, pred in enumerate(results['predictions'][-5:], 1):  # 显示最后5个结果
                    print(f"  {i}. '{pred['text']}' (置信度: {pred['confidence']:.3f})")
        
        except Exception as e:
            print(f"✗ 处理失败: {e}")
            results['error'] = str(e)
            raise
        
        return results
    
    def process_camera(
        self,
        camera_id: int = 0,
        num_frames: int = 75,
        output_path: Optional[str] = None,
        duration: int = 30
    ) -> dict:
        """
        处理摄像头，保存带识别结果的视频
        
        Args:
            camera_id: 摄像头设备ID
            num_frames: 每次推理使用的帧数
            output_path: 输出视频文件路径
            duration: 录制时长（秒）
            
        Returns:
            结果字典
        """
        print(f"\n开始摄像头录制 (摄像头ID: {camera_id}, 时长: {duration}秒)")
        
        # 生成输出路径
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = f"camera_result_{timestamp}.mp4"
        
        # 创建摄像头处理器
        self.video_processor = CameraVideoProcessor(camera_id, self.lip_detector)
        
        results = {
            'camera_id': camera_id,
            'output_path': output_path,
            'predictions': [],
            'total_frames': 0,
            'success': False
        }
        
        try:
            # 打开摄像头
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开摄像头 {camera_id}")
            
            # 设置摄像头分辨率
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = 30
            
            print(f"摄像头信息: {width}x{height}, {fps}FPS")
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # 处理摄像头
            frame_count = 0
            lip_frames_buffer = []
            current_text = ""
            current_confidence = 0.0
            start_time = time.time()
            
            print("正在录制...")
            
            while True:
                # 检查时长
                if time.time() - start_time > duration:
                    print(f"\n达到录制时长限制 ({duration}秒)")
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 检测唇部
                lip_frame = self.lip_detector.extract_lip(frame)
                
                if lip_frame is not None:
                    lip_frames_buffer.append(lip_frame)
                
                # 当缓冲区满时进行推理
                if len(lip_frames_buffer) >= num_frames:
                    lip_frames = np.array(lip_frames_buffer[-num_frames:])
                    current_text, output = self.model.infer(lip_frames)
                    current_confidence = self.model.get_confidence(output)
                    
                    lip_frames_buffer = lip_frames_buffer[-num_frames//2:]
                    
                    results['predictions'].append({
                        'text': current_text,
                        'confidence': float(current_confidence),
                        'frame': frame_count
                    })
                
                # 在帧上叠加识别结果
                display_frame = frame.copy()
                
                # 绘制唇部区域框
                lip_region = self.lip_detector.detect_lip_region(frame)
                if lip_region:
                    x, y, w, h = lip_region
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # 叠加识别文本
                if current_text:
                    cv2.rectangle(display_frame, (10, 10), (600, 50), (0, 0, 0), -1)
                    cv2.putText(display_frame, f"Text: {current_text}", (15, 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Conf: {current_confidence:.3f}", (15, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # 显示录制时间
                elapsed = time.time() - start_time
                cv2.putText(display_frame, f"Time: {elapsed:.1f}s/{duration}s", (15, height - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 写入输出视频
                out.write(display_frame)
                
                frame_count += 1
                
                # 显示进度
                if frame_count % 100 == 0:
                    print(f"  已录制: {elapsed:.1f}秒, 帧数: {frame_count}")
            
            # 释放资源
            cap.release()
            out.release()
            
            results['total_frames'] = frame_count
            results['success'] = True
            
            print(f"\n✓ 摄像头录制完成!")
            print(f"  输出文件: {output_path}")
            print(f"  总帧数: {frame_count}")
            print(f"  识别次数: {len(results['predictions'])}")
            
            # 保存JSON结果
            json_path = str(Path(output_path).with_suffix('.json'))
            self._save_json(json_path, results)
            print(f"  结果文件: {json_path}")
        
        except KeyboardInterrupt:
            print("\n用户中断录制")
            results['interrupted'] = True
        
        except Exception as e:
            print(f"✗ 处理失败: {e}")
            results['error'] = str(e)
            raise
        
        return results
    
    def _save_json(self, json_path: str, results: dict):
        """保存结果到JSON文件"""
        import json
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # 测试应用
    print("测试LipNet应用（云服务器版本）")
    
    # 检查模型文件
    model_paths = [
        'pretrain/LipNet_unseen_loss_0.44562849402427673_wer_0.1332580699113564_cer_0.06796452465503355.pt',
        'pretrain/LipNet_overlap_loss_0.07664558291435242_wer_0.04644484056248762_cer_0.019676921477851092.pt'
    ]
    
    model_path = None
    for path in model_paths:
        if Path(path).exists():
            model_path = path
            break
    
    if model_path is None:
        print("未找到模型文件")
    else:
        # 创建应用
        app = LipNetApp(model_path=model_path, use_gpu=True)
        
        # 测试MP4处理（如果有测试视频）
        test_video = "test.mp4"
        if Path(test_video).exists():
            print(f"\n测试MP4处理: {test_video}")
            results = app.process_mp4(test_video)
            print(f"处理结果: {results}")
