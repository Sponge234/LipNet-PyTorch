#!/usr/bin/env python
"""
LipNet 实时唇语识别启动脚本

功能:
- 打开摄像头实时检测唇语
- 将识别结果显示在屏幕上
- 支持GPU/CPU切换
- 支持多种参数配置

使用方法:
    python run_realtime.py                          # 使用默认GPU
    python run_realtime.py --device cpu             # 使用CPU
    python run_realtime.py --camera-id 1            # 使用第二个摄像头
    python run_realtime.py --help                   # 显示帮助
"""
import argparse
import sys
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    """
    创建命令行参数解析器
    
    Returns:
        argparse.ArgumentParser: 参数解析器
    """
    parser = argparse.ArgumentParser(
        description='LipNet 实时唇语识别 - 打开摄像头实时检测唇语并显示结果',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                          # 使用默认GPU运行
  %(prog)s --device cpu             # 使用CPU运行
  %(prog)s --camera-id 1            # 使用第二个摄像头
  %(prog)s --num-frames 50          # 使用50帧进行推理
  %(prog)s --list-gpus              # 列出所有可用GPU

操作说明:
  运行后:
    - 空格键: 暂停/继续检测
    - ESC/q: 退出程序
        """
    )
    
    # 基本参数
    parser.add_argument(
        '--camera-id',
        type=int,
        default=0,
        help='摄像头设备ID (默认: 0)'
    )
    
    # 设备参数
    parser.add_argument(
        '--device',
        choices=['gpu', 'cpu'],
        default='gpu',
        help='计算设备: gpu 或 cpu (默认: gpu)'
    )
    
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='GPU设备ID (默认: 0)'
    )
    
    # 模型参数
    parser.add_argument(
        '--model',
        type=str,
        default='pretrain/LipNet_unseen_loss_0.44562849402427673_wer_0.1332580699113564_cer_0.06796452465503355.pt',
        help='模型权重文件路径'
    )
    
    parser.add_argument(
        '--predictor',
        type=str,
        default='shape_predictor_68_face_landmarks.dat',
        help='dlib关键点预测器文件路径'
    )
    
    parser.add_argument(
        '--num-frames',
        type=int,
        default=75,
        help='推理所需的帧数 (默认: 75)'
    )
    
    # 工具参数
    parser.add_argument(
        '--list-gpus',
        action='store_true',
        help='列出所有可用的GPU设备'
    )
    
    return parser


def check_dependencies(args) -> bool:
    """
    检查依赖文件是否存在
    
    Args:
        args: 命令行参数
        
    Returns:
        bool: 是否所有依赖都存在
    """
    print("\n检查依赖文件...")
    
    errors = []
    
    # 检查模型文件
    model_path = Path(args.model)
    if not model_path.exists():
        errors.append(f"模型文件不存在: {args.model}")
    else:
        print(f"  ✓ 模型文件: {args.model}")
    
    # 检查关键点预测器文件
    predictor_path = Path(args.predictor)
    if not predictor_path.exists():
        errors.append(
            f"关键点预测器文件不存在: {args.predictor}\n"
            "    请从 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 下载"
        )
    else:
        print(f"  ✓ 关键点预测器: {args.predictor}")
    
    if errors:
        print("\n错误:")
        for error in errors:
            print(f"  ✗ {error}")
        return False
    
    print("  ✓ 所有依赖文件检查通过\n")
    return True


def check_gpu_availability(args) -> bool:
    """
    检查GPU可用性
    
    Args:
        args: 命令行参数
        
    Returns:
        bool: 是否使用GPU
    """
    import torch
    
    if args.device == 'cpu':
        print("使用CPU模式")
        return False
    
    if not torch.cuda.is_available():
        print("⚠ GPU不可用，自动降级到CPU模式")
        print("  提示: 如果需要使用GPU，请确保安装了CUDA版本的PyTorch")
        return False
    
    gpu_count = torch.cuda.device_count()
    if args.gpu_id >= gpu_count:
        print(f"⚠ GPU {args.gpu_id} 不存在，自动使用GPU 0")
        args.gpu_id = 0
    
    gpu_name = torch.cuda.get_device_name(args.gpu_id)
    print(f"✓ 使用GPU {args.gpu_id}: {gpu_name}")
    return True


def list_gpus():
    """列出所有可用的GPU设备"""
    import torch
    
    print("\n" + "=" * 50)
    print("GPU 设备列表")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("未检测到可用的GPU设备")
        print("\n可能的原因:")
        print("  1. 系统没有NVIDIA GPU")
        print("  2. 未安装CUDA驱动")
        print("  3. PyTorch未安装CUDA版本")
        return
    
    gpu_count = torch.cuda.device_count()
    print(f"\n检测到 {gpu_count} 个GPU设备:\n")
    
    for i in range(gpu_count):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {name}")
        print(f"  显存: {props.total_memory / 1024**3:.2f} GB")
        print(f"  计算能力: {props.major}.{props.minor}")
        print()


def main():
    """主函数"""
    # 解析参数
    parser = create_parser()
    args = parser.parse_args()
    
    # 列出GPU
    if args.list_gpus:
        list_gpus()
        return
    
    # 打印欢迎信息
    print("\n" + "=" * 50)
    print("LipNet 实时唇语识别系统")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies(args):
        sys.exit(1)
    
    # 检查GPU
    use_gpu = check_gpu_availability(args)
    
    # 设置GPU ID
    if use_gpu:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    # 导入并运行GUI
    try:
        from realtime_gui import RealtimeLipNetGUI
        
        # 创建GUI实例
        gui = RealtimeLipNetGUI(
            model_path=args.model,
            camera_id=args.camera_id,
            use_gpu=use_gpu,
            predictor_path=args.predictor,
            num_frames=args.num_frames
        )
        
        # 运行GUI
        gui.run()
        
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        sys.exit(1)
    
    except ImportError as e:
        print(f"\n导入错误: {e}")
        print("请确保已安装所有依赖: pip install -r requirements.txt")
        sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n用户中断")
    
    except Exception as e:
        print(f"\n运行错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
