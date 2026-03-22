"""
命令行接口模块（云服务器版本）
"""
import argparse
import sys
from pathlib import Path

from device_manager import DeviceManager
from lipnet_app import LipNetApp


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='LipNet - 唇语识别系统（云服务器版本）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理MP4文件，结果保存到新视频
  python run_lipnet.py --mode mp4 --video input.mp4
  
  # 指定输出路径
  python run_lipnet.py --mode mp4 --video input.mp4 --output result.mp4
  
  # 使用CPU处理
  python run_lipnet.py --mode mp4 --video input.mp4 --device cpu
  
  # 摄像头录制30秒
  python run_lipnet.py --mode camera --duration 30
  
  # 摄像头录制60秒并保存
  python run_lipnet.py --mode camera --duration 60 --output camera.mp4
  
  # 列出所有GPU设备
  python run_lipnet.py --list-gpus
        """
    )
    
    # 模式选择
    parser.add_argument(
        '--mode',
        type=str,
        choices=['mp4', 'camera'],
        default='mp4',
        help='运行模式: mp4(处理视频文件) 或 camera(摄像头录制)'
    )
    
    # 视频文件路径
    parser.add_argument(
        '--video',
        type=str,
        help='MP4视频文件路径 (mode=mp4时必需)'
    )
    
    # 输出路径
    parser.add_argument(
        '--output',
        type=str,
        help='输出视频文件路径 (默认自动生成)'
    )
    
    # 设备选择
    parser.add_argument(
        '--device',
        type=str,
        choices=['gpu', 'cpu'],
        default='gpu',
        help='计算设备: gpu(默认) 或 cpu'
    )
    
    # GPU设备ID
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='GPU设备ID (默认: 0)'
    )
    
    # 模型路径
    parser.add_argument(
        '--model',
        type=str,
        default='pretrain/LipNet_unseen_loss_0.44562849402427673_wer_0.1332580699113564_cer_0.06796452465503355.pt',
        help='模型权重文件路径'
    )
    
    # 关键点预测器路径
    parser.add_argument(
        '--predictor',
        type=str,
        default='shape_predictor_68_face_landmarks.dat',
        help='dlib关键点预测器文件路径'
    )
    
    # 帧数
    parser.add_argument(
        '--num-frames',
        type=int,
        default=75,
        help='每次推理使用的帧数 (默认: 75)'
    )
    
    # 摄像头ID
    parser.add_argument(
        '--camera-id',
        type=int,
        default=0,
        help='摄像头设备ID (默认: 0)'
    )
    
    # 录制时长
    parser.add_argument(
        '--duration',
        type=int,
        default=30,
        help='摄像头录制时长(秒) (默认: 30)'
    )
    
    # 列出GPU
    parser.add_argument(
        '--list-gpus',
        action='store_true',
        help='列出所有可用的GPU设备'
    )
    
    return parser


def check_dependencies():
    """检查依赖文件"""
    print("\n检查依赖文件...")
    
    # 检查模型文件
    model_paths = [
        'pretrain/LipNet_unseen_loss_0.44562849402427673_wer_0.1332580699113564_cer_0.06796452465503355.pt',
        'pretrain/LipNet_overlap_loss_0.07664558291435242_wer_0.04644484056248762_cer_0.019676921477851092.pt'
    ]
    
    model_found = False
    for path in model_paths:
        if Path(path).exists():
            print(f"  ✓ 模型文件: {path}")
            model_found = True
            break
    
    if not model_found:
        print("  ✗ 未找到模型文件")
        print("    请确保pretrain目录下有预训练权重文件")
    
    # 检查关键点预测器
    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    if Path(predictor_path).exists():
        print(f"  ✓ 关键点预测器: {predictor_path}")
    else:
        print("  ✗ 未找到关键点预测器文件")
        print("    请从以下地址下载:")
        print("    - http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    
    print()


def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 列出GPU设备
    if args.list_gpus:
        DeviceManager.list_all_gpus()
        return
    
    # 检查依赖
    check_dependencies()
    
    # 检查模型文件
    if not Path(args.model).exists():
        print(f"✗ 模型文件不存在: {args.model}")
        print("请使用 --model 参数指定正确的模型路径")
        sys.exit(1)
    
    # 检查关键点预测器
    if not Path(args.predictor).exists():
        print(f"✗ 关键点预测器文件不存在: {args.predictor}")
        print("请下载 shape_predictor_68_face_landmarks.dat 文件")
        sys.exit(1)
    
    # MP4模式检查视频文件
    if args.mode == 'mp4':
        if not args.video:
            print("✗ MP4模式需要指定视频文件路径 (--video)")
            sys.exit(1)
        if not Path(args.video).exists():
            print(f"✗ 视频文件不存在: {args.video}")
            sys.exit(1)
    
    # 创建应用
    use_gpu = (args.device == 'gpu')
    
    print("\n" + "="*60)
    print("LipNet 唇语识别系统（云服务器版本）")
    print("="*60)
    print(f"运行模式: {args.mode.upper()}")
    print(f"计算设备: {args.device.upper()}")
    print(f"模型文件: {args.model}")
    if args.mode == 'mp4':
        print(f"输入视频: {args.video}")
        if args.output:
            print(f"输出视频: {args.output}")
    else:
        print(f"录制时长: {args.duration}秒")
    print("="*60 + "\n")
    
    try:
        app = LipNetApp(
            model_path=args.model,
            use_gpu=use_gpu,
            device_id=args.gpu_id,
            predictor_path=args.predictor
        )
        
        # 根据模式运行
        if args.mode == 'mp4':
            results = app.process_mp4(
                video_path=args.video,
                output_path=args.output,
                num_frames=args.num_frames,
                save_json=True
            )
            
            if results['success']:
                print("\n" + "="*60)
                print("处理完成!")
                print("="*60)
                print(f"输出视频: {results['output_path']}")
                print(f"总帧数: {results['total_frames']}")
                print(f"识别次数: {len(results['predictions'])}")
                print("="*60)
        
        elif args.mode == 'camera':
            results = app.process_camera(
                camera_id=args.camera_id,
                num_frames=args.num_frames,
                output_path=args.output,
                duration=args.duration
            )
            
            if results['success']:
                print("\n" + "="*60)
                print("录制完成!")
                print("="*60)
                print(f"输出视频: {results['output_path']}")
                print(f"总帧数: {results['total_frames']}")
                print(f"识别次数: {len(results['predictions'])}")
                print("="*60)
    
    except KeyboardInterrupt:
        print("\n\n用户中断程序")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n✗ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
