"""
测试脚本 - 验证所有模块功能
"""
import sys
from pathlib import Path


def test_device_manager():
    """测试设备管理模块"""
    print("\n" + "="*60)
    print("测试 1: 设备管理模块")
    print("="*60)
    
    try:
        from device_manager import DeviceManager
        
        # 列出GPU
        DeviceManager.list_all_gpus()
        
        # 测试GPU模式
        print("\n创建GPU设备管理器:")
        dm_gpu = DeviceManager(use_gpu=True)
        dm_gpu.print_device_info()
        
        # 测试CPU模式
        print("\n创建CPU设备管理器:")
        dm_cpu = DeviceManager(use_gpu=False)
        dm_cpu.print_device_info()
        
        print("✓ 设备管理模块测试通过")
        return True
    
    except Exception as e:
        print(f"✗ 设备管理模块测试失败: {e}")
        return False


def test_video_processor():
    """测试视频处理模块"""
    print("\n" + "="*60)
    print("测试 2: 视频处理模块")
    print("="*60)
    
    try:
        from video_processor import LipDetector, MP4VideoProcessor, CameraVideoProcessor
        
        # 检查关键点预测器
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        if not Path(predictor_path).exists():
            print(f"⚠ 跳过测试: 未找到关键点预测器文件 {predictor_path}")
            print("  请从 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 下载")
            return True
        
        # 测试唇部检测器初始化
        print("\n初始化唇部检测器:")
        lip_detector = LipDetector(predictor_path)
        print("✓ 唇部检测器初始化成功")
        
        print("✓ 视频处理模块测试通过")
        return True
    
    except Exception as e:
        print(f"✗ 视频处理模块测试失败: {e}")
        return False


def test_model_wrapper():
    """测试模型加载模块"""
    print("\n" + "="*60)
    print("测试 3: 模型加载模块")
    print("="*60)
    
    try:
        from model_wrapper import LipNetModel
        from device_manager import DeviceManager
        
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
            print("⚠ 跳过测试: 未找到模型文件")
            print("  请确保pretrain目录下有预训练权重文件")
            return True
        
        # 测试模型加载
        print(f"\n加载模型: {model_path}")
        device_manager = DeviceManager(use_gpu=True)
        model = LipNetModel(model_path, device_manager)
        print("✓ 模型加载成功")
        
        print("✓ 模型加载模块测试通过")
        return True
    
    except Exception as e:
        print(f"✗ 模型加载模块测试失败: {e}")
        return False


def test_lipnet_app():
    """测试应用模块"""
    print("\n" + "="*60)
    print("测试 4: 应用服务模块")
    print("="*60)
    
    try:
        from lipnet_app import LipNetApp
        
        # 检查依赖文件
        model_paths = [
            'pretrain/LipNet_unseen_loss_0.44562849402427673_wer_0.1332580699113564_cer_0.06796452465503355.pt',
            'pretrain/LipNet_overlap_loss_0.07664558291435242_wer_0.04644484056248762_cer_0.019676921477851092.pt'
        ]
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        
        model_path = None
        for path in model_paths:
            if Path(path).exists():
                model_path = path
                break
        
        if model_path is None or not Path(predictor_path).exists():
            print("⚠ 跳过测试: 缺少必要的依赖文件")
            return True
        
        # 测试应用初始化
        print(f"\n初始化LipNet应用:")
        app = LipNetApp(model_path=model_path, use_gpu=True)
        print("✓ 应用初始化成功")
        
        print("✓ 应用服务模块测试通过")
        return True
    
    except Exception as e:
        print(f"✗ 应用服务模块测试失败: {e}")
        return False


def test_cli():
    """测试命令行接口"""
    print("\n" + "="*60)
    print("测试 5: 命令行接口")
    print("="*60)
    
    try:
        from run_lipnet import create_parser
        
        parser = create_parser()
        
        # 测试参数解析
        test_args = [
            ['--mode', 'mp4', '--video', 'test.mp4', '--device', 'gpu'],
            ['--mode', 'camera', '--device', 'cpu'],
            ['--list-gpus']
        ]
        
        for args in test_args:
            print(f"\n测试参数: {' '.join(args)}")
            parsed = parser.parse_args(args)
            print(f"  mode: {parsed.mode}")
            print(f"  device: {parsed.device}")
        
        print("\n✓ 命令行接口测试通过")
        return True
    
    except Exception as e:
        print(f"✗ 命令行接口测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("LipNet-PyTorch 功能测试")
    print("="*60)
    
    results = {
        '设备管理模块': test_device_manager(),
        '视频处理模块': test_video_processor(),
        '模型加载模块': test_model_wrapper(),
        '应用服务模块': test_lipnet_app(),
        '命令行接口': test_cli()
    }
    
    # 打印测试结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")
    
    # 统计
    total = len(results)
    passed = sum(results.values())
    
    print("\n" + "="*60)
    print(f"总计: {passed}/{total} 测试通过")
    print("="*60)
    
    # 返回状态码
    if passed == total:
        print("\n✓ 所有测试通过!")
        return 0
    else:
        print("\n⚠ 部分测试未通过，请检查依赖文件")
        return 1


if __name__ == "__main__":
    sys.exit(main())
