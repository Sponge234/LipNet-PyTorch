# LipNet-PyTorch 使用指南

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载必要文件

#### 预训练模型
从以下地址下载预训练模型权重：
- GitHub: https://github.com/lordmartino/lipnet_weights
- 将模型文件放置在 `models/` 目录下

#### Dlib关键点预测器
```bash
# 下载并解压
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```

### 3. 运行程序

#### 处理MP4视频文件

使用GPU（默认）：
```bash
python run_lipnet.py --mode mp4 --video your_video.mp4
```

使用CPU：
```bash
python run_lipnet.py --mode mp4 --video your_video.mp4 --device cpu
```

指定GPU设备：
```bash
python run_lipnet.py --mode mp4 --video your_video.mp4 --device gpu --gpu-id 0
```

保存结果到文件：
```bash
python run_lipnet.py --mode mp4 --video your_video.mp4 --output result.json
```

#### 摄像头实时检测

启动实时检测：
```bash
python run_lipnet.py --mode camera
```

使用CPU进行实时检测：
```bash
python run_lipnet.py --mode camera --device cpu
```

指定摄像头设备：
```bash
python run_lipnet.py --mode camera --camera-id 0
```

#### 其他功能

列出所有可用GPU：
```bash
python run_lipnet.py --list-gpus
```

运行测试：
```bash
python test_modules.py
```

## 命令行参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--mode` | str | camera | 运行模式：mp4 或 camera |
| `--video` | str | - | MP4视频文件路径（mp4模式必需） |
| `--device` | str | gpu | 计算设备：gpu 或 cpu |
| `--gpu-id` | int | 0 | GPU设备ID |
| `--model` | str | models/LipNet_405.pkl | 模型权重文件路径 |
| `--predictor` | str | shape_predictor_68_face_landmarks.dat | dlib预测器文件路径 |
| `--num-frames` | int | 75 | 每次推理使用的帧数 |
| `--camera-id` | int | 0 | 摄像头设备ID |
| `--output` | str | - | 结果输出文件路径 |
| `--no-display` | flag | False | 不显示视频窗口 |
| `--list-gpus` | flag | - | 列出所有GPU设备 |

## 键盘控制（摄像头模式）

- `q` - 退出程序
- `s` - 保存当前帧截图

## 项目结构

```
LipNet-PyTorch/
├── device_manager.py      # 设备管理模块
├── video_processor.py     # 视频处理模块
├── model_wrapper.py       # 模型加载和推理模块
├── lipnet_app.py          # 应用服务模块
├── run_lipnet.py          # 命令行接口
├── test_modules.py        # 测试脚本
├── requirements.txt       # 依赖列表
├── model.py               # LipNet模型定义
├── dataset.py             # 数据集处理
├── main.py                # 训练入口
└── demo.py                # 原始演示程序
```

## 模块说明

### 1. DeviceManager (device_manager.py)
负责GPU/CPU设备的选择和管理：
- 自动检测GPU可用性
- 设备信息查询
- 自动降级处理（GPU不可用时自动切换到CPU）

### 2. VideoProcessor (video_processor.py)
负责视频读取和唇部检测：
- MP4VideoProcessor: 处理MP4视频文件
- CameraVideoProcessor: 处理摄像头实时视频
- LipDetector: 使用dlib进行唇部检测和提取

### 3. LipNetModel (model_wrapper.py)
负责模型加载和推理：
- 模型权重加载
- 数据预处理
- CTC解码
- 置信度计算

### 4. LipNetApp (lipnet_app.py)
主应用服务，整合所有功能：
- MP4文件处理流程
- 摄像头实时检测流程
- 结果可视化
- 结果保存

### 5. CLI (run_lipnet.py)
命令行接口：
- 参数解析
- 依赖检查
- 模式选择

## 性能优化建议

1. **GPU选择**：默认使用GPU，性能最佳
2. **帧数调整**：`--num-frames` 参数影响识别准确率和速度
   - 较大值（如100）：更准确，但速度较慢
   - 较小值（如50）：速度较快，但可能降低准确率
3. **摄像头分辨率**：默认640x480，可在代码中调整

## 常见问题

### Q: 提示找不到模型文件
A: 请从 https://github.com/lordmartino/lipnet_weights 下载预训练模型，放置在 `models/` 目录

### Q: 提示找不到关键点预测器
A: 请下载 shape_predictor_68_face_landmarks.dat 文件

### Q: GPU不可用
A: 程序会自动降级到CPU模式，无需手动干预

### Q: 摄像头打不开
A: 检查摄像头是否被其他程序占用，或尝试不同的 `--camera-id`

## 示例输出

```
============================================================
LipNet 唇语识别系统
============================================================
运行模式: MP4
计算设备: GPU
模型文件: models/LipNet_405.pkl
============================================================

✓ 使用GPU设备: NVIDIA GeForce RTX 4090 D
✓ 唇部检测器初始化成功
✓ 成功加载模型: models/LipNet_405.pkl
✓ LipNet应用初始化完成

开始处理MP4文件: test.mp4
✓ 成功打开视频文件: test.mp4
  分辨率: 1920x1080
  帧率: 30.00 FPS
  总帧数: 900

正在提取唇部帧...
✓ 成功提取 75 帧唇部图像

正在进行模型推理...

识别结果: 'place red at d green soon'
置信度: 0.8234

============================================================
处理完成!
============================================================
识别结果: 'place red at d green soon'
置信度: 0.8234
============================================================
```
