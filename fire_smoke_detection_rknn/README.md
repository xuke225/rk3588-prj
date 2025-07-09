# 火灾烟雾检测算法

## 介绍
基于YOLOv8的火灾烟雾实时检测系统，专为EagleCV项目设计。该算法能够准确识别图像和视频中的火灾和烟雾，为安全监控提供智能预警功能。

## 特性
- 🔥 **火灾检测**: 精确识别明火
- 💨 **烟雾检测**: 有效检测烟雾
- ⚡ **实时处理**: 支持实时视频流分析
- 🚨 **紧急预警**: 自动判断紧急程度
- 🎯 **高精度**: 基于YOLOv8优化的检测模型
- 🔧 **易于集成**: 标准化API接口

## 环境依赖

| 程序 | 版本 | 说明 |
|------|------|------|
| Python | 3.8+ | 推荐3.9或3.10 |
| PyTorch | 2.0+ | 深度学习框架 |
| Ultralytics | 8.0+ | YOLOv8官方库 |
| OpenCV | 4.8+ | 图像处理 |
| FastAPI | 0.104+ | Web API框架 |

### GPU支持
- **NVIDIA GPU**: 安装CUDA版本的PyTorch
- **Apple Silicon**: 支持MPS加速
- **CPU**: 支持纯CPU推理（速度较慢）

## 安装依赖
```bash
pip install -r requirements.txt
```

### 针对不同平台的安装
```bash
# NVIDIA GPU (CUDA 11.8)
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# NVIDIA GPU (CUDA 12.1)  
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# CPU版本
pip install torch==2.1.0+cpu torchvision==0.16.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Apple Silicon (M1/M2)
pip install torch torchvision  # 自动支持MPS
```

## 目录结构
```
fire_smoke_detection/
├── main.py                    # FastAPI服务入口
├── algorithm.py               # 火灾烟雾检测核心算法
├── train_fire_detection.py    # 模型训练脚本
├── requirements.txt           # 依赖配置
├── README.md                 # 说明文档
├── models/                   # 模型文件目录
│   └── fire_best.pt         # 训练好的火灾检测模型
├── datasets/                 # 数据集目录（可选）
│   └── fire-8/              # 火灾数据集
└── runs/                    # 训练结果目录
    └── detect/              # 检测训练结果
```

## 快速开始

### 1. 启动API服务
```bash
python main.py
```
服务将在 `http://localhost:9706` 启动

### 2. 查看API文档
打开浏览器访问: `http://localhost:9706/docs`

### 3. 测试API接口
```bash
curl -X POST "http://localhost:9706/algorithm" \
     -H "Content-Type: application/json" \
     -d '{"image_base64": "YOUR_BASE64_ENCODED_IMAGE"}'
```

## API接口说明

### 主要端点
- `POST /algorithm` - 标准检测接口（兼容EagleCV）
- `POST /detect` - 火灾烟雾检测专用接口
- `GET /` - 健康检查
- `GET /health` - 详细状态检查
- `GET /model/info` - 模型信息

### 请求格式
```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

### 响应格式
```json
{
  "code": 1000,
  "msg": "success",
  "result": {
    "happen": true,
    "happenScore": 0.85,
    "detects": [
      {
        "x1": 100,
        "y1": 100,
        "x2": 300,
        "y2": 250,
        "class_score": 0.85,
        "class_name": "fire",
        "class_name_cn": "火灾",
        "class_id": 0
      }
    ],
    "emergency": {
      "is_emergency": true,
      "emergency_type": "fire",
      "emergency_level": "high",
      "max_confidence": 0.85
    }
  }
}
```

### 检测类别
| ID | 英文名 | 中文名 | 紧急程度 |
|----|--------|--------|----------|
| 0  | fire   | 火灾   | 高       |
| 1  | smoke  | 烟雾   | 中       |

## 模型训练

### 准备数据集
1. 将数据集放置在 `datasets/fire-8/` 目录下
2. 确保数据集结构如下：
```
datasets/fire-8/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
```

### 开始训练
```bash
# 基础训练
python train_fire_detection.py

# 自定义参数训练
python train_fire_detection.py \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --model yolov8s.pt \
    --device 0
```

### 训练参数说明
- `--epochs`: 训练轮数（默认25）
- `--batch`: 批次大小（默认16）
- `--imgsz`: 图像尺寸（默认640）
- `--model`: 基础模型（默认yolov8n.pt）
- `--device`: 训练设备（auto/cpu/mps/0）

## 配置说明

### 算法配置
在 `main.py` 中可以调整以下参数：
```python
algorithm = FireSmokeDetector(
    model_path="models/fire_best.pt",  # 模型路径
    confidence_threshold=0.5,          # 置信度阈值
    nms_threshold=0.4,                # NMS阈值
    input_size=640                    # 输入尺寸
)
```

### 服务配置
- 端口: 9706（在main.py中修改）
- 日志级别: INFO
- 并发: 支持异步处理

## 集成到EagleCV系统

### 1. 修改系统配置
编辑根目录的 `config.json` 文件：
```json
{
  "algorithmApiUrl": "http://127.0.0.1:9706/algorithm"
}
```

### 2. 重启EagleCV主系统
```bash
cd ../..  # 回到项目根目录
./run.sh
```

## 性能优化建议

### 1. 硬件优化
- 使用GPU加速（NVIDIA/Apple Silicon）
- 增加内存提高批处理大小
- 使用SSD存储提高I/O性能

### 2. 模型优化
- 使用更小的模型（yolov8n）提高速度
- 使用量化模型减少内存占用
- 调整输入尺寸平衡精度和速度

### 3. 部署优化
- 启用模型缓存
- 使用批处理处理多张图片
- 配置合适的置信度阈值

## 故障排除

### 常见问题

1. **模型文件不存在**
   ```
   警告: 模型文件 'models/fire_best.pt' 不存在
   将使用预训练的YOLOv8n模型作为演示
   ```
   解决方案: 确保模型文件路径正确，或训练新模型

2. **GPU内存不足**
   ```
   CUDA out of memory
   ```
   解决方案: 减少批次大小或使用CPU模式

3. **依赖安装失败**
   ```
   ERROR: Could not find a version that satisfies the requirement
   ```
   解决方案: 检查Python版本，使用conda环境

4. **端口被占用**
   ```
   Error: [Errno 48] Address already in use
   ```
   解决方案: 修改main.py中的端口号

### 调试模式
启用详细日志：
```bash
export PYTHONPATH=$PYTHONPATH:.
python -u main.py
```

## 版本更新

### v1.0.0 (当前版本)
- 基础火灾烟雾检测功能
- YOLOv8模型集成
- FastAPI服务接口
- 紧急状态判断
- 多设备支持

### 计划功能
- [ ] 视频流实时处理
- [ ] 历史检测记录
- [ ] 自定义预警规则
- [ ] 模型热更新
- [ ] 检测结果可视化

## 许可证
本项目遵循EagleCV项目的许可证协议。

## 技术支持
如遇问题请：
1. 检查日志输出
2. 确认模型文件完整
3. 验证依赖版本
4. 测试API接口响应

---
🔥 **Fire & Smoke Detection v1.0.0**  
基于YOLOv8的智能火灾烟雾检测系统 