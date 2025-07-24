# Handwritten-digit-recognition

基于PyTorch实现的手写数字识别系统，使用卷积神经网络(CNN)对MNIST数据集进行训练和识别。

## 项目介绍

本项目实现了一个高效的手写数字识别系统，采用卷积神经网络架构，能够达到99%以上的准确率。系统包含完整的训练、评估、模型保存和结果可视化功能，支持GPU加速训练。

## 功能特点

- 使用CNN模型实现高精度手写数字识别
- 自动下载并预处理MNIST数据集
- 训练过程实时监控损失和准确率
- 支持模型保存与加载
- 生成训练指标可视化图表
- 支持CPU/GPU自动切换

## 项目结构

```
Handwritten-digit-recognition/
├── data/               # 存储MNIST数据集
├── docs/               # 项目文档
├── models/             # 保存训练好的模型
├── results/            # 存储训练结果和可视化图表
├── scripts/            # 辅助脚本
├── src/                # 源代码
│   └── model.py        # 模型定义和训练代码
└── requirements.txt    # 项目依赖
```

## 环境要求

- Python 3.8+
- PyTorch 2.0.0+
- torchvision 0.15.1+
- matplotlib 3.7.1+
- numpy 1.24.3+

## 安装步骤

1. 克隆本仓库
```bash
git clone <repository-url>
cd Handwritten-digit-recognition
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型
```bash
python src/model.py
```

训练过程中会显示每个批次的损失值，每个epoch结束后会显示训练集和测试集的准确率。训练完成后，模型将保存到`models/digit_recognizer.pth`，训练指标图表将保存到`results/metrics.png`。

### 模型结构

网络结构包含以下层：
- 卷积层1: 32个3x3卷积核
- 卷积层2: 64个3x3卷积核
- 池化层: 2x2最大池化
- 全连接层1: 128个神经元
- Dropout层: 防止过拟合
- 全连接层2: 10个神经元(输出层)

## 结果展示

训练完成后，系统会自动生成损失曲线和准确率曲线，展示模型在训练集和测试集上的表现。

## 许可证

本项目采用MIT许可证 - 详情参见LICENSE文件

## 致谢

- MNIST数据集提供者
- PyTorch开发团队