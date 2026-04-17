# MNIST手写数字识别项目

基于PyTorch和CNN卷积神经网络的MNIST手写数字识别项目。

## 项目简介

本项目是一个完整的MNIST手写数字识别解决方案，采用CNN（卷积神经网络）进行图像分类。项目设计注重教学演示目的，代码结构清晰，注释详细，适合初学者学习深度学习和计算机视觉。

## 项目目录结构

```
Handwriten_MNIST/
├── config.py              # 配置文件，包含所有超参数
├── train.py               # 训练脚本
├── evaluate.py            # 评估脚本
├── demo.py                # 演示脚本
├── web_app.py             # Flask Web应用
├── requirements.txt       # 项目依赖
├── README.md              # 项目说明文档
├── data/                  # 数据处理模块
│   ├── __init__.py
│   ├── dataset.py         # 数据加载和预处理
│   └── mnist_data/        # MNIST数据集
├── models/                # 模型定义和保存
│   ├── __init__.py
│   ├── cnn_model.py       # CNN模型架构
│   ├── mnist_cnn_best.pth # 最佳模型（已训练）
│   ├── mnist_cnn_model.pth
│   └── mnist_cnn_epoch_*.pth  # 各轮次模型
└── results/               # 结果输出目录
    ├── training_history.json   # 训练历史
    ├── evaluation_results.json # 评估结果
    ├── confusion_matrix.png    # 混淆矩阵
    ├── per_class_accuracy.png  # 各类别准确率
    ├── sample_predictions.png  # 样本预测图
    ├── demo_results.png        # 演示结果
    ├── single_prediction.png   # 单张预测结果
    └── tensorboard/            # TensorBoard日志
```

## 环境配置

### 1. 安装依赖

```bash
# 克隆项目后安装依赖
cd Handwriten_MNIST
pip install -r requirements.txt
```

### 2. 环境要求

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (可选，用于GPU训练)

## 快速开始

项目已包含预训练模型，可直接运行演示或评估。

### 评估模型

```bash
python evaluate.py
```

输出包括：
- 测试集准确率：99.13%
- 混淆矩阵
- 各类别准确率
- 详细分类报告

### 运行演示

```bash
python demo.py
```

演示内容包括：
- 测试集图像预测展示
- 单张图像预测演示
- 预测概率分布可视化

### 启动Web界面（手写画板）

```bash
python web_app.py
```

然后在浏览器中打开: http://localhost:5000

功能特点：
- 🎨 网页手写画板，支持鼠标和触摸输入
- 🔍 实时数字识别，显示预测结果和置信度
- 📊 各类别概率分布可视化
- 📝 可加载MNIST数据集样本进行测试

## 训练模型

如需重新训练模型：

```bash
python train.py
```

训练过程中会：
- 自动下载MNIST数据集
- 显示训练进度和各项指标
- 每轮结束后进行验证
- 保存最佳模型到 `models/` 目录
- 记录TensorBoard日志

### 查看训练曲线

训练完成后，使用TensorBoard查看训练曲线：

```bash
tensorboard --logdir=results/tensorboard
```

然后在浏览器中打开: http://localhost:6006

TensorBoard记录的内容：
- Loss/train - 训练损失曲线
- Loss/val - 验证损失曲线
- Accuracy/train - 训练准确率曲线
- Accuracy/val - 验证准确率曲线
- Learning_Rate - 学习率变化曲线

## 模型架构

项目实现了一个针对MNIST优化的CNN模型：

```
输入层: 1 x 28 x 28 (灰度图像)
    ↓
卷积层1: Conv2d(1→32) + BatchNorm + ReLU + MaxPool
    ↓
卷积层2: Conv2d(32→64) + BatchNorm + ReLU + MaxPool
    ↓
Dropout: 0.25
    ↓
全连接层1: Linear(3136→128) + ReLU + Dropout
    ↓
全连接层2: Linear(128→10)
    ↓
输出层: 10类概率分布
```

### 模型特点

- **卷积层**: 使用3x3卷积核进行特征提取
- **批归一化**: 加速训练，提高稳定性
- **Dropout**: 防止过拟合
- **权重初始化**: 使用Kaiming初始化

## 训练配置

可在 `config.py` 中修改以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| BATCH_SIZE | 64 | 批次大小 |
| LEARNING_RATE | 0.001 | 学习率 |
| NUM_EPOCHS | 10 | 训练轮数 |
| CONV1_OUT_CHANNELS | 32 | 第一层卷积通道数 |
| CONV2_OUT_CHANNELS | 64 | 第二层卷积通道数 |
| FC1_HIDDEN | 128 | 全连接层隐藏单元数 |
| DROPOUT_RATE | 0.25 | Dropout比率 |

## 数据增强

训练时使用以下数据增强技术：

- 随机旋转: ±10度
- 随机平移: 10%
- 随机缩放: 0.9-1.1倍
- 归一化: 使用MNIST标准均值(0.1307)和标准差(0.3081)

## 项目成果

当前模型的测试准确率达到 99.13%！

```
测试损失: 0.0252
测试准确率: 99.13%
```

### 各类别准确率

- 数字 0: 99.80%
- 数字 1: 99.74%
- 数字 2: 99.22%
- 数字 3: 99.50%
- 数字 4: 99.39%
- 数字 5: 99.55%
- 数字 6: 99.58%
- 数字 7: 99.22%
- 数字 8: 98.87%
- 数字 9: 99.21%

### 生成的可视化图表

1. **混淆矩阵**: 展示各数字的分类情况
2. **各类别准确率**: 展示每个数字的识别准确率
3. **样本预测图**: 展示测试图像及其预测结果
4. **训练曲线**: 可通过TensorBoard查看

## 扩展学习

### 尝试不同的模型

项目提供了两个模型版本：

1. **基础CNN模型** (`MNIST_CNNModel`): 本项目默认使用
2. **带注意力机制的模型** (`CNNModelWithAttention`): 增强版，可尝试

### 进一步优化

- 调整网络架构（增加更多卷积层）
- 使用不同的优化器（SGD, AdamW等）
- 尝试学习率调度策略
- 使用更高级的数据增强（mixup, cutout等）
- 模型集成（多个模型投票）

## 常见问题

### Q: 训练时内存不足怎么办？
A: 减小 `config.py` 中的 `BATCH_SIZE`

### Q: 如何使用GPU训练？
A: 确保已安装CUDA版本的PyTorch，程序会自动检测并使用GPU

### Q: 如何继续训练？
A: 加载保存的checkpoint文件，修改 `config.py` 中的参数继续训练

### Q: TensorBoard无法启动？
A: 确保已安装tensorboard: `pip install tensorboard`

## 参考资料

- [MNIST数据集](http://yann.lecun.com/exdb/mnist/)
- [PyTorch官方文档](https://pytorch.org/docs/)
- [卷积神经网络详解](https://cs231n.github.io/convolutional-networks/)

## 许可证

本项目仅供学习交流使用。

## 作者

项目创建用于教学目的。