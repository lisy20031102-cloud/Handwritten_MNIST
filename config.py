"""
配置文件 - MNIST手写数字识别项目
包含所有超参数和路径配置
"""

import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据配置
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MNIST_DATA_PATH = os.path.join(DATA_DIR, 'mnist_data')

# 模型保存路径
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'mnist_cnn_model.pth')

# 训练结果保存路径
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# 训练超参数
BATCH_SIZE = 64          # 批次大小
LEARNING_RATE = 0.001    # 学习率
NUM_EPOCHS = 10          # 训练轮数
TEST_BATCH_SIZE = 1000   # 测试批次大小

# 图像参数
IMAGE_SIZE = 28          # MNIST图像尺寸 28x28
NUM_CLASSES = 10         # 数字0-9，共10类

# CNN模型参数
CONV1_OUT_CHANNELS = 32  # 第一个卷积层输出通道数
CONV2_OUT_CHANNELS = 64  # 第二个卷积层输出通道数
FC1_HIDDEN = 128         # 全连接层隐藏单元数
DROPOUT_RATE = 0.25      # Dropout比率

# 训练相关
RANDOM_SEED = 42         # 随机种子，保证结果可复现
USE_CUDA = True          # 是否使用GPU训练
DEVICE = 'cuda'          # 设备选择 ('cuda' 或 'cpu')

# 数据集划分
TRAIN_RATIO = 0.8        # 训练集比例
VAL_RATIO = 0.2          # 验证集比例

# 日志配置
LOG_INTERVAL = 100       # 日志打印间隔（步数）
SAVE_MODEL_EVERY = 1     # 每几轮保存一次模型

# 打印配置
PRINT_INTERVAL = 10      # 损失打印间隔