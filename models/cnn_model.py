"""
CNN卷积神经网络模型定义
用于MNIST手写数字识别

模型架构:
    - Conv1: 卷积层 -> ReLU -> 池化
    - Conv2: 卷积层 -> ReLU -> 池化
    - Dropout
    - FC1: 全连接层 -> ReLU -> Dropout
    - FC2: 全连接层 -> 输出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class MNIST_CNNModel(nn.Module):
    """
    MNIST手写数字识别CNN模型
    """
    
    def __init__(self):
        super(MNIST_CNNModel, self).__init__()
        
        # 第一个卷积块
        # 输入: 1 x 28 x 28 (灰度图)
        self.conv1 = nn.Conv2d(
            in_channels=1,                  # 输入通道数（灰度图）
            out_channels=config.CONV1_OUT_CHANNELS,  # 输出通道数
            kernel_size=3,                  # 3x3卷积核
            padding=1                       # 保持图像尺寸
        )
        self.bn1 = nn.BatchNorm2d(config.CONV1_OUT_CHANNELS)  # 批归一化
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)   # 2x2池化，尺寸减半
        # 输出: 32 x 14 x 14
        
        # 第二个卷积块
        self.conv2 = nn.Conv2d(
            in_channels=config.CONV1_OUT_CHANNELS,
            out_channels=config.CONV2_OUT_CHANNELS,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(config.CONV2_OUT_CHANNELS)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 输出: 64 x 7 x 7
        
        # Dropout层
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        
        # 全连接层
        # 输入: 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(
            config.CONV2_OUT_CHANNELS * 7 * 7,
            config.FC1_HIDDEN
        )
        self.fc2 = nn.Linear(
            config.FC1_HIDDEN,
            config.NUM_CLASSES
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用Kaiming初始化，适合ReLU激活函数
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, 1, 28, 28)
        
        Returns:
            output: 输出张量，形状为 (batch_size, 10)
        """
        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # Dropout
        x = self.dropout(x)
        
        # 全连接层
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_feature_maps(self, x):
        """
        获取特征图（用于可视化）
        
        Args:
            x: 输入张量
        
        Returns:
            conv1_features: 第一个卷积层的特征图
            conv2_features: 第二个卷积层的特征图
        """
        # 第一个卷积块
        x1 = F.relu(self.bn1(self.conv1(x)))
        conv1_features = self.pool1(x1)
        
        # 第二个卷积块
        x2 = F.relu(self.bn2(self.conv2(conv1_features)))
        conv2_features = self.pool2(x2)
        
        return conv1_features, conv2_features


class CNNModelWithAttention(nn.Module):
    """
    带注意力机制的CNN模型（增强版）
    """
    
    def __init__(self):
        super(CNNModelWithAttention, self).__init__()
        
        # 第一个卷积块
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # 第二个卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # 第三个卷积块
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Sigmoid()
        )
        
        # 全连接层
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        # 卷积块
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 全局平均池化
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        
        # 注意力机制
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def count_parameters(model):
    """
    统计模型参数数量
    
    Args:
        model: PyTorch模型
    
    Returns:
        total_params: 总参数量
        trainable_params: 可训练参数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_model_summary(model, input_size=(1, 1, 28, 28)):
    """
    打印模型结构摘要
    
    Args:
        model: PyTorch模型
        input_size: 输入尺寸
    """
    print("=" * 60)
    print("模型结构:")
    print("=" * 60)
    print(model)
    print("=" * 60)
    
    total_params, trainable_params = count_parameters(model)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print("=" * 60)


if __name__ == '__main__':
    # 测试模型
    print("=" * 60)
    print("测试CNN模型")
    print("=" * 60)
    
    # 创建模型
    model = MNIST_CNNModel()
    print_model_summary(model)
    
    # 测试前向传播
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 创建测试输入
    test_input = torch.randn(4, 1, 28, 28).to(device)
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(test_input)
    
    print(f"\n输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"\n输出概率 (Softmax):")
    probs = F.softmax(output, dim=1)
    print(probs)
    
    print("\n模型测试完成！")