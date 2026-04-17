"""
数据加载和预处理模块
负责MNIST数据集的下载、加载和预处理
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import config


def get_data_transforms():
    """
    获取数据预处理transforms
    
    Returns:
        train_transform: 训练集数据增强变换
        test_transform: 测试集变换（仅归一化）
    """
    # 训练集数据增强：随机旋转、平移、缩放
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),           # 随机旋转±10度
        transforms.RandomAffine(                 # 随机仿射变换
            degrees=0,
            translate=(0.1, 0.1),               # 平移
            scale=(0.9, 1.1)                     # 缩放
        ),
        transforms.ToTensor(),                   # 转换为张量 [0, 255] → [0, 1]
        transforms.Normalize(                    # 归一化到标准正态分布
            mean=(0.1307,),                      # MNIST数据集均值
            std=(0.3081,)                        # MNIST数据集标准差
        )
    ])
    
    # 测试集不需要数据增强
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.1307,),
            std=(0.3081,)
        )
    ])
    
    return train_transform, test_transform


def load_mnist_dataset():
    """
    加载MNIST数据集
    
    Returns:
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        test_dataset: 测试数据集
    """
    train_transform, test_transform = get_data_transforms()
    
    # 加载训练集（包含60000张图片）
    train_dataset = datasets.MNIST(
        root=config.MNIST_DATA_PATH,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # 加载测试集（包含10000张图片）
    test_dataset = datasets.MNIST(
        root=config.MNIST_DATA_PATH,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # 划分训练集和验证集
    total_train = len(train_dataset)
    train_size = int(total_train * config.TRAIN_RATIO)
    val_size = total_train - train_size
    
    # 使用random_split划分数据集，设置随机种子保证可复现
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )
    
    print(f"数据集划分:")
    print(f"  训练集: {train_size} 样本")
    print(f"  验证集: {val_size} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    
    return train_subset, val_subset, test_dataset


def get_data_loaders():
    """
    获取数据加载器
    
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
    """
    train_dataset, val_dataset, test_dataset = load_mnist_dataset()
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,           # 训练集需要打乱顺序
        num_workers=2,          # 使用2个子进程加载数据
        pin_memory=True         # 固定内存，加速数据传输到GPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.TEST_BATCH_SIZE,
        shuffle=False,          # 验证集不需要打乱
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_device():
    """
    获取可用的设备（GPU或CPU）
    
    Returns:
        device: torch.device对象
    """
    if config.USE_CUDA and torch.cuda.is_available():
        device = torch.device(config.DEVICE)
        print(f"使用GPU训练: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("使用CPU训练")
    
    return device


def display_sample_images(data_loader, num_images=6):
    """
    显示数据集中的一些样本图像
    
    Args:
        data_loader: 数据加载器
        num_images: 显示的图片数量
    """
    import matplotlib.pyplot as plt
    
    # 获取一批数据
    images, labels = next(iter(data_loader))
    
    # 反归一化以显示原始图像
    mean = 0.1307
    std = 0.3081
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx in range(num_images):
        # 反归一化
        img = images[idx].squeeze() * std + mean
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title(f'Label: {labels[idx].item()}', fontsize=14)
        axes[idx].axis('off')
    
    plt.suptitle('MNIST手写数字样例', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{config.RESULTS_DIR}/sample_images.png', dpi=150)
    plt.show()
    print(f"样本图像已保存到: {config.RESULTS_DIR}/sample_images.png")


if __name__ == '__main__':
    # 测试数据加载
    print("=" * 50)
    print("测试数据加载模块")
    print("=" * 50)
    
    train_loader, val_loader, test_loader = get_data_loaders()
    device = get_device()
    
    print(f"\n批次信息:")
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  验证批次数: {len(val_loader)}")
    print(f"  测试批次数: {len(test_loader)}")
    
    # 获取一个批次的数据
    images, labels = next(iter(train_loader))
    print(f"\n单批次数据形状:")
    print(f"  图像张量形状: {images.shape}")
    print(f"  标签张量形状: {labels.shape}")
    
    print("\n数据加载模块测试完成！")