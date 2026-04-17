"""
演示脚本 - MNIST手写数字识别
提供交互式演示功能
"""

import os
import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# 忽略matplotlib字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

import config
from data.dataset import get_device
from models.cnn_model import MNIST_CNNModel


def load_trained_model(device):
    """
    加载训练好的模型
    
    Args:
        device: 设备
    
    Returns:
        model: 加载的模型
    """
    model_path = os.path.join(config.MODEL_DIR, 'mnist_cnn_best.pth')
    
    if not os.path.exists(model_path):
        print(f"警告: 训练好的模型不存在 ({model_path})")
        print("将使用随机初始化的模型进行演示")
        model = MNIST_CNNModel().to(device)
    else:
        print(f"加载模型: {model_path}")
        model = MNIST_CNNModel()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
    
    return model


def preprocess_image(image):
    """
    预处理图像 - 与训练时保持一致
    
    Args:
        image: PIL图像
    
    Returns:
        tensor: 处理后的张量
    """
    # 转换为灰度图
    if image.mode != 'L':
        image = image.convert('L')
    
    # 调整大小为28x28
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # 转换为numpy数组并归一化到[0,1]
    img_array = np.array(image).astype('float32') / 255.0
    
    # 应用与训练时相同的标准化 (mean=0.1307, std=0.3081)
    img_array = (img_array - 0.1307) / 0.3081
    
    # 转换为张量并添加批次和通道维度 [1, 1, 28, 28]
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    
    return img_tensor


def predict_digit(model, image_tensor, device):
    """
    预测数字
    
    Args:
        model: 模型
        image_tensor: 图像张量
        device: 设备
    
    Returns:
        predicted_digit: 预测的数字
        probabilities: 各类别的概率
        confidence: 置信度
    """
    model.eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        
        # 计算概率
        probabilities = torch.softmax(output, dim=1)[0]
        
        # 获取预测结果
        confidence, predicted = torch.max(probabilities, dim=0)
        
        predicted_digit = predicted.item()
        confidence = confidence.item()
    
    return predicted_digit, probabilities.cpu().numpy(), confidence


def draw_digit_canvas():
    """
    创建手写画布
    
    Returns:
        drawing: PIL ImageDraw对象
        image: PIL Image对象
    """
    # 创建白色背景的图像
    image = Image.new('L', (280, 280), color=255)
    draw = ImageDraw.Draw(image)
    
    return draw, image


def demo_with_test_images(model, device):
    """
    使用测试集图像进行演示
    
    Args:
        model: 模型
        device: 设备
    """
    from torchvision import datasets
    
    print("\n" + "=" * 50)
    print("使用MNIST测试集图像进行演示")
    print("=" * 50)
    
    # 加载测试集
    test_dataset = datasets.MNIST(
        root=config.MNIST_DATA_PATH,
        train=False,
        download=True
    )
    
    # 随机选择一些图像
    np.random.seed(42)
    indices = np.random.choice(len(test_dataset), 10, replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for idx, ax in zip(indices, axes):
        # 获取图像和标签
        img, label = test_dataset[idx]
        
        # 预处理 - 与训练时保持一致
        img_tensor = preprocess_image(img)
        
        # 预测
        pred, probs, conf = predict_digit(model, img_tensor, device)
        
        # 显示
        ax.imshow(img, cmap='gray')
        color = 'green' if pred == label else 'red'
        ax.set_title(f'真实: {label} | 预测: {pred}\n置信度: {conf*100:.1f}%', 
                    color=color, fontsize=10)
        ax.axis('off')
    
    plt.suptitle('MNIST手写数字识别演示', fontsize=14)
    plt.tight_layout()
    demo_path = os.path.join(config.RESULTS_DIR, 'demo_results.png')
    plt.savefig(demo_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n演示结果已保存到: {demo_path}")


def demo_interactive(model, device):
    """
    交互式演示
    
    Args:
        model: 模型
        device: 设备
    """
    print("\n" + "=" * 50)
    print("交互式演示")
    print("=" * 50)
    print("请在弹出的窗口中绘制数字...")
    print("注意: 这个演示需要图形界面支持")
    print("如果没有图形界面，请使用脚本的其他演示功能")
    
    try:
        # 创建一个简单的交互式演示
        import matplotlib
        matplotlib.use('TkAgg')
        
        # 创建画布
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('点击选择测试图像', fontsize=14)
        
        # 加载测试集
        from torchvision import datasets
        test_dataset = datasets.MNIST(
            root=config.MNIST_DATA_PATH,
            train=False,
            download=True
        )
        
        # 显示可点击的图像
        rows, cols = 3, 3
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                img, label = test_dataset[idx]
                
                ax_img = fig.add_axes([0.1 + j * 0.28, 0.65 - i * 0.28, 0.25, 0.25])
                ax_img.imshow(img, cmap='gray')
                ax_img.axis('off')
                ax_img.set_title(f'样本 {idx+1}', fontsize=9)
        
        def on_click(event):
            if event.inaxes:
                # 获取点击位置的索引
                x, y = event.xdata, event.ydata
                if x is not None and y is not None:
                    idx = int((0.65 - y) / 0.28) * 3 + int((x - 0.1) / 0.28)
                    if 0 <= idx < 9:
                        img, label = test_dataset[idx]
                        
                        # 预处理和预测 - 与训练时保持一致
                        img_tensor = preprocess_image(img)
                        pred, probs, conf = predict_digit(model, img_tensor, device)
                        
                        print(f"\n样本 {idx+1}:")
                        print(f"  真实标签: {label}")
                        print(f"  预测结果: {pred}")
                        print(f"  置信度: {conf*100:.2f}%")
                        print(f"  各类别概率: ")
                        for i in range(10):
                            print(f"    数字 {i}: {probs[i]*100:.2f}%")
        
        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()
        
    except Exception as e:
        print(f"交互式演示不可用: {e}")
        print("将使用批量演示替代")


def plot_prediction_probabilities(probabilities):
    """
    绘制预测概率分布
    
    Args:
        probabilities: 各类别的概率
    """
    plt.figure(figsize=(10, 6))
    digits = range(10)
    probs = probabilities * 100
    
    bars = plt.bar(digits, probs, color='steelblue', edgecolor='black')
    
    # 找出最高概率
    max_idx = np.argmax(probs)
    bars[max_idx].set_color('green')
    
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{prob:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('数字', fontsize=12)
    plt.ylabel('概率 (%)', fontsize=12)
    plt.title('预测概率分布', fontsize=14)
    plt.xticks(digits)
    plt.ylim(0, max(probs) * 1.2)
    plt.grid(axis='y', alpha=0.3)
    
    prob_path = os.path.join(config.RESULTS_DIR, 'prediction_probabilities.png')
    plt.savefig(prob_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"概率分布图已保存到: {prob_path}")


def demo_single_prediction(model, device):
    """
    单张图像预测演示
    
    Args:
        model: 模型
        device: 设备
    """
    from torchvision import datasets
    
    print("\n" + "=" * 50)
    print("单张图像预测演示")
    print("=" * 50)
    
    # 加载测试集
    test_dataset = datasets.MNIST(
        root=config.MNIST_DATA_PATH,
        train=False,
        download=True
    )
    
    # 选择一个样本
    idx = 42  # 选择第42个样本
    img, label = test_dataset[idx]
    
    print(f"\n选择测试图像索引: {idx}")
    print(f"真实标签: {label}")
    
    # 预处理 - 与训练时保持一致
    img_tensor = preprocess_image(img)
    
    # 预测
    pred, probs, conf = predict_digit(model, img_tensor, device)
    
    print(f"预测结果: {pred}")
    print(f"置信度: {conf*100:.2f}%")
    
    # 显示图像
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    color = 'green' if pred == label else 'red'
    plt.title(f'真实: {label} | 预测: {pred}', color=color, fontsize=14)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    digits = range(10)
    bars = plt.bar(digits, probs * 100, color='steelblue')
    bars[pred].set_color('red')
    plt.xlabel('数字')
    plt.ylabel('概率 (%)')
    plt.title('各类别预测概率')
    plt.xticks(digits)
    
    plt.tight_layout()
    
    single_path = os.path.join(config.RESULTS_DIR, 'single_prediction.png')
    plt.savefig(single_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n单张预测结果已保存到: {single_path}")


def main():
    """主函数"""
    print("=" * 60)
    print(" MNIST手写数字识别 - 演示")
    print("=" * 60)
    
    # 获取设备
    device = get_device()
    
    # 加载模型
    model = load_trained_model(device)
    
    # 运行各种演示
    print("\n运行演示...")
    
    # 演示1: 使用测试集图像
    demo_with_test_images(model, device)
    
    # 演示2: 单张图像预测
    demo_single_prediction(model, device)
    
    print("\n演示完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()