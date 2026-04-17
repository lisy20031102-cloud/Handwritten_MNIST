"""
评估脚本 - MNIST手写数字识别
对训练好的模型进行测试和评估
"""

import os
import json
import warnings
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 忽略matplotlib字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

import config
from data.dataset import get_data_loaders, get_device
from models.cnn_model import MNIST_CNNModel


class Evaluator:
    """
    评估器类
    负责模型的测试和评估
    """
    
    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        
        # 预测结果存储
        self.all_predictions = []
        self.all_targets = []
        self.all_probabilities = []
    
    def evaluate(self):
        """
        评估模型
        
        Returns:
            accuracy: 测试准确率
            avg_loss: 平均测试损失
        """
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播
                output = self.model(data)
                test_loss += criterion(output, target).item()
                
                # 获取预测结果
                probabilities = torch.softmax(output, dim=1)
                _, predicted = output.max(1)
                
                # 存储结果
                self.all_predictions.extend(predicted.cpu().numpy())
                self.all_targets.extend(target.cpu().numpy())
                self.all_probabilities.extend(probabilities.cpu().numpy())
                
                # 统计
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = test_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def get_predictions(self):
        """
        获取所有预测结果
        
        Returns:
            predictions: 预测标签
            targets: 真实标签
            probabilities: 预测概率
        """
        return (np.array(self.all_predictions), 
                np.array(self.all_targets), 
                np.array(self.all_probabilities))
    
    def compute_per_class_accuracy(self):
        """
        计算每个类别的准确率
        
        Returns:
            per_class_acc: 每个类别的准确率
        """
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        
        per_class_acc = {}
        for digit in range(10):
            mask = targets == digit
            if mask.sum() > 0:
                acc = (predictions[mask] == targets[mask]).sum() / mask.sum() * 100
                per_class_acc[digit] = acc
            else:
                per_class_acc[digit] = 0.0
        
        return per_class_acc
    
    def plot_confusion_matrix(self, save_path=None):
        """
        绘制混淆矩阵
        
        Args:
            save_path: 保存路径
        """
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        
        # 计算混淆矩阵
        cm = confusion_matrix(targets, predictions)
        
        # 绘制
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=range(10), yticklabels=range(10))
        plt.xlabel('预测标签', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        plt.title('MNIST CNN模型混淆矩阵', fontsize=14)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"混淆矩阵已保存到: {save_path}")
        
        plt.close()
        
        return cm
    
    def plot_per_class_accuracy(self, save_path=None):
        """
        绘制每个类别的准确率
        
        Args:
            save_path: 保存路径
        """
        per_class_acc = self.compute_per_class_accuracy()
        
        digits = list(per_class_acc.keys())
        accuracies = list(per_class_acc.values())
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(digits, accuracies, color='steelblue', edgecolor='black')
        
        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('数字', fontsize=12)
        plt.ylabel('准确率 (%)', fontsize=12)
        plt.title('各类别手写数字识别准确率', fontsize=14)
        plt.xticks(digits)
        plt.ylim(0, 105)
        plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"各类别准确率图已保存到: {save_path}")
        
        plt.close()
        
        return per_class_acc
    
    def plot_sample_predictions(self, num_samples=16, save_path=None):
        """
        绘制样本预测结果
        
        Args:
            num_samples: 展示的样本数量
            save_path: 保存路径
        """
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        
        # 获取原始图像（反归一化）
        mean = 0.1307
        std = 0.3081
        
        # 创建网格
        rows = int(np.sqrt(num_samples))
        cols = num_samples // rows
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx in range(num_samples):
            # 获取原始图像数据
            img = self.test_loader.dataset[idx][0].squeeze()
            img = img * std + mean
            
            axes[idx].imshow(img, cmap='gray')
            
            # 设置标题颜色（正确=绿色，错误=红色）
            if predictions[idx] == targets[idx]:
                color = 'green'
            else:
                color = 'red'
            
            axes[idx].set_title(f'真实: {targets[idx]} | 预测: {predictions[idx]}', 
                               color=color, fontsize=10)
            axes[idx].axis('off')
        
        plt.suptitle('MNIST手写数字识别样例展示', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"样例预测图已保存到: {save_path}")
        
        plt.close()


def load_model(model_path, device):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型路径
        device: 设备
    
    Returns:
        model: 加载的模型
    """
    model = MNIST_CNNModel()
    
    # 尝试加载完整checkpoint或仅权重
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"从checkpoint加载模型 (epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            model.load_state_dict(checkpoint)
            print("从权重文件加载模型")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None
    
    model = model.to(device)
    model.eval()
    
    return model


def save_evaluation_results(results, save_path):
    """
    保存评估结果
    
    Args:
        results: 评估结果字典
        save_path: 保存路径
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"评估结果已保存到: {save_path}")


def main():
    """主函数"""
    print("=" * 60)
    print(" MNIST手写数字识别 - 模型评估")
    print("=" * 60)
    
    # 获取设备
    device = get_device()
    
    # 加载测试数据
    print("\n加载测试数据集...")
    _, _, test_loader = get_data_loaders()
    
    # 模型路径
    model_path = os.path.join(config.MODEL_DIR, 'mnist_cnn_best.pth')
    
    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"\n警告: 模型文件不存在 ({model_path})")
        print("请先运行训练脚本 train.py 训练模型")
        
        # 使用未训练的模型进行演示
        print("\n使用未训练的模型进行演示...")
        model = MNIST_CNNModel().to(device)
    else:
        # 加载训练好的模型
        print(f"\n加载模型: {model_path}")
        model = load_model(model_path, device)
    
    # 创建评估器
    evaluator = Evaluator(model, test_loader, device)
    
    # 评估模型
    print("\n开始评估...")
    avg_loss, accuracy = evaluator.evaluate()
    
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"测试损失: {avg_loss:.4f}")
    print(f"测试准确率: {accuracy:.2f}%")
    
    # 获取预测结果
    predictions, targets, probabilities = evaluator.get_predictions()
    
    # 打印分类报告
    print("\n" + "-" * 40)
    print("分类报告:")
    print("-" * 40)
    report = classification_report(targets, predictions, 
                                   target_names=[str(i) for i in range(10)])
    print(report)
    
    # 计算每个类别的准确率
    per_class_acc = evaluator.compute_per_class_accuracy()
    print("\n各类别准确率:")
    print("-" * 40)
    for digit, acc in per_class_acc.items():
        print(f"  数字 {digit}: {acc:.2f}%")
    
    # 保存评估结果
    results = {
        'test_loss': float(avg_loss),
        'test_accuracy': float(accuracy),
        'per_class_accuracy': {str(k): float(v) for k, v in per_class_acc.items()}
    }
    
    results_path = os.path.join(config.RESULTS_DIR, 'evaluation_results.json')
    save_evaluation_results(results, results_path)
    
    # 绘制并保存可视化结果
    print("\n生成可视化图表...")
    
    # 混淆矩阵
    cm_path = os.path.join(config.RESULTS_DIR, 'confusion_matrix.png')
    evaluator.plot_confusion_matrix(save_path=cm_path)
    
    # 各类别准确率
    acc_path = os.path.join(config.RESULTS_DIR, 'per_class_accuracy.png')
    evaluator.plot_per_class_accuracy(save_path=acc_path)
    
    # 样本预测
    sample_path = os.path.join(config.RESULTS_DIR, 'sample_predictions.png')
    evaluator.plot_sample_predictions(num_samples=16, save_path=sample_path)
    
    print("\n评估完成!")
    print("=" * 60)
    
    return accuracy


if __name__ == '__main__':
    main()