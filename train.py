"""
训练脚本 - MNIST手写数字识别
使用CNN卷积神经网络进行训练
"""

import os
import time
import json
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

# 忽略matplotlib字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

import config
from data.dataset import get_data_loaders, get_device
from models.cnn_model import MNIST_CNNModel, print_model_summary


class Trainer:
    """
    训练器类
    负责模型的训练和验证
    """
    
    def __init__(self, model, train_loader, val_loader, device, criterion, optimizer):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        
        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=2 
            
        )
        
        # 训练历史记录
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # 最佳验证准确率
        self.best_val_acc = 0.0
        
        # 确保结果目录存在
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        
        # 初始化TensorBoard
        tensorboard_dir = os.path.join(config.RESULTS_DIR, 'tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tensorboard_dir)
        print(f"TensorBoard日志目录: {tensorboard_dir}")
        print("查看训练曲线命令: tensorboard --logdir={tensorboard_dir}".format(tensorboard_dir=tensorboard_dir))
        
        # 记录模型结构
        self.writer.add_text('model_summary', str(model))
        self.writer.flush()
    
    def train_epoch(self, epoch):
        """
        训练一个epoch
        
        Args:
            epoch: 当前epoch编号
        
        Returns:
            avg_loss: 平均训练损失
            accuracy: 训练准确率
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # 将数据移到设备上
            data, target = data.to(self.device), target.to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 打印训练进度
            if (batch_idx + 1) % config.LOG_INTERVAL == 0:
                progress = 100. * (batch_idx + 1) / len(self.train_loader)
                print(f'  Epoch [{epoch}] Progress: {progress:.1f}% | '
                      f'Loss: {loss.item():.4f} | '
                      f'Acc: {100. * correct / total:.2f}%')
        
        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """
        验证模型
        
        Returns:
            avg_loss: 平均验证损失
            accuracy: 验证准确率
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = val_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, num_epochs):
        """
        训练模型
        
        Args:
            num_epochs: 训练轮数
        
        Returns:
            history: 训练历史记录
        """
        print("=" * 60)
        print("开始训练")
        print("=" * 60)
        
        total_start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 40)
            
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 更新学习率
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            epoch_time = time.time() - epoch_start_time
            
            # 打印当前epoch结果
            print(f"\n  训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%")
            print(f"  验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.2f}%")
            print(f"  学习率: {current_lr:.6f} | 用时: {epoch_time:.2f}秒")
            
            # 记录到TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            self.writer.flush()
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model(is_best=True)
                print(f"  *** 最佳模型已保存! 验证准确率: {val_acc:.2f}% ***")
            
            # 定期保存模型
            if epoch % config.SAVE_MODEL_EVERY == 0:
                self.save_model(epoch=epoch)
        
        total_time = time.time() - total_start_time
        
        print("\n" + "=" * 60)
        print("训练完成!")
        print(f"总用时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
        print(f"最佳验证准确率: {self.best_val_acc:.2f}%")
        print("=" * 60)
        
        # 保存训练历史
        self.save_history()
        
        # 关闭TensorBoard writer
        self.writer.close()
        
        return self.history
    
    def save_model(self, is_best=False, epoch=None):
        """保存模型"""
        # 保存模型参数
        model_path = config.MODEL_SAVE_PATH
        if epoch is not None:
            model_path = os.path.join(config.MODEL_DIR, f'mnist_cnn_epoch_{epoch}.pth')
        
        torch.save({
            'epoch': epoch if epoch else config.NUM_EPOCHS,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
        }, model_path)
        
        if is_best:
            best_path = os.path.join(config.MODEL_DIR, 'mnist_cnn_best.pth')
            torch.save(self.model.state_dict(), best_path)
    
    def save_history(self):
        """保存训练历史"""
        history_path = os.path.join(config.RESULTS_DIR, 'training_history.json')
        
        # 转换numpy数组为列表(如果存在)
        history = {}
        for key, value in self.history.items():
            history[key] = [float(v) for v in value]
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        
        print(f"\n训练历史已保存到: {history_path}")


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)


def main():
    """主函数"""
    print("=" * 60)
    print(" MNIST手写数字识别 - CNN训练")
    print("=" * 60)
    
    # 设置随机种子
    set_seed(config.RANDOM_SEED)
    print(f"随机种子: {config.RANDOM_SEED}")
    
    # 获取设备和数据加载器
    device = get_device()
    print("\n加载数据集...")
    train_loader, val_loader, test_loader = get_data_loaders()
    
    # 创建模型
    print("\n创建模型...")
    model = MNIST_CNNModel().to(device)
    print_model_summary(model)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # 创建训练器并开始训练
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        criterion=criterion,
        optimizer=optimizer
    )
    
    # 开始训练
    history = trainer.train(num_epochs=config.NUM_EPOCHS)
    
    print("\n训练完成!")
    
    return history


if __name__ == '__main__':
    main()