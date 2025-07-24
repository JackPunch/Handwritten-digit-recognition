import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class DigitRecognizer(nn.Module):
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        # Dropout防止过拟合
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 卷积->ReLU->池化
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 展平特征图
        x = x.view(-1, 64 * 7 * 7)
        # 全连接层->ReLU->Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # 输出层
        x = self.fc2(x)
        return x

class DigitRecognitionSystem:
    def __init__(self):
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 初始化模型
        self.model = DigitRecognizer().to(self.device)
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # 数据转换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        # 加载数据集
        self.train_dataset = datasets.MNIST(
            root='../data', train=True, download=True, transform=self.transform
        )
        self.test_dataset = datasets.MNIST(
            root='../data', train=False, download=True, transform=self.transform
        )
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=64, shuffle=True
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=64, shuffle=False
        )
        # 训练历史
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []

    def train(self, epochs=5):
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                # 前向传播
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 统计训练数据
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 每100个批次打印一次信息
                if i % 100 == 99:
                    print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

            # 计算每个epoch的准确率
            train_accuracy = 100 * correct / total
            self.train_accuracies.append(train_accuracy)
            self.train_losses.append(running_loss / len(self.train_loader))

            # 在测试集上评估
            test_loss, test_accuracy = self.evaluate()
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_accuracy)

            print(f'Epoch {epoch+1}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')

        print('Finished Training')

    def evaluate(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss /= len(self.test_loader)
        test_accuracy = 100 * correct / total
        return test_loss, test_accuracy

    def save_model(self, path='../models/digit_recognizer.pth'):
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}')

    def load_model(self, path='../models/digit_recognizer.pth'):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        print(f'Model loaded from {path}')

    def plot_metrics(self):
        plt.figure(figsize=(12, 4))

        # 绘制损失
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # 绘制准确率
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.test_accuracies, label='Test Accuracy')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.tight_layout()
        plt.savefig('../results/metrics.png')
        plt.show()

if __name__ == '__main__':
    # 创建识别系统实例
    digit_system = DigitRecognitionSystem()
    # 训练模型
    digit_system.train(epochs=5)
    # 保存模型
    digit_system.save_model()
    # 绘制训练指标
    digit_system.plot_metrics()