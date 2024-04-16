import time
import os

from torch.utils.tensorboard import SummaryWriter
from torchvision.models import ResNet18_Weights
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import matplotlib.pyplot as plt
from torchvision import models
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import torch.optim as optim
# import model

# 获取计算硬件
# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

from torchvision import transforms

# 训练集图像预处理：缩放裁剪、图像增强、转 Tensor、归一化
train_transform = transforms.Compose([transforms.RandomResizedCrop(224),#224*224
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])

# 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
test_transform = transforms.Compose([# 640*480
                                     transforms.RandomResizedCrop(224),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                    ])

# 数据集文件夹路径
dataset_dir = 'data'
train_path = os.path.join(dataset_dir, 'train')
test_path = os.path.join(dataset_dir, 'val')
print('训练集路径', train_path)
print('测试集路径', test_path)
# 载入训练集
train_dataset = datasets.ImageFolder(train_path, train_transform)
# 载入测试集
test_dataset = datasets.ImageFolder(test_path, test_transform)
# 各类别名称
class_names = train_dataset.classes
n_class = len(class_names)
# 映射关系：类别 到 索引号
print(train_dataset.class_to_idx)

# 映射关系：索引号 到 类别
idx_to_labels = {y:x for x,y in train_dataset.class_to_idx.items()}

# 保存为本地的 npy 文件
np.save('idx_to_labels.npy', idx_to_labels)
np.save('labels_to_idx.npy', train_dataset.class_to_idx)

BATCH_SIZE = 128

# 训练集的数据加载器
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          # num_workers=0#线程数
                         )

# 测试集的数据加载器
test_loader = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         # num_workers=0
                        )


model = models.resnet18(weights = ResNet18_Weights.IMAGENET1K_V1) # 载入预训练模型
# model.fc = nn.Linear(model.fc.in_features, n_class)
# print(model.fc)  # 查看修改后的全连接层
# model = torch.load('checkpoints/mymodel/zzsb.pth', map_location=torch.device('cuda'))
# model = model.CNN_easy(4)
# optimizer = optim.Adam(model.fc.parameters())  # 只优化全连接层的参数
learning_rate = 5e-3
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)#优化器
model = model.to(device)

writer = SummaryWriter('logs_train')
#启动tensorboard
# tensorboard --logdir=logs_train
#安装tensorboard，pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
correct_over = 0
criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
EPOCHS = 50
for epoch in tqdm(range(EPOCHS)):
    model.train()
    for images, labels in train_loader:  # 获取训练集的一个 batch，包含数据和标注
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)  # 前向预测，获得当前 batch 的预测结果
        loss = criterion(outputs, labels)  # 比较预测结果和标注，计算当前 batch 的交叉熵损失函数
        writer.add_scalar('Loss/train', loss.item(), epoch) # 记录训练损失


        optimizer.zero_grad()
        loss.backward()  # 损失函数对神经网络权重反向传播求梯度
        optimizer.step()  # 优化更新神经网络权重
    # 测试集上的准确率
    model.eval()
    with torch.no_grad():# 不计算梯度
        correct = 0
        total = 0
        for images, labels in tqdm(test_loader): # 获取测试集  的一个 batch，包含数据和标注
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)              # 前向预测，获得当前 batch 的预测置信度
            _, preds = torch.max(outputs, 1)     # 获得最大置信度对应的类别，作为预测结果
            total += labels.size(0)
            correct += (preds == labels).sum()   # 预测正确样本个数
            writer.add_scalar('Accuracy/test', 100 * correct / total, epoch)  # 记录测试准确率
        print('测试集上的准确率为 {:.3f} %'.format(100 * correct / total))
        if correct > correct_over:
            torch.save(model, 'checkpoints/mymodel/zzsb.pth')
            correct_over = correct