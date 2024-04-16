'''
Author: wds-dxh wdsnpshy@163.com
Date: 2024-04-15 19:00:21
LastEditors: wds-dxh wdsnpshy@163.com
LastEditTime: 2024-04-15 20:18:36
FilePath: /mathor_cup/questions_4/preditc.py
Descriptioc v
微信: 15310638214 
邮箱：wdsnpshy@163.com 
Copyright (c) 2024 by ${wds-dxh}, All Rights Reserved. 
'''
import os
import time
import numpy as np
import pandas as pd
import cv2 # opencv-python
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm # 进度条
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms


# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)
# 导入中文字体，指定字号
font = ImageFont.truetype('SimHei.ttf', 32)
idx_to_labels = np.load('idx_to_labels.npy', allow_pickle=True).item()
print(idx_to_labels)
model = torch.load('Oracle_Script_Recognition.pth', map_location=torch.device('cpu'))
model = model.eval().to(device)


# 测试集图像预处理-RCTN：缩放裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                    ])



# 处理帧函数
def process_frame(img):
    start_time = time.time()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)# array 转 PIL
    input_img = test_transform(img_pil).unsqueeze(0).to(device)
    pred_logits = model(input_img)  # pred_logits:是一个tensor
    pred_softmax = F.softmax(pred_logits, dim=1)

    top_n = torch.topk(pred_softmax, 1)  # 仅获取最高置信度的一个结果
    values = top_n.values.cpu().detach().numpy().squeeze()
    indices = top_n.indices.cpu().detach().numpy().squeeze()
    draw = ImageDraw.Draw(img_pil)
    pred_class = idx_to_labels[int(indices)]  # 将 NumPy 数组转换为标量值作为字典的键
    text = '{:<8} {:>.3f}'.format(pred_class, values)
    # 文字坐标，中文字符串，字体，rgba颜色
    draw.text((50, 100), text, font=font, fill=(255, 0, 0, 1))
    img = np.array(img_pil)  # PIL 转 array
    img = np.array(img_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    end_time = time.time()
    FPS = 1 / (end_time - start_time)
    img = cv2.putText(img, 'FPS: ' + str(int(FPS)), (50, 80),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      1,#字体大小
                      (255, 0, 255),
                      4,
                      cv2.LINE_AA)#org:左下角坐标

    return img, pred_class, values


input_path = './result_split'
output_path = './result'
img_list = os.listdir(input_path)
for img_name in img_list:
    img = cv2.imread(os.path.join(input_path, img_name))
    if img is None:
        continue
    img = cv2.resize(img, (640, 480))
    img, pred_class, values = process_frame(img)
    cv2.imshow('result', img)
    cv2.waitKey(1000)
    print('save img:', img_name)
    cv2.imwrite(os.path.join(output_path, img_name), img)
cv2.destroyAllWindows()






