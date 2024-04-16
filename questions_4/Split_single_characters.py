'''
Author: wds-dxh wdsnpshy@163.com
Date: 2024-03-17 15:13:44
LastEditors: wds-dxh wdsnpshy@163.com
LastEditTime: 2024-04-15 20:10:49
FilePath: /mathor_cup/questions_4/Split_single_characters.py
Description: 对图像文字分割
微信: 15310638214 
邮箱：wdsnpshy@163.com 
Copyright (c) 2024 by ${wds-dxh}, All Rights Reserved. 
'''
import time
import cv2
import os
os.environ['YOLO_VERBOSE'] = str(False)#不打印yolov8信息
from ultralytics import YOLO
import numpy as np

from get_need_result import convert_boxes
from get_feature import FeatureExtract

# 加载YOLOv8模型
model = YOLO('/Users/dsw/Library/CloudStorage/OneDrive-个人/workspace/now/mathor_cup/在已有模型基础上面训练/weights/best.pt')


def get_result(img,conf=0.45):
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    results = model.predict(img,conf=0.2,imgsz=(640, 640),max_det=10,save=False,iou=0.1)
    # 在帧上可视化结果
    annotated_frame = results[0].plot()
    boxes = results[0].boxes
    xywh, cls = convert_boxes(boxes)
    return xywh, cls, annotated_frame


input_path = './4_Recognize/test'
output_path = './result_split'
# 读取文件夹下所有图片
img_list = os.listdir(input_path)
feature_extract = FeatureExtract()
count  = 1
for img_name in img_list:
    img = cv2.imread(os.path.join(input_path, img_name))
    img = feature_extract.get_feature(img)
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    #从灰度图转换为三通道图 #不需要转换，本来就已经转换回去了
    xywh, cls, annotated_frame = get_result(img)
    # cv2.imshow('img', annotated_frame)
    print('xywh:', xywh)    
    for i in range(len(xywh)):
        x, y, w, h = xywh[i]
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        cropped_object = img[y1:y2, x1:x2] 
        # 使用计数器来命名裁剪出的物体图片
        cropped_img_name = f"{os.path.splitext(img_name)[0]}_{count}.jpg"
        cv2.imwrite(os.path.join(output_path, cropped_img_name), cropped_object)
        # 更新计数器
        count += 1

    print('save img:', img_name)
    count = 1
    


