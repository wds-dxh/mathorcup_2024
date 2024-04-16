'''
Author: wds-dxh wdsnpshy@163.com
Date: 2024-03-17 15:13:44
LastEditors: wds-dxh wdsnpshy@163.com
LastEditTime: 2024-04-15 19:36:07
FilePath: /mathor_cup/questions_3/main.py
Description: 
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
import openpyxl
# 加载YOLOv8模型
model = YOLO('/Users/dsw/Library/CloudStorage/OneDrive-个人/workspace/now/mathor_cup/在已有模型基础上面训练/weights/best.pt')


def get_result(img,conf=0.45):
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    results = model.predict(img,conf=0.1,imgsz=(640, 640),max_det=10,save=False,iou=0.1)
    # 在帧上可视化结果
    annotated_frame = results[0].plot()
    boxes = results[0].boxes
    xywh, cls = convert_boxes(boxes)
    return xywh, cls, annotated_frame


input_path = './Figures'
output_path = './result'
excel_path = './Test_results1.xlsx'
# 读取文件夹下所有图片
img_list = os.listdir(input_path)
feature_extract = FeatureExtract()

# 创建Excel文件并添加表头
workbook = openpyxl.load_workbook(excel_path)
sheet = workbook.active
# sheet.append(['图像名称', '标记'])

for img_name in img_list:
    img = cv2.imread(os.path.join(input_path, img_name))
    img = feature_extract.get_feature(img)
    #从灰度图转换为三通道图 #不需要转换，本来就已经转换回去了

    xywh, cls, annotated_frame = get_result(img)

    # 保存xmini ymini xmaxi ymaxi cls   1️以列表形式保存
    cv2.imwrite(os.path.join(output_path, img_name), annotated_frame)
    cv2.imshow('result', annotated_frame)
    # cv2.waitKey(500)
    print('save img:', img_name)
    for i in range(len(xywh)):
        xywh[i].append(1)
    xywh_str = ', '.join([f"[{', '.join(map(str, sublist))}]" for sublist in xywh])
    sheet.append([img_name, xywh_str])
    workbook.save(excel_path)