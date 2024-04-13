'''
Author: wds-dxh wdsnpshy@163.com
Date: 2024-04-12 19:32:31
LastEditors: wds-dxh wdsnpshy@163.com
LastEditTime: 2024-04-12 19:48:27
FilePath: /mathor_cup/questions_2/get_data.py
Description: 把标注数据转换为YOLO格式
微信: 15310638214 
邮箱：wdsnpshy@163.com 
Copyright (c) 2024 by ${wds-dxh}, All Rights Reserved. 
'''
import json
import os
from PIL import Image

# 定义转换函数
def convert_to_yolo_format(json_file_path, image_file_path, output_dir):
    # 读取JSON文件
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # 读取图片尺寸
    image = Image.open(image_file_path)
    width, height = image.size

    # 转换为YOLO格式
    yolo_annotations = []
    for ann in data['ann']:
        x_min, y_min, x_max, y_max, class_id= ann
        x_center = (x_min + x_max) / (2 * width)
        y_center = (y_min + y_max) / (2 * height)
        width_ratio = (x_max - x_min) / width
        height_ratio = (y_max - y_min) / height
        yolo_annotations.append(f"{int(0)} {x_center:.6f} {y_center:.6f} {width_ratio:.6f} {height_ratio:.6f}")

    # 保存为YOLO格式的文本文件
    yolo_file_name = os.path.splitext(os.path.basename(json_file_path))[0] + '.txt'
    yolo_file_path = os.path.join(output_dir, yolo_file_name)
    
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(yolo_file_path, 'w') as f:
        for ann in yolo_annotations:
            f.write(ann + '\n')

    print(f"YOLO格式标注已保存到 {yolo_file_path}")

# 设置文件夹路径
folder_path = '/Users/dsw/Downloads/file/2_Train'  # 替换为您的文件夹路径
output_dir = 'txt'  # TXT文件的输出目录

# 遍历文件夹中的所有JSON文件
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        json_file_path = os.path.join(folder_path, filename)
        image_file_path = os.path.join(folder_path, os.path.splitext(filename)[0] + '.jpg')
        convert_to_yolo_format(json_file_path, image_file_path, output_dir)
