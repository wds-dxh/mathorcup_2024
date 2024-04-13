'''
Author: wds-dxh wdsnpshy@163.com
Date: 2024-04-12 22:52:54
LastEditors: wds-dxh wdsnpshy@163.com
LastEditTime: 2024-04-13 11:40:42
FilePath: /mathor_cup/questions_1/test.py
Description: 
微信: 15310638214 
邮箱：wdsnpshy@163.com 
Copyright (c) 2024 by ${wds-dxh}, All Rights Reserved. 
'''
# 导入所需的库
import cv2
import numpy as np
from PIL import Image
def preprocess(image):
    # 将图片转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 对图像进行二值化处理
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # 去除噪声，使用高斯滤波器
    blur = cv2.GaussianBlur(thresh, (5,5), 0)
    # 提取图像边缘，使用Canny边缘检测算法
    edges = cv2.Canny(blur, 50, 150, apertureSize = 3)
    # 对图像进行膨胀操作，以填充空洞和断开的边缘
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(edges,kernel,iterations = 1)
    # 提取图像轮廓，使用findContours函数
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 对图像轮廓进行排序，以找出最大轮廓
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:1]
    # 对最大轮廓进行外接矩形拟合，以得到文字区域的坐标
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt) # x,y为矩形左上角坐标，w,h为矩形宽高,boundingRect是一个函数，用来计算轮廓的外接矩形
    # 对图像进行裁剪，得到文字区域
    cropped = image[y:y+h, x:x+w]
    # 提取图像特征，如颜色直方图、梯度直方图、形状特征等
    color_hist = cv2.calcHist([cropped], [0], None, [256], [0, 256])#calcHist函数用来计算直方图
    # 返回处理后的图片和提取的特征
    return cropped, color_hist



# 读取图片
image1 = cv2.imread("./img/h02060.jpg")
image2 = cv2.imread("./img/w01637.jpg")
image3 = cv2.imread("./img/w01870.jpg")
# 调用预处理函数
cropped1, color_hist1 = preprocess(image1)
cropped2, color_hist2 = preprocess(image2)
cropped3, color_hist3 = preprocess(image3)
# 打印提取的特征
print(color_hist1)
print(color_hist2)
print(color_hist3)