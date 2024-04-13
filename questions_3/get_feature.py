'''
Author: wds-dxh wdsnpshy@163.com
Date: 2024-04-12 15:57:25
LastEditors: wds-dxh wdsnpshy@163.com
LastEditTime: 2024-04-12 18:51:19
FilePath: /mathor_cup/questions_1/questions_1.py
Description: 甲骨文特征提取
微信: 15310638214 
邮箱：wdsnpshy@163.com 
Copyright (c) 2024 by ${wds-dxh}, All Rights Reserved. 
'''
import cv2
import numpy as np


class FeatureExtract:
    def __init__(self,kernel = np.ones((2,2),dtype=np.int8), thresh = 100):
        self.kernel = kernel
        self.thresh = thresh
        print('init FeatureExtract')

    def get_feature(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #开运算
        opening1 = cv2.morphologyEx(gray.copy(),cv2.MORPH_OPEN,self.kernel, iterations=1)
        # 闭运算
        closing1 = cv2.morphologyEx(opening1.copy(), cv2.MORPH_CLOSE, self.kernel,iterations=1)
        #ret是阈值，thresh1是二值化后的图像
        ret,thresh1 = cv2.threshold(closing1,self.thresh,255,cv2.THRESH_BINARY)  
        #再次开运算，去除噪声
        opening2 = cv2.morphologyEx(thresh1.copy(),cv2.MORPH_OPEN,self.kernel, iterations=1)

        return opening2


if __name__ == '__main__':
    input_path = './img'
    output_path = './result'
    # 读取文件夹下所有图片
    import os
    img_list = os.listdir(input_path)
    # 创建特征提取对象
    feature_extract = FeatureExtract()
    for img_name in img_list:
        img = cv2.imread(os.path.join(input_path, img_name))
        feature = feature_extract.get_feature(img)
        cv2.imwrite(os.path.join(output_path, img_name), feature)
        print('save img:', img_name)