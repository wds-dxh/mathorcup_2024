'''
Author: wds-dxh wdsnpshy@163.com
Date: 2024-04-12 15:57:25
LastEditors: wds-dxh wdsnpshy@163.com
LastEditTime: 2024-04-14 15:57:12
FilePath: /mathor_cup/questions_3/get_feature.py
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

        #膨胀
        dilation = cv2.dilate(closing1,self.kernel,iterations = 1)
        #高斯滤波
        gaussian = cv2.GaussianBlur(dilation,(3,3),0)#高斯滤波，参数：1.原图像，2.卷积核大小，3.x方向标准差
    
        #腐蚀
        erosion = cv2.erode(gaussian,self.kernel,iterations = 1)
        #开运算
        opening = cv2.morphologyEx(erosion,cv2.MORPH_OPEN,self.kernel, iterations=1)
        # ret是阈值，thresh1是二值化后的图像
        _,thresh1 = cv2.threshold(opening,self.thresh,255,cv2.THRESH_BINARY)  
        #转换为三通道
        thresh1 = cv2.cvtColor(thresh1,cv2.COLOR_GRAY2BGR)

        return thresh1


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