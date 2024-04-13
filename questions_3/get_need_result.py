import numpy as np

def convert_boxes(boxes):
    xywh = boxes.xywh
    xywh = xywh.tolist()
    #将二维列表转化为整数二维列表
    xywh = [[int(j) for j in i] for i in xywh]
    cls = boxes.cls
    cls = cls.tolist()
    cls = [int(i) for i in cls]
    return xywh,cls