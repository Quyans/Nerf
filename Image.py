"""
图片模块
"""
import torch
import cv2 as cv
import os


def loadimage(dir):
    """
    读取图片返回tensor
    :param dir: 给出路径，读取图片，
    :return: imgList ( 512*512*3 )
    """

    for root, dirs, files in os.walk(dir):
        for d in dirs:
            print(d)
        for file in files:
            # print(file)
            #读入图像
            img_path = root + '/' + file
            img = cv.imread(img_path, 1)
            # print(img_path, img.shape)
            print(img.shape)   # 756*1000*3
            # print(img[0][0])

    imgList = torch.rand(512*512*3)
    return imgList


loadimage("/home/qys/Documents/gitClone/nerf-master/data/nerf_llff_data/fern/images_8")