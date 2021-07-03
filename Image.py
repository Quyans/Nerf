"""
图片模块
"""
import torch


def loadimage(dir):
    """
    读取图片返回tensor
    :param dir: 给出路径，读取图片，
    :return: img ( 512*512*3 )
    """
    img = torch.rand(512*512*3)
    return img

