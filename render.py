# 渲染部分的代码
from __future__ import print_function
import torch
import torch.nn as nn
from network import Net
import cmath
import include.read_write_model as tools

cameras = {}
images = {}

camerasDir = "./data/sparse/cameras.bin"
imagesDir = "./data/sparse/images.bin"

def LoadPose():
    """
    加载计算出的相机的
    :return:
    """
    cameraInfo = tools.read_cameras_binary(camerasDir)
    imagesInfo = tools.read_images_binary(imagesDir)


def calPose(pose):
    """
    :param pose:
    :return: 一个tensor [n,rayStart,rgb] n=512*512  3:光线起始点坐标 3：rgb三维
    """


def imgRender(pose):
    """
    图像渲染
    :param pose: 相机位姿
    :return: img
    """

    [n_square, rayStart, rgb] = calPose(pose)
    n = n_square ** 0.5
    img = [n, n]

    for i in range(n):
        for j in range(n):
            img[i][j] = rayRender(rayStart, rgb)
    return img


def queryGroundTruth(origin, dir):
    """
    查询标签
    :param origin:
    :param dir:
    :return: rgb值
    """
    return torch.rand(3)


def calDistance(x1, x2):
    """
    计算两点间的距离
    :param x1:  顶点1
    :param x2:  顶点2
    :return: dis 距离
    """
    return cmath.sqrt(((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2 + (x1[2]-x2[2])**2))

def rayRender(origin, dir):
    """
    渲染一条光线
    :param origin: 3 vector 光线起点
    :param dir: 2*1 方向
    :return: radiance:3*1
    光线就是 o+td o是光线射出点，t是标量，包围盒近的是tn 远的是tf
    """

    tn = torch.rand(1)
    tf = torch.rand(1)+5

    # 采样code
    N = 20  # 将光线分为N个bins
    step = (tf-tn)/N
    sample_arr = torch.zeros(N+2, 3)    # 最后2维度分别是 tn tf
    sample_net_rgb = torch.zeros(N, 3)  # sample的rgb输出
    sample_net_sig = torch.zeros(N)     # sample的sigma输出
    sample_net_dis = torch.zeros(N-1)   # sample 之间的距离
    for i in range(N):
        sample_arr[i] = origin + (tn + i*step + torch.rand(1).item())*dir
        if i>0:
            sample_net_dis[i-1] = calDistance(sample_arr[i-1],sample_arr[i])

    sample_arr[N] = origin + tn * dir
    sample_arr[N+1] = origin + tf * dir
    # 调用网络
    net = Net()
    loss_function = nn.BCELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    radiance = torch.rand(3)

    Cr = 0
    for ind in range(N):
        sig, rad = net.forward(sample_arr[ind], dir)
        sample_net_sig[ind] = sig
        sample_net_rgb[ind] = rad
        Ti_sum = 0
        for j in range(ind-1):
            Ti_sum -= sample_net_sig[j] * sample_net_dis[j]
        Ti = cmath.exp(Ti_sum)
        if ind < N-1:
            Cr += Ti * (1 - cmath.exp(-sample_net_sig[ind] * sample_net_dis[ind])) * sample_net_rgb[ind]
            continue
        # 得到之后应该进行蒙特卡洛积分 论文P6
        Cr += Ti * (1 - cmath.exp(-sample_net_sig[ind] * calDistance(sample_arr[ind], sample_arr[N+1]))) * sample_net_rgb[ind]

    # 计算出的Cr* 和groundtruth进行比较然后反向传播
    loss = loss_function(Cr, queryGroundTruth(origin, dir))
    loss.backward()
    optimizer.step()

    # 返回光线渲染，即一个RGB的tensor 三个数

    return radiance
