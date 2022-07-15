from base64 import encode
from hashlib import new
from multiprocessing import shared_memory
from os import abort
from turtle import pos
import numpy as np
# import SimpleITK as sitk
import cv2 as cv2
from collections import defaultdict
import matplotlib.pyplot as plt
import numba as nb
import os
import time

from FFD_SSD import cal_ncc


# 反向变换
def backward_trans(source, position_map):# position_map对应local affine返回的结果
    '''根据变换函数position_map完成反向图变换'''
    height,width = source.shape[0],source.shape[1]

    newImg = np.zeros_like(source,dtype='float')   # 变换后的target
    for i in range(height):
        for j in range(width): # 遍历target每一个像素点
            X = position_map[i, j] # source中对应的点（精确位置）
            u, v = X - X.astype(int)
            neigh_i, neigh_j = X.astype(int) # 得到附近像素点和垂直水平距离u,v
            # 进行插值，得到X位置的像素值，也就是newImg在Y处像素值
            if neigh_i < 0 or neigh_j < 0 or neigh_i+1 >source.shape[0]-1  or neigh_j+1 >source.shape[1]-1:
                continue #如果这个点的位置出界了就不填
            newImg[i,j] = (1 - u) * (1 - v) * source[neigh_i, neigh_j] + (1 - u) * v * source[neigh_i,neigh_j + 1] +\
                    u * (1 - v) * source[neigh_i +1,neigh_j] +  u * v * source[neigh_i + 1,neigh_j + 1]

    return newImg  # 变成像素需要的int

def affine_trans(source, para):
    '''局部仿射函数,寻找X=T^{-1} (Y)的函数关系 即position_map'''

    height, width= source.shape
    position_map = defaultdict(lambda: np.eye(2))  # 构造position_map映射字典，初始为0向量
    # print(para.shape)
    G = para[:4].reshape((2,-1))
    b = para[4:].reshape((-1))
    # 得到Gi
    for x in range(height):
        for y in range(width): # 遍历每个个点
            Y = (x,y)
            position_map[Y] = np.dot(G, np.array([x,y]))   # 利用w对Gi加权
            position_map[Y] = position_map[Y]+b     # 把向量最后的1去除
            # print(position_map[Y])


    newImg = backward_trans(source,position_map)
    return newImg

def normalize(img):
    '''
    归一化图片
    '''
    return np.array((img-img.min())/(img.max()-img.min())*255)

#SSD相似性度量
@nb.jit()
def SSD_similarity(source,target):
    '''
    SSD相似性度量
    :param source， target 浮动图像和基准图像
    :output SSD score
    '''
    return ((source-target)**2).mean()



def get_fun_bpline(source, target, grid_points):
    '''
    计算目标函数 包含三部分: transform+插值+计算互相关
    input: source, target, grid_points
    output: ncc value
    '''
    # out = np.zeros_like(source)
    out = affine_trans(source, grid_points)# 得到Y和X对应关系
    result = SSD_similarity(out,target)
    # result = cal_ncc(source,target,5,5)
    return result

def cal_gradient(source,target,grid_points,gradient):
    '''
    计算梯度
    input: source, target,grid_points, old gradient
    output: new gradient update
    '''
    eps=0.0005#超参数，可以改
    a1=get_fun_bpline(source,target,grid_points)
    grid_p = grid_points.copy()
    for i in range(grid_points.shape[0]):
        if i>=4:#如果太大
            eps=0.1
        grid_p[i]+=eps #修改grid的i位元素的大小
        a2 = get_fun_bpline(source, target, grid_p) #每次都要做一次仿射+相似度估计
        grid_p[i]-=eps #再把它改回来
        gradient[i] = (a2-a1)/eps#梯度
        # print('第',i,'个梯度大小为：',gradient[i])
    return gradient

def update_grid_points(grid_points, gradient, alpha):
    '''
    根据梯度下降的逻辑更新grid
    '''
    for i in range(grid_points.shape[0]):
        grid_points[i]=grid_points[i]-gradient[i]*alpha
    return grid_points

def optimization_gd_affine(source, target,grid_points):
    '''
    最优化迭代的具体步骤
    超参数需要进一步调整
    :input 输入图像，特征点
    :output 输出图像，SSD下降曲线
    '''
    loss=[]
    max_iter=500#最多迭代次数
    e =0.0001#迭代精度
    count=0#迭代次数统计
    # alpha = 0.05#初始alpha
    alpha = 0.00005
    gradient = np.zeros_like(grid_points,dtype='float')#初始化梯度
    gradient = cal_gradient(source, target, grid_points, gradient) #求梯度
    out_cnt=0
    while count<max_iter:
        # print(gradient)
        pred_grid_points= grid_points.copy()
        grid_points = update_grid_points(grid_points,gradient,alpha)
        ret1 = get_fun_bpline(source, target, pred_grid_points)#原本点的相关系数
        ret2 = get_fun_bpline(source, target, grid_points)#更新后点的相关系数
        print('original:%9f'%ret1,'after: %9f'%ret2,', alpha=',alpha)
        if ret2>ret1: #如果当前轮迭代的目标函数值大于上一轮的函数值，则减小步长并重新计算梯度、重新更新参数
            if alpha < 0.0000001: 
                float = affine_trans(source, grid_points)# 得到Y和X对应关系
                print(grid_points)
                print('alpha too small')
                return float,loss
            alpha*=0.5
            grid_points = pred_grid_points.copy()
            print('update alpha...')
            continue
        loss.append(ret1)
        if np.abs(ret2-ret1)<e: #如果前后的变化比e小
            out_cnt+=1
            print('small enough!')
            loss.append(ret2)
            if out_cnt>=2: #如果连续2次目标函数值不变则认为达到最优解停止迭代
                print(grid_points)
                float = affine_trans(source, grid_points)# 得到Y和X对应关系
                return float,loss
        else:
            out_cnt=0
        pre_gradient = gradient.copy()
        gradient = cal_gradient(source, target, grid_points,gradient)#计算新的梯度
        if np.linalg.norm(gradient,2)>np.linalg.norm(pre_gradient,2): #如果新的梯度比原来的二范数大，则增加alpha步长
            print('add alpha')
            alpha*=2
        count+=1
        print('第 ',count,'次迭代:')
    return -1,loss


if __name__ == '__main__':
    PATH = os.path.abspath(__file__)
    PATH = os.path.abspath(os.path.dirname(PATH) + os.path.sep + ".")
    print(PATH)
    source = cv2.imread("float_1.jpg" ,cv2.IMREAD_GRAYSCALE)   #浮动图像
    target = cv2.imread("base_1.jpg" ,cv2.IMREAD_GRAYSCALE) #基准图像
    target = cv2.resize(target, source.shape[::-1])
    A = np.random.randn(2,2)*0.01+np.eye(2)
    b = np.random.randn(2,1)*5
    A=A.reshape((-1))#展平
    b=b.reshape((-1))
    grid_points = np.concatenate((A,b)) #展成一维向量
    print('init param', grid_points)
    # grid_points.reshape((-1))#变成一维向量

    t1=time.time()
    out,loss = optimization_gd_affine(source,target, grid_points)

    diff = np.abs(np.array(target,dtype='int64')-np.array(out,dtype='int64'))
    t2 =time.time()
    print('spend time:',t2-t1)
    plt.subplot(2, 2, 1), plt.axis("off")
    plt.imshow(source,cmap='gray'),plt.title('float')
    plt.subplot(2, 2, 2), plt.axis("off")
    plt.imshow(target,cmap='gray'),plt.title('baseline')
    plt.subplot(2, 2, 3), plt.axis("off")
    plt.imshow(out,cmap='gray'),plt.title('out')
    cv2.imwrite('affine_out_1.jpg', out) 
    plt.subplot(2, 2, 4), plt.axis("off")
    plt.imshow(diff,cmap='gray'),plt.title('out-base')
    plt.savefig('affine_result_1.jpg')

    plt.clf()
    X1=range(len(loss))
    plt.plot(X1,loss,'g--')
    plt.title('SSD similarity')
    plt.xlabel('iteration')
    plt.ylabel('SSD')
    plt.savefig('affine_SSD_1.jpg')