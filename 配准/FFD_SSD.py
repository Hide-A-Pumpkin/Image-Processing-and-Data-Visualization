import numpy as np
import cv2
from matplotlib import pyplot as plt
from collections import defaultdict
import numba as nb
BPLINE_BOARD_SIZE=3
import os
import time

#FFD代码:
@nb.jit()
def Bspline_Ffd_kernel(source, out, row_block_num, col_block_num, grid_points):
    '''
    B样条FFD变换函数
    '''
    row, col = source.shape
    delta_x = col*1.0/col_block_num #列的块长度
    delta_y = row*1.0/row_block_num#行的块长度
    grid_rows = row_block_num + BPLINE_BOARD_SIZE
    grid_cols = col_block_num + BPLINE_BOARD_SIZE #加padding目的是防止越界
    grid_size = grid_rows*grid_cols #网格大小
    for y in range(row): 
      for x in range(col):
        x_block = x / delta_x #在第几块
        y_block = y / delta_y #在第几块
        
        j = int(x_block)#取整
        i = int(y_block)#取整
        u = x_block - j#该点距离块边界的距离
        v = y_block - i

        #B样条基函数
        pX=np.zeros((4))
        pY=np.zeros((4))
        u2 = u*u
        u3 = u2*u
        pX[0] = (1 - u3 + 3 * u2 - 3 * u) / 6.0
        pX[1] = (4 + 3 * u3 - 6 * u2) / 6.0
        pX[2] = (1 - 3 * u3 + 3 * u2 + 3 * u) / 6.0
        pX[3] = u3 / 6.0
        v2 = v*v #v平方
        v3 = v2*v#v三次方
        pY[0] = (1 - v3 + 3 * v2 - 3 * v) / 6.0
        pY[1] = (4 + 3 * v3 - 6 * v2) / 6.0
        pY[2] = (1 - 3 * v3 + 3 * v2 + 3 * v) / 6.0
        pY[3] = v3 / 6.0

        Tx = x
        Ty = y
        for m in range(4):
          for n in range(4):
            control_point_x = j + n
            control_point_y = i + m #控制点的位置
            temp = pY[m] * pX[n] #B样条
            # print(control_point_y*grid_cols+control_point_x)
            Tx += temp*grid_points[int(control_point_y*grid_cols+control_point_x)]#累加x
            Ty += temp*grid_points[int(control_point_y*grid_cols+control_point_x+grid_size)]
        x1 = int(Tx)
        y1 = int(Ty)
        
        if (x1 < 1 or x1 >= col-1 or y1 < 1 or y1 >= row-1):
          out[y,x] = 0    #直接设为黑色
        else:
          x2 = x1 + 1   
          y2 = y1 + 1
          # print(y1,col,x1)
          #双线性插值
          gray = (x2 - Tx)*(y2 - Ty)*source[y1,x1] - \
              (x1 - Tx)*(y2 - Ty)*source[y1,x2] - (x2 - Tx)*(y1 - Ty)*source[y2,x1] + \
                              (x1 - Tx)*(y1 - Ty)*source[y2,x2]
          out[y,x] = gray
    return out

# 随机生成2*(row_block_num+3)*(col_block_num+3)个范围在min~max之间的随机数作为控制参数：
def init_param(source, row_block_num, col_block_num, min_x, max_x):
  grid_rows = row_block_num + BPLINE_BOARD_SIZE
  grid_cols = col_block_num + BPLINE_BOARD_SIZE#网格的行和列
  grid_size = grid_rows*grid_cols
  grid_points = (max_x-min_x) * np.random.random_sample((2*grid_size)) +min_x #生成一维向量，保存x和y，用随机数初始化
  return grid_points

@nb.jit()
def cal_ncc(source, target, row, col):
    '''
    归一化互相关系数，用分块计算的方法
    input: source, target, row, col
    output: ncc
    '''
    row_size = int(source.shape[0]/row)
    col_size = int(source.shape[1]/col)#行列大小
    ncc=0
    for i in range(row):
        i_begin = i*row_size #分块起始位置
        for j in range(col):
            sum1=0
            sum2=0
            sum3=0
            j_begin = j*col_size
            for t1 in range(i_begin,np.min((source.shape[0],i_begin+row_size))):
                for t2 in range(j_begin,np.min((source.shape[1],j_begin+col_size))):
                    sum1+=int(source[t1,t2])*int(target[t1,t2])#a*b,转成int防止数值溢出
                    sum2+=np.square(source[t1,t2],dtype='int64') #a**2，转成int64防止数值溢出
                    sum3+=np.square(target[t1,t2],dtype='int64') #b**2，转成int64防止数值溢出
            ncc+=np.sqrt(sum2*sum3)/(sum1+0.0000000001)#分块计算相似度，加0.001为了防止分母为0
    ncc/=(row*col)
    return ncc


#SSD相似性度量
@nb.jit()
def SSD_similarity(source,target):
    '''
    SSD相似性度量
    :param source， target 浮动图像和基准图像
    :output SSD score
    '''
    return ((source-target)**2).mean()


def get_fun_bpline(source, target, grid_points, row_block_num, col_block_num,mode):
    '''
    计算目标函数 包含三部分: transform+插值+计算互相关
    input: source, target, grid_points
    output: ncc value
    '''
    # U,U_target = np.array_split(grid_points, 2)
    # U = U.reshape((-1,2))
    # U_target = U_target.reshape((-1,2))#提取出两个控制点
    out = np.zeros_like(source)
    out = Bspline_Ffd_kernel(source, out,row_block_num, col_block_num, grid_points)# 得到Y和X对应关系
    if mode=='SSD':
        result = SSD_similarity(out,target)
    else:
        result = cal_ncc(out,target,5,5)#分s5*5的小块计算互相关
    return result



def cal_gradient(source,target,grid_points,gradient,row_block_num,col_block_num,mode):
    '''
    差分法计算梯度
    input: source, target,grid_points, old gradient
    output: new gradient update
    '''
    eps=1#超参数，可以改
    a1=get_fun_bpline(source,target,grid_points,row_block_num,col_block_num,mode)
    grid_p = grid_points.copy()
    for i in range(grid_points.shape[0]):
        grid_p[i]+=eps #修改grid的i位元素的大小
        a2 = get_fun_bpline(source, target, grid_p,row_block_num, col_block_num,mode) #每次都要做一次仿射+相似度估计
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



def optimization_gd_ffd(source, target, out, row_block_num, col_block_num,grid_points,mode='SSD'):
    '''
    最优化迭代的具体步骤
    超参数需要进一步调整
    :input 输入图像，行块个数，列块个数，特征点
    :output 输出图像，SSD下降曲线
    '''
    loss=[]
    max_iter=500#最多迭代次数
    e =0.0001#迭代精度
    count=0#迭代次数统计
    if mode=='SSD':
        alpha=0.1#初始alpha
    else:#ncc相似度
        alpha=1000
    gradient = np.zeros_like(grid_points,dtype='float')#初始化梯度
    gradient = cal_gradient(source, target, grid_points, gradient, row_block_num,col_block_num,mode)
    out_cnt=0
    while count<max_iter:
        pred_grid_points= grid_points.copy()
        grid_points = update_grid_points(grid_points,gradient,alpha)
        ret1 = get_fun_bpline(source, target, pred_grid_points,row_block_num,col_block_num,mode)#原本点的相关系数
        ret2 = get_fun_bpline(source, target, grid_points,row_block_num,col_block_num,mode)#更新后点的相关系数
        print('original:%9f'%ret1,'after: %9f'%ret2,', alpha=',alpha)
        if ret2>ret1: #如果当前轮迭代的目标函数值大于上一轮的函数值，则减小步长并重新计算梯度、重新更新参数
            if alpha < 0.000001: 
                float = Bspline_Ffd_kernel(source, out, row_block_num, col_block_num, grid_points)# 得到Y和X对应关系
                print('alpha too small')
                return float,loss
            alpha*=0.5
            grid_points = pred_grid_points.copy()
            print('update alpha...')
            continue
        loss.append(ret2)
        if np.abs(ret2-ret1)<e: #如果前后的变化比e小
            out_cnt+=1
            print('small enough!')
            if out_cnt>=2: #如果连续2次目标函数值不变则认为达到最优解停止迭代
                float = Bspline_Ffd_kernel(source, out, row_block_num, col_block_num, grid_points)# 得到Y和X对应关系
                return float,loss
        else:
            out_cnt=0
        pre_gradient = gradient.copy()
        gradient = cal_gradient(source, target, grid_points,gradient,row_block_num,col_block_num,mode)#计算新的梯度
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
    source = cv2.imread(PATH+"/float_3.jpg" ,cv2.IMREAD_GRAYSCALE)   #浮动图像
    target = cv2.imread(PATH+"/base_3.jpg" ,cv2.IMREAD_GRAYSCALE) #基准图像
    target = cv2.resize(target, source.shape[::-1])
    row_block_num = 10
    col_block_num = 10
    grid_points = init_param(source, row_block_num, col_block_num, -10, 10)
    '''
    如果row和col的大小是3，那么grid_points的长度为6*6*2=72,梯度的长度为72
    '''
    out = np.zeros_like(source,dtype='float')
    t1=time.time()
    out,loss = optimization_gd_ffd(source, target, out, row_block_num, col_block_num,grid_points,mode='SSD')

    diff = np.abs(np.array(target,dtype='int64')-np.array(out,dtype='int64'))

    plt.subplot(2, 2, 1), plt.axis("off")
    plt.imshow(source,cmap='gray'),plt.title('float')
    plt.subplot(2, 2, 2), plt.axis("off")
    plt.imshow(target,cmap='gray'),plt.title('baseline')
    plt.subplot(2, 2, 3), plt.axis("off")
    plt.imshow(out,cmap='gray'),plt.title('out')
    cv2.imwrite('out_1.jpg', out) 
    plt.subplot(2, 2, 4), plt.axis("off")
    plt.imshow(diff,cmap='gray'),plt.title('out-base')
    plt.savefig('result_1.jpg')

    plt.clf()
    X1=range(len(loss))
    plt.plot(X1,loss,'g--')
    plt.title('SSD similarity')
    plt.xlabel('iteration')
    plt.ylabel('SSD')
    plt.savefig('SSD_1.jpg')