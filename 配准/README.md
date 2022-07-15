# 图像处理期末报告

赵心怡 19307110452

刘思源 19307130018

[TOC]

## 1. 问题和项目介绍

### 1.1 任务叙述

完成图像配准任务，设计算法和界面，并作用于测试图像，实现该选题的全部流程。

1.任务包括GUI界面的实现，不需要复杂的功能，能够在界面上展示读取的图像，与处理或渲染后的结果即可

2.需要选择与本课程的主题相关的算法，可以参考合适的文献

### 1.2 本项目介绍

我们小组的开发环境为python3.9，需要安装的第三方包matplotlib, numpy, numba, simpleITK, PyQt5, opencv-python, tensorflow.

在本项目中，我们小组实现了基于梯度下降法的仿射变换和FFD变换配准，包含了两种变换方法和SSD,NCC两种相似度，我们额外探究了光流场理论的Demons算法和基于深度学习的图像配准算法。比较了不同算法之间的优劣，并设计了GUI页面和少量交互来展示我们的配准结果。

如需运行我们小组的代码，首先保证抬头的第三方包已经全部安装，将压缩包内的代码解压到同一文件夹中，运行代码

```shell
python newGUI.py
```

### 1.3 分工

赵心怡： 仿射配准和FFD变换配准，SSD相似度和cc相似度的比较，GUI界面，对应部分报告和视频

刘思源：demons和voxel模型配准和比较，对应部分报告

## 2. 数据和浮动图像生成

### 2.1 数据来源

心脏图像数据来自于Myops挑战赛的心脏图像 •https://zmiclab.github.io/zxh/0/myops20/ 我们选择了test20文件中myops_test_201_C0.nii文件的第1、3个图像，并用仿射变换与FFD变换对他们进行处理。

由于心脏图像不太直观，比较起来较为困难，我们也使用了lena图像和经典的圆方图来测试。

voxelMorph模型的训练数据集来自他们提供的 [脑部 MRI（2D 切片）数据集](https://surfer.nmr.mgh.harvard.edu/pub/data/voxelmorph/tutorial_data.tar.gz)，通过随机选择其中的图片为基准图像和浮动图像进行训练。

### 2.2 浮动图像生成

此部分的代码在generate_float.py中，可以直接运行。

为了测试我们的配准算法好坏，我们写了generate_float函数来生成浮动图像，包括了随机的仿射变换和随机的FFD变换，我们选择了心脏图片和lena的图片进行了变换，心脏图片的随机参数设置较大，浮动较明显，lena图像的参数较小，浮动不明显。

生成浮动图像的代码：

```python
def generate_float_img(seed=1234,idx=1):
    '''
    浮动图像生成，并对比浮动图像生成效果和前后差异。
    :input seed随机种子，idx图像的位置
    '''
    np.random.seed(seed)
    img_path = 'test20\myops_test_201_C0.nii.gz'
    itk_img = sitk.ReadImage(img_path)
    img = sitk.GetArrayFromImage(itk_img)
    target_slice=img[idx,:,:]#读取第i张图片
    height,width = target_slice.shape

    A = np.random.randn(2,2)*0.05+np.eye(2)
    b = np.random.randn(2,1)*0.5
    parameters_affine={'A':A,'b':b}#生成随机参数

    row_block_num=5
    col_block_num=5

    grid_points = init_param(row_block_num,col_block_num,-10,10)#FFD的随机参数

    #人工生成浮动图像，计算形变场
    out = affine_trans(target_slice, parameters_affine)#先做一次仿射
    out = Bspline_Ffd_kernel(out,row_block_num,col_block_num,grid_points)
```

初始化FFD参数的代码：

```python
def init_param(row_block_num, col_block_num, min_x, max_x):
    '''
    初始化FFD参数的代码
    :input 行块列块个数，随机数的最小和最大值
    :output 将行列块的特征点展平成一维向量
    '''
    grid_rows = row_block_num + BPLINE_BOARD_SIZE
    grid_cols = col_block_num + BPLINE_BOARD_SIZE#网格的行和列
    grid_size = grid_rows*grid_cols
    grid_points = (max_x-min_x) * np.random.random_sample((2*grid_size)) +min_x #生成一维向量，保存x和y，用随机数初始化
    return grid_points
```

我们输出了生成的图像并对他们的灰度值差异进行比较

首先是心脏图的灰度值差异对比

![image-20220604183224309.png](https://s2.loli.net/2022/06/05/FgyGwzjbUx6LC3Z.png)

lena图的灰度值差异对比：

![image-20220604183338230.png](https://s2.loli.net/2022/06/05/gVLYtpu31DAhTob.png)

lena图的灰度值分布较为均匀，但是由于图像的背景不是黑色，因此变换会让图像的周围产生黑色的锯齿。



## 3.基于梯度下降法的FFD和仿射变换配准

![未命名文件.png](https://s2.loli.net/2022/06/05/v85O6IGEuwecyrR.png)

首先我们回顾了课堂知识，并查阅了网上相关的文章，最后比较认同这一网址文章的梳理 [图像配准系列之基于FFD形变与梯度下降法的图像配准](https://mp.weixin.qq.com/s?__biz=Mzk0NjE2NDcxMw==&mid=2247484347&idx=1&sn=b8dc703688d70a0e327b7e597812fd9c&chksm=c30b053df47c8c2b5a8bc93c086bde6f59ccb8518f97174ac4dfd94afa2324e241779076ca76&scene=178&cur_album_id=1631097270580396035#rd)。

如图所示，我们分解其中的每一部分的代码逐一进行解释。

### 3.1 梯度下降法原理

梯度下降法是一个一阶最优化算法，迭代公式可以记为
$$
b=a-\gamma\bigtriangledown F(a)
$$

```python
def optimization_gd_affine(source, target,grid_points):
    '''
    最优化迭代的具体步骤
    超参数需要进一步调整
    :input 输入图像，特征点
    :output 输出图像，SSD下降曲线
    '''
    loss=[]
    max_iter=500#最多迭代次数
    e =0.0000001#迭代精度
    count=0#迭代次数统计
    alpha = 0.0005#初始alpha
    gradient = np.zeros_like(grid_points,dtype='float')#初始化梯度
    gradient = cal_gradient(source, target, grid_points, gradient) #梯度计算
    out_cnt=0
    while count<max_iter:
        # print(gradient)
        pred_grid_points= grid_points.copy()
        grid_points = update_grid_points(grid_points,gradient,alpha)
        ret1 = get_fun_bpline(source, target, pred_grid_points)#原本点的相关系数
        ret2 = get_fun_bpline(source, target, grid_points)#更新后点的相关系数
        print('original:%9f'%ret1,'after: %9f'%ret2,', alpha=',alpha)
        if ret2>ret1: #如果当前轮迭代的目标函数值大于上一轮的函数值，则减小步长并重新计算梯度、重新更新参数
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
        # pre_gradient = gradient.copy()
        gradient = cal_gradient(source, target, grid_points,gradient)#计算新的梯度
        if np.linalg.norm(gradient,2)>np.linalg.norm(pre_gradient,2): #如果新的梯度比原来的二范数大，则增加alpha步长
            print('add alpha')
            alpha*=2
        count+=1
        print('第 ',count,'次迭代:')
    return -1,loss
```

需要说明的是，alpha的值会随着梯度下降而更新，如果前后的变化比e小，说明alpha太大需要缩小，如果新的梯度的范数比原来的大，说明alpha太小需要放大。

差分法估计梯度的计算公式，我们选用：
$$
\frac{\partial f(x)}{\partial x} \approx \frac{f(x+h)-f(x)}{h}
$$

```python
def cal_gradient(source,target,grid_points,gradient):
    '''
    计算梯度
    input: source, target,grid_points（特征点）, old gradient
    output: new gradient update
    '''
    eps=0.0005#超参数，可以改
    a1=get_fun_bpline(source,target,grid_points)
    grid_p = grid_points.copy()
    for i in range(grid_points.shape[0]):
        if i>=4:
            eps=0.1
        grid_p[i]+=eps #修改grid的i位元素的大小
        a2 = get_fun_bpline(source, target, grid_p) #每次都要做一次仿射+相似度估计
        grid_p[i]-=eps #再把它改回来
        gradient[i] = (a2-a1)/eps#梯度
    return gradient
```

梯度更新代码：

```python
def update_grid_points(grid_points, gradient, alpha):
    '''
    根据梯度下降的逻辑更新grid
    '''
    for i in range(grid_points.shape[0]):
        grid_points[i]=grid_points[i]-gradient[i]*alpha
    return grid_points
```

### 3.2 FFD变换和仿射变换原理

基于B样条的FFD变换属于网格型的非刚性形变模型，它按照一定的间距在图像上分布一系列的网格点，使用网格点的位置来计算每个像素点的坐标偏移，最后根据坐标偏移对图像进行像素重采样，实现其非刚性形变。FFD的特征是擅长处理局部扭曲的点。

![image-20220603204236291.png](https://s2.loli.net/2022/06/05/kQaxG9K7Y5Xcm1F.png)

**FFD变换**代码如下：

```python
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
            Tx += temp*grid_points[int(control_point_y*grid_cols+control_point_x)]#累加x
            Ty += temp*grid_points[int(control_point_y*grid_cols+control_point_x+grid_size)]
        x1 = int(Tx)
        y1 = int(Ty)
        
        if (x1 < 1 or x1 >= col-1 or y1 < 1 or y1 >= row-1):
          out[y,x] = 0    #直接设为黑色
        else:
          x2 = x1 + 1   
          y2 = y1 + 1
          #双线性插值
          gray = (x2 - Tx)*(y2 - Ty)*source[y1,x1] - \
              (x1 - Tx)*(y2 - Ty)*source[y1,x2] - (x2 - Tx)*(y1 - Ty)*source[y2,x1] + \
                              (x1 - Tx)*(y1 - Ty)*source[y2,x2]
          out[y,x] = gray
    return out
```

**仿射变换**

仿射变换适用于处理图像的整体变动，二维空间有六个自由度我们可以用$y=Ax+b$来表示，为了方便梯度运算，我们将A和b展开合并成一个长度为6的一维向量，这样得到的梯度也是一维，方便更新和计算范数。

```python
def affine_trans(source, para):
    '''局部仿射函数,寻找X=T^{-1} (Y)的函数关系 即position_map'''

    height, width= source.shape
    position_map = defaultdict(lambda: np.eye(2))  # 构造position_map映射字典，初始为0向量
    # print(para.shape)
    G = para[:4].reshape((2,-1))
    b = para[4:].reshape((-1))
    # 得到Gi
    for x in range(height):
        for y in range(width): # 遍历每个点
            Y = (x,y)
            position_map[Y] = np.dot(G, np.array([x,y]))   # 利用w对Gi加权
            position_map[Y] = position_map[Y]+b     # 把向量最后的1去除
    newImg = backward_trans(source,position_map)
    return newImg
```

用反向变换+双线性插值的方法生成新图像，反向变换的介绍在hw6中已经作出。

代码如下：

```python
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

    return newImg 
```

### 3.3 相似度计算

常用的图像相似度衡量指标有峰值信噪比（PSNR）、结构相似度（SSIM）、归一化互相关（NCC）、归一化互信息（NMI）、均方误差（MSE）等。

归一化互相关系数计算较为简便，我们尝试了SSD相似度和归一化互相关系数大小，并比较了两种不同互相关系数的效果。

SSD相似度全称误差平方和算法，计算公式如下
$$
\mathrm{SSD}=\sum_{x_{A} \in \Omega_{A, B}^{T}}\left|A\left(x_{A}\right)-B^{T}\left(x_{A}\right)\right|^{2}
$$
代码如下：

```python
def SSD_similarity(source,target):
    '''
    SSD相似性度量
    :input source， target 浮动图像和基准图像
    :output SSD score
    '''
    return ((source-target)**2).mean()
```

NCC是归一化互相关系数，计算公式如下：
$$
\operatorname{NCC}(A, B)=\frac{\sum_{i=0}^{m-1} \sum_{j=0}^{n-1} A(i, j) * B(i, j)}{\sqrt{\sum_{i=0}^{m-1} \sum_{j=0}^{n-1} A(i, j)^{2}} * \sqrt{\sum_{i=0}^{m-1} \sum_{j=0}^{n-1} B(i, j)^{2}}}
$$
归一化互相关NCC越大，说明图像A与图像B越相似，反之则两者差异越大。由于梯度下降法为求解目标函数的最小值，所以我们需要把以上函数取个倒数，使得NCC越小。

很多时候，图像之间存在很多局部的形变差异，所以通过分块来求解的NCC'更能表现两图的相似度。我们在以上基础上，在对图像A与图像B进行相同的分块，计算两图中对应位置块的NCC'，最后再取所有块的NCC'的平均值作为整张图像的NCC'。假设把图像的高平均分为r块，宽平均分为c块，那么最终的目标函数F的表达式如下
$$
F(A, F F D(B, X))=\frac{1}{r^{*} c} \sum_{i=0}^{r-1} \sum_{j=0}^{c-1} N C C^{-1}\left(A b l o c k(i, j), \operatorname{Bblock}_{F F D}(i, j)\right)
$$
NCC相似度计算的代码：

```python
def cal_ncc(source, target, row, col):
    '''
    归一化互相关系数，用分块计算的方法
    :input: source, target, row, col图像，行与列的块个数
    :output: ncc
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
```



### 3.4 分析和讨论

本部分分为三个板块，首先进行FFD和仿射变换的比较，然后是SSD相似度和NCC相似度的效果比较和分析，最后对于实验中出现的问题进行解释。

#### 3.4.1 FFD和仿射的比较

我们对心脏图的基准和浮动图像分别进行FFD和仿射变换，以SSD为相似度，设置Num_blocks为10，alpha初始值为4000，处理结果如下：

<img src="https://s2.loli.net/2022/06/05/GXWjUKMcm4boSNH.png" alt="image-20220604233738609.png" style="zoom:80%;" />

SSD相似度变化趋势如图

<img src="https://s2.loli.net/2022/06/05/6fPo28QJ4qAgsKj.png" alt="image-20220603182946853.png" style="zoom:80%;" />

然后进行仿射变换，选择初始alpha为0.0005，同样使用SSD相似度进行衡量，结果如下：

<img src="https://s2.loli.net/2022/06/05/zsGoSnPhxVHw8Zt.png" alt="image-20220604234846412.png" style="zoom:80%;" />



<img src="https://s2.loli.net/2022/06/05/bxlJhrIcXdMv7Qq.png" alt="image-20220605135057468.png" style="zoom:80%;" />

对比迭代次数和配准结果可以看见，两种配准方法都可以达到一个较为良好的结果，但是FFD算法需要的时间比仿射更久，二者的时间比较，可以看到在同样的迭代次数下，FFD变换将比仿射花费两倍以上的时间，如果增加Grid块的个数，所需要的时间还会更久：

|      | FFD     | Affine |
| ---- | ------- | ------ |
| 时间 | 10563 s | 3889 s |

仿射变换的缺点是输出结果不稳定，需要较好的初始值，当初始变换设置不好时，仿射变换有一定概率将物体移出边界范围，输出全黑的图像，A矩阵最终变成几十几百的量级。因为此时的算法认为无论是否变换都无法使得图片的相似度比黑色图像相似度高，可以类比人工智能的狼放弃捉羊直接撞死在石头上的情节，朝着另一个方向尝试快速收敛。

造成收敛方向错误的一个原因是SSD相似度变换只考虑对应点的图像性质而不考虑与周围点的关系。从SSD相似度的公式中可以看出它只单纯对每个像素点的灰度值进行减法，如果浮动图像和基准图像的差别过大会导致无法找到正确的收敛方向而一头撞死的问题。这也让我们试着改变度量相似度的指标来进行测试。

#### 3.4.2 SSD和NCC的比较

我们在lena图上比较SSD和NCC两种相似度的结果。lena图是只经过了FFD变换生成浮动的图像，因此变换的结果更接近初始图像。

SSD变换的结果

<img src="https://s2.loli.net/2022/06/05/GbmpCfLDSZnVsIE.png" alt="image-20220605130619871.png" style="zoom:80%;" />

<img src="https://s2.loli.net/2022/06/05/WHzdM25ORTtBbSh.png" alt="image-20220604181528131.png" style="zoom:80%;" />

NCC变换的结果

<img src="https://s2.loli.net/2022/06/05/uKxFalMwjUERmyv.png" alt="image-20220605130820207.png" style="zoom:80%;" />

<img src="https://s2.loli.net/2022/06/05/TlZ5Uu2Dd3Jbgwx.png" alt="image-20220604193026321.png" style="zoom:80%;" />

对比两种方法看见最终的lena图的灰度值差类似，但是NCC相似度的迭代步数少于20步。

从计算时间的角度，NCC的计算量较大，会花费更多时间。

|      | SSD    | NCC    |
| ---- | ------ | ------ |
| time | 3068 s | 2554 s |

从另一方面，NCC似乎可以更直观的看出图像的相似度的趋近，因为完全相同的图片的相似度为1，离1越近说明图片的相似度越高。但是从SSD相似度值上看似乎离0还有一定距离。

从以上观察中得出，归一化互相关系数计算相对复杂，但其具有的良好凹凸特性利于求解最优参数。而SSD的优点是快速，但它有一定概率收敛到错误的方向上，需要提供较好的初始参数。



## 4. 基于光流场理论的 Demons 算法

### 4.1 Demons 算法概念

非刚性图像配准的方法可以分为三类：基于特征的方法，基于灰度的方法及基于灰度与特征结合的方法。

* 基于特征的非刚性图像配准往往需要预先提取图像的点线面特征，实现配准，运算速度快。但特征点的选择往往会影响到最后的配准精度。获取特征点的算法有SIFT算法，最近点迭代算法ICP等。特征面则可以通过类似图像分割，对组织边缘进行检测。
* 基于灰度的非刚性图像配准不需要预先提取，而是利用灰度信息，建立搜索空间，搜索策略和相似度度量，找到最优的变换场，使两个图像灰度相似度最大。该类型可以分为两种：基于基函数拟合的方法和基于物理模型的方法。
* 基于基函数拟合的方法：利用基函数来拟合或插值变形场是基于基函数方法的基本思路，常用的包含样条函数和多项式函数。多项式的阶数会严重影响配准算法性能。样条函数包括B样条，薄板样条等。
* 基于物理模型的方法：Thirion受到MaxWell的热力学分子扩散模型的启发，将分子扩散模型应用在图像配准之中。

在本次项目中，我们手动实现了Demons的多种算法版本，并比较了这些版本的效果差异。

#### 4.1.1 原始Demons

Demons的概念最早源于Maxwell的热力学分子扩散理论，Demons作为一种选择器，可以将不同类型的两个分子区分开。如图所示，一开始左右两个区域通过Demons薄膜隔开，两边各有一些分子。Demons只允许黑色分子向左扩散，白色分子向右扩散。经过一段时间分子运动后，左边区域只有黑色分子，右边区域只有白色分子。

![demons](https://s2.loli.net/2022/06/05/m9oGHlCsBDXFR1t.png)

原始的Demons算法即将这一分子扩散模型运用在图像配准中。假设参考图像S上存在选择器 Demons，待配准图像M的像素可被标记为图像外点或内点，被驱动力场推出目标区域或者推入目标区域，直到达到平衡。

![diffusing_models](https://s2.loli.net/2022/06/05/gwer35DYOukXvUN.png)

对于图像中任一坐标p，s(p),m(p)代表图像在该坐标处的灰度值，如果s(p)>m(p),则该点为外点，受到向内的驱动力，受力方向与参考图像的梯度方向相同；反之为内点，受到向外的驱动力，受力方向与参考图像的梯度方向相反。

![deformable_model](https://s2.loli.net/2022/06/05/fvdZ3WQSxV2oCiI.png)


驱动力的计算基于光流场理论的，假设参考图像与浮动图像为某图像连续运动过程中的两帧。图像配准就是计算出这两帧之间的驱动力向量U。驱动力计算公式为：
$$
U=\frac{(m-f)\nabla f}{\lVert \nabla f \rVert^2 +(m-f)^2}
$$

原始的Demons配准流程如下：

1. 输入参考图像和浮动图像，设置迭代次数N和误差精度
2. 根据上式计算待配准图像的形变向量场
3. 将形变向量场通过高斯滤波器计算形变向量场，得到光滑的形变向量场
4. 依据形变向量场对待配准图像进行图像配准。
5. 判断迭代次数是否达到N次，或者待配准图像的误差精度是否满足。满足则终止迭代，否则继续。

算法流程图如下：
![demons_process](https://s2.loli.net/2022/06/05/hSMY1ak2dyUAb6j.jpg)

因为原始Demons可能会出现分母过小导致形变过大的问题，一般会在分母的$(m-f)^2$处乘以$\alpha ^2$来控制波动。本次项目的实现也是使用加了Alpha的版本，这里命名为Alpha Demons。

#### 4.1.2 Active Demons

由于Demons算法单纯的依靠原图像的梯度信息来计算，所以原始Demons往往因变形驱动力不足，在配准过程中具有收敛速度慢和配准效果差的缺点。当原图像
中的梯度信息较小时，上式的驱动力会接近零，待配准图像的形变方向将不能确定，导致图像配准结果不准确。因此Active Demons在原始图像上进行了改进，在原始Demons基础上加上了反向驱动力，符合物理模型两个物体间的相互作用力。Wang 等人根据实验结果验证，该算法很好的解决了 Demons 算法不能解决
相对较大形变的图像问题，在图像配准速度和精度都有了一定的提高。且算法的配准速度取决于Alpha的选取。Alpha大时，收敛速度较慢，且适合较小形变区域的配准，针对较大形变区域图像配准需要很长的配准时间。

$$
U=\frac{(m-f)\nabla f}{\lVert \nabla f \rVert^2 +\alpha^2(m-f)^2}+\frac{(m-f)\nabla m}{\lVert \nabla m \rVert^2 +\alpha^2(m-f)^2}
$$

#### 4.1.3 Inertial demons

Inertial demons函数在Active demons算法的基础做了改进,把上一层迭代计算得到的偏移量加入到当前层迭代的偏移量计算当中，使系统拥有惯性防止突然的变动，提升了收敛速度和配准精度。驱动力d迭代公式如下：
$$
U^n=U^{n-1}*momentum+U^n 
$$

#### 4.1.4 Symmetric Demons

Rogel 等人在原始的 Demons 算法基础上提出了 Symmetric Demons 算法，相比于Active Demons，把浮动图像和参考图像梯度平均化，同时算法中的参考图像和浮动图像的全局空间坐标可以相互重叠。

$$
U=\frac{(m-f)\nabla J }{\lVert \nabla J \rVert^2 +\alpha^2(m-f)^2},~J=\frac{1}{2}(\nabla f+\nabla m)
$$



#### 4.1.5 Diffeomorphic log-demons

原始Demons及其驱动力变种算法作为基于光流理论发展起来的配准方法，与其它光流算法一样，具有小运动的约束条件。如果参考图像和浮动图像的差别很大，也即不满足小运动条件，那么demons算法的配准效果会很差。而且Demons算法容易引起配准结果图的结构信息（纹理、线，边界）等容易发生重叠。这三种方法形变向量场是不可逆的，因此在图像配准过程中的拓扑结果容易遭受破坏。

因此Vercauteren等人提出将demons算法计算得到的位移场转换为微分同胚映射，从而对大运动或大形变都具有较理想的配准效果。

同胚映射通常用于形容拓扑空间的映射关系，如果两张图像的所有像素点都满足一一对应关系，那么这个映射系就是同胚映射。如果两个原始连续的像素点映射后还是连续的，那么映射f与逆映射f-1都是光滑的映射，这种映射也叫微分同胚映射。

 Diffeomorphic log-demons （微分同胚）模型中的能量函数表示为，
$$
E(U,U_c)=\frac{1}{\lambda_i}\lVert f-m \circ exp(U_c)\rVert_{L_2}^2+\frac{1}{\lambda_x}\lVert log( exp(-U)\circ exp(U_c))\rVert_{L_2}^2+\frac{\delta_i}{\delta}\lVert \nabla U \rVert_{L_2}^2
$$

其中$E(U,U_c)$为图像配准的能量函数,$\lambda_i$表示图像中出现的噪声,$\lambda_x$表示迭代运算中形变向量场更新的强度,$\circ$表示卷积运算，$M\circ exp(U_c)$表示待配准图像进行指数映射的变换.$\delta_i$是对图像噪声的一种全局估计,$\delta_x$则表示对形变向量场的一种约束关系。

在实现上，微分同胚算法引入了BakerCampbell-Hausdorff公式，假设$\delta U$的值很小，即可使用BCH的近似值计算：
$$
Z(U,\delta U)=U+\delta U+\frac{1}{2}[U,\delta U]+\frac{1}{12}[U,[U,\delta U]]+O(\lVert \delta U \rVert)^2
$$

算法整体流程如下,每轮迭代时：

1. 初始化形变场 $U_0$
2. 迭代循环，通过$U^{n-1}$计算形变速度场$\delta U^{n}$
3. 对更新的变形场进行fluid高斯滤波的正则化效果
4. 根据BCH公式对形变速度场进行更新，得到$U_c^n$
5. 再次进行diffusion高斯滤波
6. 对待配准图像进行驱动
7. 判断能量函数是否达到最小，达到最小则输出配准结果，否则继续迭代


使用Active demons算法类配准的时候，每轮迭代都更新了最优位移场。实际上，并不是每轮迭代都是朝着配准效果更好的方向前进的，有的迭代配准效果反而更差，所以每轮迭代都更新最优位移场是有问题的。于是使用相似度来判断是否要更新最优位移场的方法被提了出来


### 4.2 Demons 算法实现

在这个章节中，我们会具体介绍各Demons算法的代码实现。除加速版本的Diffeomorphic log_Demons，其他代码均为手写实现，仅参照了原始论文公式或C语言版本。加速版本的Diffeomorphic log_Demons在理解源代码的基础上，将原始版的手写代码进行改进，达到了相近的加速效果。

#### 4.2.1 Alpha Demons 

Alpha demons代码具体见demons.py中的函数Alpha_demons。函数输入为参考图像:S,浮动图像:M,归一化因子: alpha,训练轮数:turns,相似函数:similarity,record：是否记录NCC；函数输出为配准后图像（uint8）格式。

```python
def Alpha_demons( S,  M,  alpha, turns,similarity='NCC',record=False):
    ....
```

函数流程会通过函数normal（），迭代计算每轮的驱动力，随后通过函数 movepixels_2d（）让浮动图像基于变换场进行变换。函数使用NCC计算每轮迭代的准确度。

normal()函数的输入为：参考图像:S,浮动图像:M;参考图像在x，y方向的梯度Sx,Sy;扩散速度系数α；函数输出为：x,y方向变形场。函数会先计算浮动图像和参考图像的差值和参考图像在x，y方向上的梯度，随后对于每个点，计算该点的变形场，再对求解的变形场进行高斯平滑。

```python
def normal(S,  M,  alpha):
    ....
    Sx, Sy=get_mat_gradient(S) 
    ....
    for i in range(rows):
        for j in range(cols):
            ....
            ux = Sx[i,j] / sdenominator 
            Ux[i,j] = (-diff[i,j]*ux)
    
            uy = Sy[i,j] / sdenominator 
            Uy[i,j] = (-diff[i,j]*uy)
    ....
    Ux=cv2.GaussianBlur(Ux,(ksize, ksize), sigma, sigma) 
    Uy=cv2.GaussianBlur(Uy, (ksize, ksize), sigma, sigma)
    
```

movepixels_2d（）函数则会对待变换图每个坐标点，计算该坐标点基于变形场Ux，Uy后的x，y坐标，通过双三样条插值求出变形后的图像。

#### 4.2.2 Active Demons 

Active demons代码具体见demons.py中的函数Active_demons。

```python
def Active_demons( S,  M,  alpha, turns,similarity='NCC',record=False):
    ....
```

相比Alpha Demons，该函数将计算变形场的normal（）函数替换为了active（）函数。active（）相比normal（）函数替换了变形场的计算模块，引入了相互作用力。

```python
ux = Sx[i,j] / sdenominator + Mx[i,j] / mdenominator
Ux[i,j] = (-diff[i,j]*ux)

uy = Sy[i,j] / sdenominator + My[i,j] / mdenominator
Uy[i,j] = (-diff[i,j]*uy)
```

#### 4.2.3 Inertial Demons

Inertial demons代码具体见demons.py中的函数Inertial_demons。

```python
def Inertial_demons( S,  M,  alpha, turns,similarity='NCC',record=False):
    ....
```

相比Active Demons,Inertial Demons 更改了迭代模块，每一轮的驱动力为当轮驱动力+上一轮驱动力乘以动量。

```python
....
for i in range(rows):
    for j in range(cols):
    ....
    Ux,Uy = Ux+Ux_last*momentum , Uy+Uy_last*momentum 
    ....
```

#### 4.2.4 Symmetric Demons

Symmetric demons代码具体见demons.py中的函数Symmetric_demons。  

```python
def Symmetric_demons( S,  M,  alpha, turns,similarity='NCC',record=False):
    ....
```

相比Active Demons,Symmetric Demons 更改了驱动力计算模块，将原始的active（）函数更改为了symmetric（）函数。symmetric（）函数在计算每个点的形变场时，将两张图片间的双向吸引力进行平均。

```python
....
for i in range(rows):
    for j in range(cols):
    ....
    denominator = pow(Sx[i,j]+Mx[i,j], 2) + pow(Sy[i,j]+My[i,j], 2)+ 4*pow(alpha, 2)*pow(diff[i,j], 2)
    ....
    ux = (Sx[i,j]+Mx[i,j]) / denominator
    Ux[i,j] = (-2*diff[i,j]*ux)
```

#### 4.2.5 Diffeomorphic log_Demons

Diffeomorphic demons代码具体见demons.py中的函数Diffeomorphic_demons。  

```python
def Diffeomorphic_demons( S,  M,  alpha,  sigma_fluid,sigma_diffusion,turns,similarity='NCC',record=False):
    ....
```

相比之前的Demons算法，这个版本多了两个输入参数，分别是sigma_fluid和sigma_diffusion。这两个参数是用于每轮驱动力计算后的重采样高斯平滑卷积。在每轮迭代时，函数会先基于active函数计算驱动力，随后使用img_gaussian（）函数对驱动力进行高斯平滑卷积，复合至原驱动力上。随后，函数会继续做一次高斯平滑卷积，再通过微分同胚映射获得更新后的变换场。

注意，在这个版本中，我们特别加上了最佳位移场判断，这是因为每轮迭代时不一定超着更好的方向变换，有时或许更坏，这在后续的实验结果中也有所体现。因此，在这个版本中，我们尝试了在每轮迭代时判断NCC相似度，只保存所有轮数中变换相似度最高的形变场。

```python
for i in range(turns):

    #计算这一轮迭代的驱动力
    Ux, Uy=active(S, M_tmp, alpha)

    #使用Ux、Uy分别对自身进行像素重采样的操作得到Ux'和Uy'
    Ux=img_gaussian(Ux,sigma_fluid)
    Uy=img_gaussian(Uy,sigma_fluid)

    #计算Ux+Ux'和Uy+Uy'的运算得到复合运算
    Vx = Vx + Ux
    Vy = Vy + Uy

    Vx=img_gaussian(Vx,sigma_diffusion)
    Vy=img_gaussian(Vy,sigma_diffusion)

    #微分同胚映射
    Sx, Sy=exp_field(Vx, Vy )

    #浮动图像基于变换场进行变换
    M_tmp=movepixels_2d(M, Sx, Sy,cv2.INTER_LINEAR) 

    if similarity=='NCC':
      e=NCC(S,M_tmp)

    #如果相似度提高，则更新最佳位移场
    if e > e_max:  
      e_max = e
      Sx_min = Sx.copy()
      Sy_min = Sy.copy()

```

微分同胚映射函数exp_field()将输入的位移场转化为近似微分同胚映射。函数会先计算位移场的平方和矩阵，找到矩阵中的最大元素，根据公式计算映射矩阵，再对映射矩阵做n次复合运算。

```python
def exp_field(Vx, Vy):
    #计算位移场Ux、Uy的平方和矩阵
  NormV2 = cv2.multiply(Vx,Vx) + cv2.multiply(Vy,Vy)
  
  #求矩阵NormV2的最大值max
  max=np.max(NormV2)

  #根据公式 n=max(ceil(log2（sqrt(max) / 0.5）),0) 计算
  n = np.ceil(math.log2(math.sqrt(max) / 0.5))
  n = int(n) if n>0.0 else 0
  a = pow(2.0, -n)

  #计算矩阵Vx，Vy
  Vx = Vx * a
  Vy = Vy * a
 
  #对矩阵作n次复合运算，最后结果就是近似的微分同胚映射
  for i in range(n):
    Vx,Vy=composite(Vx,Vy)
  return Vx,Vy
```

#### 4.2.6 Accelearted Diffeomorphic log_Demons

传统的Demons算法最大的问题在于迭代次数久，每轮开销大。这让各个版本虽然在配准精度上逐步提高，但代价便是开销逐渐增加。为解决这个问题，我对算法实现做了诸多尝试，如将点循环改为矩阵运算，但效果都不尽如人意。在翻阅了一些github开源项目后，Deep Diffeomorphic Demons项目中对于微分同胚算法的加速吸引了我。因为期末Project要求必须手动实现，因此我们在充分理解算法加速方法中，用这一方法改写我们原始版算法，取得了接近两个数量级的提升。具体代码详见accelerate_diffeomorphic_demons.py文件。

不同于numpy+OpenCV实现，代码使用了tensorflow进行加速，所有能用tensorflow进行的矩阵计算都尽量使用，来保证算法底层实现时的并行优化。在计算矩阵梯度时,代码通过pad（）函数来对矩阵进行卷积。

```python
def get_mat_gradient(S):

    Tx = tf.pad((S[2:,:]-S[:-2,:])/2,paddings=([[1,1],[0,0]]))
    Ty = tf.pad((S[:,2:]-S[:,:-2])/2,paddings=([[0,0],[1,1]]))
    return tf.stack([Tx, Ty], axis=0)
```

在做高斯滤波时，tensorflow_addons类中的图像高斯滤波函数速度远快于Open CV中的实现。

```python
tfa.image.gaussian_filter2d(U,padding="SYMMETRIC",sigma=fluid_sigma)
```

在进行每轮的变形场计算时，使用数值计算方法而非循环遍历每个点也能更快加速计算。

```python
Ux,Uy = -((diff_intensity*J)/(tf.norm(J,axis=0)**2+diff_intensity**2+1e-6)) 
```

等等代码上的优化让算法迭代效率更高，2000轮迭代时间也仅需2-3分钟。



### 4.3 实验结果

#### 4.3.1 Lena图像

我们项目对局部大形变的Lena图和经典的圆方图进行了实验。左图为参考图片，右图为浮动图片。
![](https://s2.loli.net/2022/06/05/N2gQsmGXW7ZKL9r.jpg)

下面三张图从左到右分别为Alpha_demons,Active Demons和Symmetric Demons的结果。可以看到随着模型的改进，配准效果逐渐变好。比如Active相比Alpha边缘处黑边更平整，且Lena鼻子更直。Symmetric相比Active帽子更加平整。
![16544106628978.jpg](https://s2.loli.net/2022/06/05/sMZkLFzfOWh7Vpg.jpg)

下面两张图从左到右分别为Inertial Demons和 Diffeomorphic log_Demons。可见微分同胚的效果最好，对人脸的复原效果最自然，帽子也很光滑，但对边缘处的处理却不好，出现了些许黑丝的污染。
![16544107112424.jpg](https://s2.loli.net/2022/06/05/KNvUSzXV6JYQr5j.jpg)


下面三张图从左到右分别为Alpha_demons,Active Demons和Symmetric Demons配准后图和配准图的差别，可见Symmetric Demons白色光点更小，也即更相似。
![16544106876323.jpg](https://s2.loli.net/2022/06/05/wkFNgEz5IpfHuGL.jpg)

下面两张图从左到右分别为Inertial Demons和 Diffeomorphic log_Demons配准后图和配准图的差别。可见虽然中间区域光电更少，但是边缘处多了些白色斑点。
![16544107273813.jpg](https://s2.loli.net/2022/06/05/otOxzk57B9AC3ec.jpg)

而从这些模型配准的NCC变化来看，其实100轮时已经达到了较好的效果，如果继续迭代反而会向更糟糕的方向迭代。其中微分同胚的变化在运用了迭代限制，即只会向更好的方向迭代，效果虽然随着迭代轮粗逐步变好，但后续效果增加有限。比较异常的结果是Inertial方法并未收敛，在排查结果时发现，高斯滤波的核大小会影响到最终的收敛结果，在这个实验中所有的高斯滤波核固定为127，无法促进Inertial收敛。

![NCC_demons.png](https://s2.loli.net/2022/06/05/e3CgyJti8fxlK5s.png)


而当高斯滤波核固定为59后，可以看到Inertial效果逐步变好，达到了之前Symmetric的同等水平。针对之前Alpha收敛值过低的问题，我们尝试将每轮计算的变形场乘10，可见收敛效果相比之前好了很多，但是25以后反而会变差。我们分析原因可知，这可能是源于乘10后没步驱动力太大，无法有效向最优值收敛。
![NCC_loss.png](https://s2.loli.net/2022/06/05/BjELcG6i1RUDavq.png)

这两个矫正结果也暴露出Demons算法的一贯问题，即需要针对每次配准的图形手动调整超参数，包括卷积核大小，步长，惯性等。且随着迭代的增加效果可能会逐步变差。

#### 4.3.2 二值图像

我们项目还手动制作了一些二值图像，即圆形和方块，这两张图的白色区域面积相同，且像素大小相同，可以实现一一映射关系。其中方形图像为参考图像，而圆形图像为浮动图像。
![16544007438903.jpg](https://s2.loli.net/2022/06/05/r582YtZkRdSxgnA.jpg)

传统的Demons对于二值图像计算过慢，原因是计算驱动力时的梯度计算缓慢导致。我们项目因此只对加速的微分同胚Demons做了实验，实验效果上圆形转方形效果很好，随着逐步的迭代图像的四个角越发显现。迭代2000次后图片的SSD相似度逐渐减小，且暂未有收敛的趋势，可见对于这种全局大形变图像处理效果较好。右下角两张图是X，Y方向的变形场。

![demons_suqare_circle.png](https://s2.loli.net/2022/06/05/gDCGnkoHSijIc6Y.png)


## 5 基于深度学习的图像配准

图像配准即通过预测一个位移形变场，让一张图像对齐到另一张图像，使得对齐后的图像尽可能相似。传统的配准方法往往基于数学或物理模型优化的方法，之前的实验也验证了这些方法虽然准确度高且稳定，但耗费的时间往往较长。

近几年，基于深度学习的配准方法越发火热，即运用大量训练集训练深度网络模型，然后用这个训练好的模型对未知的图像对进行配准。这种方式的优点在于虽然训练过程较为缓慢，但配准（测试）过程比传统方法快很多，缺点则在于可供训练的数据较少，且没有传统方法稳定，还存在着可解释性差等弊端。

在本次项目中，我们尝试实验了经典的VoxelMorph模型架构，这一模型架构具有代表性，是许多无监督学习的图像配准基础框架。

### 5.1 模型概述

VoxelMorph架构输入的是浮动图像和参考图像的拼接，经过类Unet架构的encoder-decoder网络，预测从浮动图像到参考图像的位移场，最后根据位移场对图像进行变换，重采样得到配准后的图像。

![voxel_morph](https://s2.loli.net/2022/06/05/9nsEHbkvIlAJFNW.jpg)


**损失函数选择**：配准网络的损失一般包括图像的相似性和形变场的平滑正则项损失。图像相似性损失有很多种，常见的有相关系数（CC）、归一化的相关系数（NCC），互信息（MI）、均方误差（MSE）等。如果不加入平滑正则项，虽然模型在训练集上可能匹配的更准确，但往往这一映射会导致形变场的大尺度扭曲。下图是对VoxelMorph自带的脑部图像进行配准时的形变场，左图为加了平滑正则项，右图没加，可见加了后图像变化更具有局部性。

![with_without_regular](https://s2.loli.net/2022/06/05/YvE4K1SsraWejZl.png)


**网络模型架构**：VoxelMorph采用了Unet架构，与图像分割中的架构类似。先对输入的两个图像做隔一跳的下采样，不断提取特征,减半长宽到2*2，随后模型再进行隔一跳的上采样，进行图像还原。对于每层，模型会有个跳转架构skip输入，林阿杰下采样层和上采样层。在最后昨晚图像还原后VoxelMorph还接了两层卷积层作为Decoder。

![](https://s2.loli.net/2022/06/05/xKLVgOrHfhz3Aaq.jpg)

### 5.2 模型实现

**数据处理**:数据处理分为训练时的数据处理和与前端交互的数据处理。训练时的数据预处理详见vxm_data_generator()函数，函数生成时会对数据集中所有数据根据batch_size随机生成浮动图像和参考图像对，并将初始化输出格式，即配准图像和梯度图像。

```python
def vxm_data_generator(x_data, batch_size=16):
    """
    数据生成器,将输入的数据随机匹配成输入图像和输出图像的格式

    Inputs:  浮动图像: [bs, H, W, 1], 参考图像: [bs, H, W, 1]
    Outputs: 目标图像: [bs, H, W, 1], 全0梯度图像: [bs, H, W, 2]
    """
```

对于前端交互传入的图片,算法会先将图片进行标准化，即转化到0-1之间，随后将两张图片转化成【n，rows，cols，1】的格式一起传给模型进行预测。

```python
M = np.float32((M-np.min(M))/(np.max(M)-np.min(M)))
S = np.float32((S-np.min(S))/(np.max(S)-np.min(S)))

#处理输入
A=np.array([M,S])
val_input=[A[[0], ..., np.newaxis],A[[1], ..., np.newaxis]]
val_pred = self.model.predict(val_input)
```

**模型训练**：模型损失函数包含MSE损失和L2正则化函数，两者加权权重为10:1，模型选用Adam优化，初始学习率设置为1e-4。

```python
# 设置损失为MSE和梯度L2损失
losses = ['mse', vxm.losses.Grad('l2').loss]

#设置两个损失的权重
loss_weights = [1, 0.1]

#模型预编译，优化器选择Adam，初始学习率为1e-4
vxm_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=losses, loss_weights=loss_weights)
```

### 5.3 实验结果

 模型训练了20轮，平均一轮需要训练4分钟，损失趋于收敛。

 ![loss.png](https://s2.loli.net/2022/06/05/XhxiSCsHyY5QUrt.png)

随机选取脑部图像进行图像配准，一次配准仅需4s。虽然通用模型配准精度有限，配准速度确实相比传统方法快了很多。在神经网络模型进行粗配准后，再用传统模型进行更精细化调整是个不错的选择。
![registration.png](https://s2.loli.net/2022/06/05/ZcSd9uIHFxqAhOP.png)



## 6. GUI页面设计

我们使用PyQt5的包来设计页面，展示了基准图像，浮动图像，算法变换后的输出图像和灰度值差异比较。

### 6.1 页面代码

由于篇幅，只展示了设置主窗口的代码

```python
class work(QWidget):
    def __init__(self):
        '''
        窗口大小等默认值设置
        '''
        super().__init__()
        self.title = "图像处理期末PJ展示"
        self.left = 300
        self.top = 300
        self.width = 1020
        self.height = 800
        self.initUI()
        self.out=0

    def normalize(self,img):#归一化图像
        return np.array((img-img.min())/(img.max()-img.min())*255)

    def initUI(self):
        #主窗口
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.target = target
        self.source = source
        
        #修改小图标
        self.setWindowIcon(QIcon('base2.jpg')) 
        
        #各按钮显示设计
        button_display = QPushButton("加载图片", self)
        button_display.resize(100,50)
        button_display.move(30, 40)

        button_load = QPushButton("打开新图片", self)
        button_load.resize(100,50)
        button_load.move(30, 120)
        
        
        button_gen_float = QPushButton("生成浮动图像", self)
        button_gen_float.resize(100,50)
        button_gen_float.move(30, 200)


        button_ffd_cc = QPushButton("B样条FFD", self)
        button_ffd_cc.resize(100,50)
        button_ffd_cc.move(30, 280)

        button_ffd_local = QPushButton('仿射变换',self)
        button_ffd_local.resize(100,50)
        button_ffd_local.move(30, 360)

        button_ffd_nn = QPushButton('Demons',self)
        button_ffd_nn.resize(100,50)
        button_ffd_nn.move(30, 440)

        button_save_img = QPushButton('保存图像',self)
        button_save_img.resize(100,50)
        button_save_img.move(30,520)


        self.label_1 = QLabel(self)
        self.label_1.setGeometry(QRect(480, 20, 92, 31))
        self.label_1.setStyleSheet("font: 14pt \"Arial\";font-weight:bold;")
        self.label_1.setObjectName("label_1")
        # 图像显示组件
        self.label_3 = QLabel(self)
        self.label_3.setGeometry(QRect(200, 90, 300, 300))#x，y，长，宽的值
        self.label_3.setStyleSheet("font:28px;\n"
                                   "border-style:solid;\n"
                                   "border-width:1px;\n"
                                   "border-color:rgb(45, 45, 45);\n"
                                   "\n"
                                   "")#格式(有点像css)
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.label_4 = QLabel(self)
        self.label_4.setGeometry(QRect(600, 90, 300, 300))
        self.label_4.setStyleSheet("font:28px;\n"
                                   "border-style:solid;\n"
                                   "border-width:1px;\n"
                                   "border-color:rgb(45, 45, 45);\n"
                                   "\n"
                                   "")
        self.label_4.setText("")
        self.label_4.setObjectName("label_4")



        self.label_5 = QLabel(self)
        self.label_5.setGeometry(QRect(300, 50, 91, 31))
        self.label_5.setStyleSheet("font: 14pt \"Arial\";")
        self.label_5.setObjectName("label_5")
        self.label_6 = QLabel(self)
        self.label_6.setGeometry(QRect(700, 50, 92, 31))
        self.label_6.setStyleSheet("font: 14pt \"Arial\";")
        self.label_6.setObjectName("label_6")

                
        self.label_7 = QLabel(self)
        self.label_7.setGeometry(QRect(200, 460, 300, 300))
        self.label_7.setStyleSheet("font:28px;\n"
                                   "border-style:solid;\n"
                                   "border-width:1px;\n"
                                   "border-color:rgb(45, 45, 45);\n"
                                   "\n"
                                   "")
        self.label_7.setText("")
        self.label_7.setObjectName("label_7")
        self.label_8 = QLabel(self)
        self.label_8.setGeometry(QRect(600, 460, 300, 300))
        self.label_8.setStyleSheet("font:28px;\n"
                                   "border-style:solid;\n"
                                   "border-width:1px;\n"
                                   "border-color:rgb(45, 45, 45);\n"
                                   "\n"
                                   "")
        self.label_8.setText("")
        self.label_8.setObjectName("label_8")

        
        self.label_9 = QLabel(self)
        self.label_9.setGeometry(QRect(300, 420, 91, 31))
        self.label_9.setStyleSheet("font: 14pt \"Arial\";")
        self.label_9.setObjectName("label_9")
        self.label_10 = QLabel(self)
        self.label_10.setGeometry(QRect(700, 420, 92, 31))
        self.label_10.setStyleSheet("font: 14pt \"Arial\";")
        self.label_10.setObjectName("label_10")


        self.label_1.raise_()
        self.label_3.raise_()
        self.label_4.raise_()
        self.label_5.raise_()
        self.label_6.raise_()
        self.label_7.raise_()
        self.label_8.raise_()
        self.label_9.raise_()
        self.label_10.raise_()
        

        # 按钮的点击交互函数
        button_display.clicked.connect(self.display)
        button_gen_float.clicked.connect(self.gen_float)
        button_load.clicked.connect(self.load_img)
        button_save_img.clicked.connect(self.save_image)
        button_ffd_cc.clicked.connect(self.ffd_chose)
        button_ffd_local.clicked.connect(self.affine_SSD)
        button_ffd_nn.clicked.connect(self.show_child)
        #文字组件
        self.label_1.setText('结果展示')
        self.label_5.setText("基准图像")
        self.label_6.setText("浮动图像")
        self.label_9.setText("配准结果")
        self.label_10.setText("灰度值差异")
       
        self.show()
```

### 6.2 页面展示

页面展示如图

![image-20220604204356034.png](https://s2.loli.net/2022/06/05/V9XoTztWDvjl4gJ.png)

页面分为ABC三个板块，A部分设置的按钮，用户可以点击来进行不同的交互动作。B部分是结果的展示，分为了四个不同视图，包括基准图像，浮动图像，输出图像和灰度值差的展示。

### 6.3 页面功能介绍

通过点击`加载图片`按钮加载默认图像，可以方便我们测试模型的效果，不用去一一选择。

通过``打开新图片``按钮可以打开文件夹，选择新的基准图像和浮动图像，打开支持jpg,jpeg,png格式的图片。

如果没有浮动图像，可以点击``生成浮动图像``按钮来生成新的浮动图像，结果显示在右边的图片框中。

有总体四种配准方法可以选择，在每种方法内还可以选择不同的具体算法，点击`B样条FFD`按钮会跳出选择相似度的窗口C，包括一个下拉表单，用户可以通过选择SSD或者NCC相似度来进行下一步运算并比较他们的结果。点击`仿射变换`按钮会进行仿射变换，`Demons`按钮下会跳出新的选择窗口，用户可以选择一种算法来进行下一步运算，`Voxel`按钮可以进行Voxel网络的预测。

最后通过``保存图像``按钮可以将四张图像一起保存在本地。

除了三个板块，为了分析相似度的优化情况，FFD，仿射变换和Demon算法在运行结束后会绘制相似度随迭代次数的曲线图，并通过一个新窗口来显示它。新窗口不会在切换页面的时候关闭，以方便比较不同算法的相似度优化趋势和结果。



## 7. References：

### 参考论文

[1] Thirion, J.P., 1998. Image matching as a diffusion process: an analogy with Maxwell's demons. Medical image analysis, 2(3), pp.243-260.

[2] Vercauteren, T., Pennec, X., Perchant, A. and Ayache, N., 2009. Diffeomorphic demons: Efficient non-parametric image registration. NeuroImage, 45(1), pp.S61-S72.

[3] Balakrishnan, G., Zhao, A., Sabuncu, M.R., Guttag, J. and Dalca, A.V., 2019. VoxelMorph: a learning framework for deformable medical image registration. IEEE transactions on medical imaging, 38(8), pp.1788-1800.

[4] Thirion, J.P., 1998. Image matching as a diffusion process: an analogy with Maxwell's demons. Medical image analysis, 2(3), pp.243-260.

[5] Rogelj, P. and Kovačič, S., 2006. Symmetric image registration. Medical image analysis, 10(3), pp.484-493.

[6] Wang, H., Dong, L., O'Daniel, J., Mohan, R., Garden, A.S., Ang, K.K., Kuban, D.A., Bonnen, M., Chang, J.Y. and Cheung, R., 2005. Validation of an accelerated ‘demons’ algorithm for deformable image registration in radiation therapy. Physics in Medicine & Biology, 50(12), p.2887.

[7] Balakrishnan, G., Zhao, A., Sabuncu, M.R., Guttag, J. and Dalca, A.V., 2019. VoxelMorph: a learning framework for deformable medical image registration. IEEE transactions on medical imaging, 38(8), pp.1788-1800.


[8]杨佩. 基于Active Demons算法的非刚性图像配准方法研究[D].山东大学,2015.

[9]张丹. 基于Demons的非刚性图像配准算法研究[D].昆明理工大学,2020.DOI:10.27200/d.cnki.gkmlu.2020.000431.

### 参考文章

图像配准系列之基于FFD形变与梯度下降法的图像配准
https://blog.csdn.net/shandianfengfan/article/details/113750401

基于深度学习的单模医学图像配准综述（附VoxelMorph配准实例）
https://blog.csdn.net/zuzhiang/article/details/108601599

微分同胚demons配准算法原理与C++/Opencv实现
https://blog.csdn.net/shandianfengfan/article/details/123159271

图像配准算法之demons算法
https://blog.csdn.net/shandianfengfan/article/details/116550052

VoxelMorph Tutorial
https://colab.research.google.com/drive/1WiqyF7dCdnNBIANEY80Pxw_mVz4fyV-S?usp=sharing#scrollTo=HlxvpuoGPXPk

### 参考项目

https://github.com/SaJoke/Deep-Diffeomorphic-Demons

https://github.com/voxelmorph/voxelmorph

https://zmiclab.github.io/zxh/0/myops20/