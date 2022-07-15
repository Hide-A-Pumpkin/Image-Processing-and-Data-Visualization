import cv2
import numpy as np
import time
import math
import matplotlib.pyplot as plt

def Inertial_demons( S,  M,  alpha, turns,similarity='NCC',record=False,momentum=0.75):
  '''
  Input:参考图像:S,浮动图像:M,归一化因子: alpha,训练轮数:turns,相似函数:similarity,record:是否记录NCC:动量:momentum
  Output:配准后图像（uint8）格式
  算法特点:Inertial demons函数在Active demons算法的基础做了改进,
  把上一层迭代计算得到的偏移量加入到当前层迭代的偏移量计算当中，使系统拥有惯性防止突然的变动
  提升了收敛速度和配准精度
  '''

  #初始化每个点在x，y方向上的变形驱动力，格式为float32
  Ux=np.zeros_like(S,dtype='float32')
  Uy=np.zeros_like(S,dtype='float32')

  #将原始图像转换格式为float32，方便后续处理
  S=np.array(S,dtype='float32') 
  M_tmp=np.array(M,dtype='float32')

  #初始化最大归一化系数NCC
  loss=[]
  e=0

  #进行turns轮迭代
  for i in range(turns):

    #保存上一轮迭代的驱动力
    Ux_last,Uy_last  = Ux.copy(),Uy.copy() 

    #计算这一轮迭代的驱动力
    Ux, Uy=active(S, M_tmp, alpha)

    #每一轮的驱动力为当轮驱动力+上一轮驱动力乘以动量
    Ux,Uy = Ux+Ux_last*momentum , Uy+Uy_last*momentum 


    #浮动图像基于变换场进行变换
    M_tmp=movepixels_2d(M, Ux, Uy) 

    if similarity=='NCC':
      e=NCC(S,M_tmp)

    if record:
      loss.append(e)
  
  #将浮点型转换为uchar型输出
  D=np.array(M_tmp,dtype='uint8')
  return D,loss


def Symmetric_demons( S,  M,  alpha, turns,similarity='NCC',record=False,momentum=0.75):
  '''
  Input:参考图像:S,浮动图像:M,归一化因子: alpha,训练轮数:turns,相似函数:similarity,record：是否记录NCC，动量:momentum
  Output:配准后图像（uint8）格式
  算法特点：Symmetric demons函数不同于Inertial demons，将形变向量根据牛顿第三定理改为对称梯度。
          该方法图像对参考图像和浮动图像上的力求了平均。
  '''

  #初始化每个点在x，y方向上的变形驱动力，格式为float32
  Ux=np.zeros_like(S,dtype='float32')
  Uy=np.zeros_like(S,dtype='float32')

  #将原始图像转换格式为float32，方便后续处理
  S=np.array(S,dtype='float32') 
  M_tmp=np.array(M,dtype='float32')

  #初始化归一化系数
  loss=[]
  e=0

  #进行turns轮迭代
  for i in range(turns):

    #保存上一轮迭代的驱动力
    Ux_last,Uy_last  = Ux.copy(),Uy.copy() 

    #计算这一轮迭代的驱动力
    Ux, Uy=symmetric(S, M_tmp, alpha)

    #每一轮的驱动力为当轮驱动力+上一轮驱动力乘以动量
    Ux,Uy = Ux+Ux_last*momentum , Uy+Uy_last*momentum 

    #浮动图像基于变换场进行变换
    M_tmp=movepixels_2d(M_tmp, Ux, Uy) 

    if similarity=='NCC':
      e=NCC(S,M_tmp)

    if record:
      loss.append(e)
  
  #将浮点型转换为uchar型输出
  D=np.array(M_tmp,dtype='uint8')
  return D,loss

def movepixels_2d(S,  Ux, Uy, interpolation=cv2.INTER_CUBIC):
  '''
  Input：待变换图像：S;x,y方向上的变形场:Ux,Uy ,插值方法：interpolation,默认双三样条插值
  Output:变形后图像
  函数流程：对待变换图每个坐标点，计算该坐标点基于变形场Ux，Uy后的x，y左边，通过双三样条插值求出变形后的图像
  '''

  #将原始图转化为float32格式
  S=np.mat(S,dtype='float32')
  rows,cols=S.shape

  #初始化变形后图像
  Tx_map=np.zeros_like(S,dtype='float32')
  Ty_map=np.zeros_like(S,dtype='float32')

  #对原图每个坐标点进行坐标变换和插值
  for i in range(rows):
      for j in range(cols):

        #根据坐标偏移计算坐标映射
        x = float(j + Ux[i, j])
        y = float(i + Uy[i, j])

        #判断坐标映射是否超出范围,没越界就赋值
        if x >= 0 and x < cols and y >= 0 and y < rows:
          Tx_map[i, j] = x
          Ty_map[i, j] = y

  #像素重采样,使用双三样条插值，Tx_map表示（i，j）点变形的x值， Ty_map表示（i，j）点变形的y值
  S=cv2.remap(S, Tx_map, Ty_map, interpolation) 
  return S

def active(S,  M,  alpha):
  '''
  Input：参考图像:S,浮动图像:M;参考图像在x，y方向的梯度Sx,Sy;扩散速度系数α
  Output:x,y方向变形场
  函数流程：先计算浮动图像和参考图像的差值，计算浮动图像和参考图像在x，y方向上的梯度，
          随后计算每个点的变形场，再对求解的变形场进行高斯平滑。
  '''

  #计算浮动图像和参考图像的差值
  diff = M - S   

  #初始化变形场
  Ux=np.zeros_like(S,dtype='float32')
  Uy=np.zeros_like(S,dtype='float32')
 
  #计算参考图像和浮动图像在x，y方向的梯度Sx,Sy
  Mx, My=get_mat_gradient(M) 
  Sx, Sy=get_mat_gradient(S) 

  #计算原始图像的shape
  rows,cols= S.shape

  for i in range(rows):
    for j in range(cols):
      #计算驱动扩散的内力为的分母（参考图像和浮动图像）
      sdenominator = pow(Sx[i,j], 2) + pow(Sy[i,j], 2) + pow(alpha, 2)*pow(diff[i,j], 2)
      mdenominator = pow(Mx[i,j], 2) + pow(My[i,j], 2) + pow(alpha, 2)*pow(diff[i,j], 2)

      #为防止驱动力过大，当分母太小时，将该坐标下的驱动置0
      if ((sdenominator > -0.0000001 and sdenominator < 0.0000001) or (mdenominator > -0.0000001 and mdenominator < 0.0000001)):
        Ux[i,j] = 0.0
        Uy[i,j] = 0.0
      else:  
        #计算x方向上的驱动力，注意是梯度反方向
        ux = Sx[i,j] / sdenominator + Mx[i,j] / mdenominator
        Ux[i,j] = (-diff[i,j]*ux)

        uy = Sy[i,j] / sdenominator + My[i,j] / mdenominator
        Uy[i,j] = (-diff[i,j]*uy)

  Ux=Ux*10
  Uy=Uy*10

  #对变形场进行高斯平滑，减小毛刺
  ksize,sigma=127, 10.0
  Ux=cv2.GaussianBlur(Ux,(ksize, ksize), sigma, sigma) 
  Uy=cv2.GaussianBlur(Uy, (ksize, ksize), sigma, sigma)

  return Ux,Uy

def get_mat_gradient(S):
  '''
  Input:矩阵S
  Output:矩阵在x，y方向上的梯度矩阵
  '''
  #定义Sobel梯度算子
  ker_x = np.array( [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0]).reshape(3,3)
  ker_y = np.array( [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0]).reshape(3,3)

  #将图像转化为float32格式
  S=np.array(S,dtype='float32')

  #使用Sobel算子作卷积运算，得到x、y方向的梯度
  Tx=cv2.filter2D(S,ddepth=-1, kernel=ker_x)
  Ty=cv2.filter2D(S,ddepth=-1, kernel=ker_y)

  return  Tx,Ty

def symmetric( S,  M,  alpha):
  '''
  Input：参考图像:S,浮动图像:M;参考图像在x，y方向的梯度Sx,Sy;扩散速度系数α
  Output:x,y方向变形场
  函数流程：先计算浮动图像和参考图像的差值，计算浮动图像和参考图像在x，y方向上的梯度，
          随后计算每个点的变形场，再对求解的变形场进行高斯平滑。
  '''

  #计算浮动图像和参考图像的差值
  diff = M - S

  #初始化变形场
  Ux=np.zeros_like(S,dtype='float32')
  Uy=np.zeros_like(S,dtype='float32')
 
  #计算参考图像和浮动图像在x，y方向的梯度Sx,Sy
  Mx, My=get_mat_gradient(M) 
  Sx, Sy=get_mat_gradient(S) 

  #计算原始图像的shape
  rows,cols= S.shape

  for i in range(rows):
    for j in range(cols):
      #计算驱动扩散的内力为的分母（参考图像和浮动图像）
      denominator = pow(Sx[i,j]+Mx[i,j], 2) + pow(Sy[i,j]+My[i,j], 2)+ 4*pow(alpha, 2)*pow(diff[i,j], 2)

      #为防止驱动力过大，当分母太小时，将该坐标下的驱动置0
      if denominator > -0.0000001 and denominator < 0.0000001:
        Ux[i,j] = 0.0
        Uy[i,j] = 0.0
      else:  
      #计算x方向上的驱动力，注意是梯度反方向
        ux = (Sx[i,j]+Mx[i,j]) / denominator
        Ux[i,j] = (-2*diff[i,j]*ux)

        uy = (Sy[i,j]+My[i,j]) / denominator
        Uy[i,j] = (-2*diff[i,j]*uy)

  Ux=Ux*10
  Uy=Uy*10

  #对变形场进行高斯平滑，减小毛刺
  ksize,sigma=127, 10.0
  Ux=cv2.GaussianBlur(Ux,(ksize, ksize), sigma, sigma) 
  Uy=cv2.GaussianBlur(Uy, (ksize, ksize), sigma, sigma)

  return Ux,Uy

def Alpha_demons( S,  M,  alpha, turns,similarity='NCC',record=False):
  '''
  Input:参考图像:S,浮动图像:M,归一化因子: alpha,训练轮数:turns,相似函数:similarity,record：是否记录NCC
  Output:配准后图像（uint8）格式
  函数流程：每轮计算当轮的驱动力，对浮动图像进行变换
  算法特点：Alpha demons算法在原始Demons算法基础上加入了扩散速度系数α ,α越大偏移量越小
  '''

  #初始化每个点在x，y方向上的变形驱动力，格式为float32
  Ux=np.zeros_like(S,dtype='float32')
  Uy=np.zeros_like(S,dtype='float32')

  #将原始图像转换格式为float32，方便后续处理
  S=np.array(S,dtype='float32') 
  M_tmp=np.array(M,dtype='float32')


  #初始化归一化系数
  loss=[]
  e=0
  
  #进行turns轮迭代
  for i in range(turns):
    
    #计算这一轮迭代的驱动力
    Ux, Uy=normal(S, M_tmp, alpha)

    #浮动图像基于变换场进行变换
    M_tmp=movepixels_2d(M_tmp, Ux, Uy) 

    if similarity=='NCC':
      e=NCC(S,M_tmp)
      

    if record:
      loss.append(NCC(S,M_tmp))
  
  #将浮点型转换为uchar型输出
  D=np.array(M_tmp,dtype='uint8')
  return D,loss

def Active_demons( S,  M,  alpha, turns,similarity='NCC',record=False):
  '''
  Input:参考图像:S,浮动图像:M,归一化因子: alpha,训练轮数:turns,相似函数:similarity,record：是否记录NCC
  Output:配准后图像（uint8）格式
  函数流程：每轮计算当轮的驱动力，对浮动图像进行变换
  算法特点：Active demons函数把浮动图像的梯度加入偏移量的计算中,提高图形收敛速度
  '''

  #初始化每个点在x，y方向上的变形驱动力，格式为float32
  Ux=np.zeros_like(S,dtype='float32')
  Uy=np.zeros_like(S,dtype='float32')

  #将原始图像转换格式为float32，方便后续处理
  S=np.array(S,dtype='float32') 
  M_tmp=np.array(M,dtype='float32')

  #初始化归一化系数
  loss=[]
  e=0

  #进行turns轮迭代
  for i in range(turns):

    #计算这一轮迭代的驱动力
    Ux, Uy=active(S, M_tmp, alpha)

    #浮动图像基于变换场进行变换
    M_tmp=movepixels_2d(M_tmp, Ux, Uy) 

    if similarity=='NCC':
      e=NCC(S,M_tmp)

    if record:
      loss.append(e)
  
  #将浮点型转换为uchar型输出
  D=np.array(M_tmp,dtype='uint8')
  return D,loss

def normal(S,  M,  alpha):
  '''
  Input：参考图像:S,浮动图像:M;参考图像在x，y方向的梯度Sx,Sy;扩散速度系数α
  Output:x,y方向变形场
  函数流程：先计算浮动图像和参考图像的差值，计算参考图像在x，y方向上的梯度，
          随后计算每个点的变形场，再对求解的变形场进行高斯平滑。
  '''

  #计算浮动图像和参考图像的差值
  diff = M - S  

  #初始化变形场
  Ux=np.zeros_like(S,dtype='float32')
  Uy=np.zeros_like(S,dtype='float32')
 
  #计算参考图像在x，y方向的梯度Sx,Sy
  Sx, Sy=get_mat_gradient(S) 

  #计算原始图像的shape
  rows,cols= S.shape

  for i in range(rows):
    for j in range(cols):
      #计算驱动扩散的内力为的分母（参考图像和浮动图像）
      sdenominator = pow(Sx[i,j], 2) + pow(Sy[i,j], 2) + pow(alpha, 2)*pow(diff[i,j], 2)

      #为防止驱动力过大，当分母太小时，将该坐标下的驱动置0
      if sdenominator > -0.0000001 and sdenominator < 0.0000001:
        Ux[i,j] = 0.0
        Uy[i,j] = 0.0
      else:  
        #计算x方向上的驱动力，注意是梯度反方向
        ux = Sx[i,j] / sdenominator 
        Ux[i,j] = (-diff[i,j]*ux)

        uy = Sy[i,j] / sdenominator 
        Uy[i,j] = (-diff[i,j]*uy)

  #对变形场进行高斯平滑，减小毛刺
  ksize,sigma=127, 10.0
  Ux=cv2.GaussianBlur(Ux,(ksize, ksize), sigma, sigma) 
  Uy=cv2.GaussianBlur(Uy, (ksize, ksize), sigma, sigma)

  return Ux,Uy

def NCC(S , M ):
  '''
  Input:两个矩阵
  Output:计算矩阵互相关相似度
  '''
  S=np.array(S,dtype='float32') 
  M=np.array(M,dtype='float32') 

  SM,SS,MM=0,0,0

  rows,cols=S.shape

  for i in range(rows):
    for j in range(cols):
        SM += S[i,j]*M[i,j]
        SS += pow(S[i,j],2)
        MM += pow(M[i,j],2)
      
  return SM / math.sqrt(SS*MM)

def img_gaussian(U ,sigma):
  '''
  Input:图像U，高斯核标准方差
  Output:高斯平滑后图像，高斯核大小为3*3
  函数介绍：图像的高斯平滑
  '''
  radius = np.ceil(3 * sigma)
  ksize = int(2 * radius + 1)
  return cv2.GaussianBlur(U,(ksize, ksize), sigma, sigma) 

def exp_field(Vx, Vy):
  '''
  Input: x，y方向上的位移场Vx，Vy
  Output:x，y方向上的微分同胚映射Vx，Vy
  函数介绍：将位移场转换为近似微分同胚映射
  '''

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
 
 
def composite(Vx,Vy):
  '''
  Input:x,y方向的微分同胚矩阵Vx，Vy
  Output：复合运算的微分同胚矩阵
  '''

  bxp=movepixels_2d(Vx, Vx, Vy, cv2.INTER_LINEAR)
  byp=movepixels_2d(Vy, Vx, Vy, cv2.INTER_LINEAR)
 
 
  return  Vx + bxp,Vy + byp

def Diffeomorphic_demons( S,  M,  alpha, sigma_fluid,sigma_diffusion,turns,similarity='NCC',record=False):

  '''
  Input:参考图像:S,浮动图像:M,归一化因子: alpha,高斯滤波参数:sigma_fluid sigma_diffusion,训练轮数:turns,相似函数:similarity,record:是否记录NCC
  Output:配准后图像(uint8)格式
  算法特点:Diffeomorphic demons算法
  '''

  #初始化每个点在x，y方向上的变形驱动力，格式为float32
  Vx=np.zeros_like(S,dtype='float32')
  Vy=np.zeros_like(S,dtype='float32')

  #将原始图像转换格式为float32，方便后续处理
  S=np.array(S,dtype='float32') 
  M_tmp=np.array(M,dtype='float32')

  #计算参考图像在x，y方向的梯度Sx,Sy
  Sx, Sy=get_mat_gradient(S) 
  Sx_min=Sx.copy()
  Sy_min=Sy.copy()

  #初始化归一化系数
  e_max = -(1e6)
  e=0
  loss=[]

  #进行turns轮迭代
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

    if record:
      loss.append(e)
  
  M_tmp=movepixels_2d(M, Sx_min, Sy_min, cv2.INTER_LINEAR)

  #将浮点型转换为uchar型输出
  D=np.array(M_tmp,dtype='uint8')
  return D,loss
 



def Plot(base,float,out,diff):
    plt.subplot(2, 2, 1), plt.axis("off")
    plt.imshow(float,cmap='gray'),plt.title('float')
    plt.subplot(2, 2, 2), plt.axis("off")
    plt.imshow(base,cmap='gray'),plt.title('baseline')
    plt.subplot(2, 2, 3), plt.axis("off")
    plt.imshow(out,cmap='gray'),plt.title('out')
    plt.subplot(2, 2, 4), plt.axis("off")
    plt.imshow(diff,cmap='gray'),plt.title('out-base')
  

if __name__ == '__main__':
  #初始化参数
  img1 = cv2.imread("base1.jpeg",cv2.IMREAD_GRAYSCALE)
  img2 = cv2.imread("float1.jpeg",cv2.IMREAD_GRAYSCALE)
  turns=200
  alpha=10.0


  #计算Alpha_demons变换
  #out,Alpha_cc=Alpha_demons(img1, img2, alpha , turns,similarity='NCC',record=True)
  #diff = np.abs(np.array(img1,dtype='int64')-np.array(out,dtype='int64'))
  #cv2.imwrite("Alpha_out.png",out)
  #cv2.imwrite("Alpha_diff.png",diff)

  #计算Active_demons变换
  #out,Active_cc=Active_demons(img1, img2, alpha , turns,similarity='NCC',record=True)
  #diff = np.abs(np.array(img1,dtype='int64')-np.array(out,dtype='int64'))
  #cv2.imwrite("Active_out.png",out)
  #cv2.imwrite("Active_diff.png",diff)

  #计算Inertial_demons变换
  #out,Inertial_cc=Inertial_demons(img1, img2, alpha , turns,similarity='NCC',record=True)
  #diff = np.abs(np.array(img1,dtype='int64')-np.array(out,dtype='int64'))
  #cv2.imwrite("Innertial_out.png",out)
  #cv2.imwrite("Innertial_diff.png",diff)

  #计算Symmetric_demons变换
  out,Symmetric_cc=Symmetric_demons(img1, img2, alpha , turns,similarity='NCC',record=True)
  diff = np.abs(np.array(img1,dtype='int64')-np.array(out,dtype='int64'))
  #cv2.imwrite("Symmetric_out.png",out)
  #cv2.imwrite("Symmetric_diff.png",diff)

  #计算Diffeomorphic_demons变换
  #out,Diffeomorphic_cc=Diffeomorphic_demons(img1, img2, alpha ,0.05,0.05, turns,similarity='NCC',record=True)
  #diff = np.abs(np.array(img1,dtype='int64')-np.array(out,dtype='int64'))
  #cv2.imwrite("Diffeomorphic_out.png",out)
  #cv2.imwrite("Diffeomorphic_diff.png",diff)

