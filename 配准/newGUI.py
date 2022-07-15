import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import cv2 as cv
import numpy as np
import time
from matplotlib import pyplot as plt
import SimpleITK as sitk
from generate_float import generate_float
import os
from FFD_SSD import optimization_gd_ffd,init_param 
from affine import optimization_gd_affine
from Demons import *
from normal_voxelmorph import *

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
        '''
        主页面UI设置
        '''
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

        button_ffd_voxel = QPushButton('Voxel',self)
        button_ffd_voxel.resize(100,50)
        button_ffd_voxel.move(30,520)

        button_save_img = QPushButton('保存图像',self)
        button_save_img.resize(100,50)
        button_save_img.move(30,600)


        self.label_1 = QLabel(self)
        self.label_1.setGeometry(QRect(480, 20, 250, 31))
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
        button_ffd_voxel.clicked.connect(self.show_voxel)
        #文字组件
        self.label_1.setText('结果展示')
        self.label_5.setText("基准图像")
        self.label_6.setText("浮动图像")
        self.label_9.setText("配准结果")
        self.label_10.setText("灰度值差异")


                
        self.show()

    
    def display(self):
        '''
        图片展示部分，默认图片是预先存好的图片
        '''
        print("图片显示")
        im = QImage(target.copy(), target.shape[1], target.shape[0], target.shape[1], QImage.Format_Grayscale8)
        img = QPixmap(im).scaled(self.label_3.width(), self.label_3.height(),Qt.KeepAspectRatio)
        self.label_3.setPixmap(img)

        im2 = QImage(source.copy(), source.shape[1], source.shape[0], source.shape[1], QImage.Format_Grayscale8)
        img2 = QPixmap(im2).scaled(self.label_4.width(), self.label_4.height(),Qt.KeepAspectRatio)
        self.label_4.setPixmap(img2)
    
    def load_img(self):
        '''
        从文件中读取基准图像和浮动图像
        其中用到了QT里的Qfiledialog函数读取文件读入文件支持jpg,png,jpeg格式
        图像默认格式是QImage,在这里转换成np.int8保存在类中
        '''
        print('来自文件：')
        global imgNamepath  # 这里为了方便别的地方引用图片路径，将其设置为全局变量
 
        try:
            # 弹出一个文件选择框，第一个返回值imgName记录选中的文件路径+文件名，第二个返回值imgType记录文件的类型
            # QFileDialog就是系统对话框的那个类第一个参数是上下文，第二个参数是弹框的名字，第三个参数是默认打开的路径，第四个参数是需要的格式
            imgNamepath, imgType = QFileDialog.getOpenFileName(self, "读取基准图片...",
                                                                "","*.jpg;*.jpeg;*.png;;All Files(*)")
            # 通过文件路径获取图片文件，并设置图片长宽为label控件的长、宽
            target =cv.imdecode(np.fromfile(imgNamepath,dtype=np.uint8),-1)
            # target = np.int8(target)
            if len(target.shape)>2:
                target = cv2.cvtColor(target,cv2.COLOR_RGB2GRAY)
            #转化成灰度图像
            print('从%s中加载图片'%imgNamepath)
            self.target = target
            # im = QImage(target.copy(), target.shape[1], target.shape[0], target.shape[1], QImage.Format_Grayscale8)
            # img = QPixmap(im).scaled(self.label_3.width(), self.label_3.height(),Qt.KeepAspectRatio)
            img = QPixmap(imgNamepath).scaled(self.label_3.width(), self.label_3.height(),Qt.KeepAspectRatio)
            self.label_3.setPixmap(img)

            imgNamepath, imgType = QFileDialog.getOpenFileName(self, "读取浮动图片...",
                    "","*.jpg;*.jpeg;*.png;;All Files(*)")
            # 通过文件路径获取图片文件，并设置图片长宽为label控件的长、宽
            source =cv.imdecode(np.fromfile(imgNamepath,dtype=np.uint8),-1)#高级版cv.imread，防止中文文件名无法识别的问题
            # source = np.int8(source)
            if len(source.shape)>2:
                source = cv2.cvtColor(source,cv2.COLOR_RGB2GRAY)
            print('从%s中加载图片'%imgNamepath)
            self.source = source#保存在类中

            img2 = QPixmap(imgNamepath).scaled(self.label_3.width(), self.label_3.height(),Qt.KeepAspectRatio)
            self.label_4.setPixmap(img2)

        except:
            print('读取图片失败！')
            # cv.waitKey(0)
            cv.destroyAllWindows()

         
    def gen_float(self):
        '''
        生成随机浮动图像
        '''
        print("生成随机浮动图像")
        # gen_float_img(img)
        start=time.time()
        # dst=img.copy()
        try:
            dst = generate_float(target)
            end=time.time()
            self.source = dst

            im2 = QImage(self.source.copy(), self.source.shape[1], self.source.shape[0], self.source.shape[1], QImage.Format_Grayscale8)
            img2 = QPixmap(im2).scaled(self.label_4.width(), self.label_4.height(),Qt.KeepAspectRatio)
            # print(self.label_4.width(), self.label_4.height())
            self.label_4.setPixmap(img2)

            print("spend time:",end-start,"s")
            # cv.imshow("float_img",dst)
            # cv.waitKey(0) 
            # cv.destroyAllWindows() 
        except:
            print('还未导入图片')

    def ffd_chose(self):
        child_window = ChoseSim()
        child_window.show()
        # child_window.exec()
        if child_window.item!='':
            self.ffd_cc(child_window.item)
        cv.destroyAllWindows()

    def ffd_cc(self,item):
        '''
        计算FFD变换的函数
        调用了FFD_ssd.py中的函数，手动计算了灰度值差值
        '''
        row_block_num = 5
        col_block_num = 5
        source = self.source
        target = self.target
        grid_points = init_param(source, row_block_num, col_block_num, -10, 10)
        '''
        如果row和col的大小是3,那么grid_points的长度为6*6*2=72,梯度的长度为72
        '''
        out = np.zeros_like(source,dtype='float')
        t1=time.time()
        out,loss = optimization_gd_ffd(source, target, out, row_block_num, col_block_num,grid_points,item)
        self.label_1.setText('FFD_'+item+'结果')
        out = np.int8(out)


        diff = (np.int32(out)-np.int32(target))+130
        diff = np.int8(diff)

        t2 = time.time()
        print('spend time:', t2-t1)

        out = QImage(out.copy(), out.shape[1], out.shape[0],out.shape[1], QImage.Format_Grayscale8)
        out = QPixmap(out).scaled(self.label_7.width(), self.label_7.height(),Qt.KeepAspectRatio)
        self.label_7.setPixmap(out)

        diff = QImage(diff.copy(), diff.shape[1], diff.shape[0],diff.shape[1], QImage.Format_Grayscale8)
        diff = QPixmap(diff).scaled(self.label_8.width(), self.label_8.height(),Qt.KeepAspectRatio)

        self.label_8.setPixmap(diff)
        print('Bspline—FFD完成')
        print('total iteration:',len(loss))

        X1=range(len(loss))
        plt.clf()
        plt.plot(X1,loss,'g--')
        plt.title(item+'similarity')
        plt.xlabel('iteration')
        plt.ylabel(item)
        plt.savefig(item+'ffd_loss.jpg')

        dst = cv.imread(item+'ffd_loss.jpg',cv.IMREAD_GRAYSCALE)#输出损失函数，自动保存。
        cv.imshow(item+"loss",dst)
        cv.waitKey(0) 
        cv.destroyAllWindows() 

        
    def affine_SSD(self):
        '''
        SSD仿射变换展示
        '''
        source = self.source
        target = self.target
        A = np.random.randn(2,2)*0.1+np.eye(2)
        b = np.random.randn(2,1)*5
        # A = np.array([[1,0],[0,1]])
        A=A.reshape((-1))#把初始参数展开成一维
        b=b.reshape((-1))
        grid_points = np.concatenate((A,b)) #展成一维向量
        print('init param', grid_points)
        # grid_points.reshape((-1))#变成一维向量

        t1=time.time()
        out,loss = optimization_gd_affine(source,target, grid_points)
        self.label_1.setText('仿射_SSD'+'结果')
        out = np.int8(out)
        self.out = out
        t2 = time.time()
        print('spend time:', t2-t1)

        
        diff =  (np.int32(out)-np.int32(target))+130
        diff = np.int8(diff)

        out = QImage(out.copy(), out.shape[1], out.shape[0],out.shape[1], QImage.Format_Grayscale8)
        out = QPixmap(out).scaled(self.label_7.width(), self.label_7.height(),Qt.KeepAspectRatio)
        self.label_7.setPixmap(out)

        diff = QImage(diff.copy(), diff.shape[1], diff.shape[0],diff.shape[1], QImage.Format_Grayscale8)
        diff = QPixmap(diff).scaled(self.label_8.width(), self.label_8.height(),Qt.KeepAspectRatio)
        self.label_8.setPixmap(diff)
        print('仿射配准完成')
        print('total iteration:',len(loss))

        X1=range(len(loss))
        plt.clf()
        plt.plot(X1,loss,'g--')
        plt.title('SSD similarity')
        plt.xlabel('iteration')
        plt.ylabel('SSD')
        plt.savefig('affine_result.jpg')

        dst = cv.imread('affine_result.jpg',cv.IMREAD_GRAYSCALE)
        cv.imshow("SSD affine",dst)
        cv.waitKey(0) 
        cv.destroyAllWindows() 

    def show_child(self):
        '''
        打开表单窗口
        '''
        child_window = ChoseForm()
        child_window.show()
        # child_window.exec()
        if child_window.item!='':
            self.display_demons(child_window.item)
        cv.destroyAllWindows()

        

    def display_demons(self,item):
        '''
        demons算法运算
        '''
        turns=10
        alpha=10.0
        t1 = time.time()
        if item == 'Active':
            out,loss=Active_demons(self.target, self.source, alpha , turns,similarity='NCC',record=True)
        elif item == 'Alpha':
            out,loss=Alpha_demons(self.target, self.source, alpha , turns,similarity='NCC',record=True)
        elif item == 'Inertial':
            out,loss=Inertial_demons(self.target, self.source, alpha , turns,similarity='NCC',record=True)
        elif item == 'Symmetric':
            out,loss=Symmetric_demons(self.target, self.source, alpha , turns,similarity='NCC',record=True)
        elif item == 'Diffeomorphic':
            out,loss=out,loss=Diffeomorphic_demons(self.target, self.source, alpha ,0.05,0.05, turns,similarity='NCC',record=True)
        out = np.int8(out)
        self.out = out
        t2 = time.time()
        print('spend time:', t2-t1)

        
        diff =  (np.int32(out)-np.int32(self.target))+130
        diff = np.int8(diff)

        out = QImage(out.copy(), out.shape[1], out.shape[0],out.shape[1], QImage.Format_Grayscale8)
        out = QPixmap(out).scaled(self.label_7.width(), self.label_7.height(),Qt.KeepAspectRatio)
        self.label_7.setPixmap(out)

        diff = QImage(diff.copy(), diff.shape[1], diff.shape[0],diff.shape[1], QImage.Format_Grayscale8)
        diff = QPixmap(diff).scaled(self.label_8.width(), self.label_8.height(),Qt.KeepAspectRatio)
        self.label_8.setPixmap(diff)
        print('demons配准完成')
        self.label_1.setText(item+'_demons 结果')
        print('total iteration:',len(loss))

        X1=range(len(loss))
        plt.clf()
        plt.plot(X1,loss,'g--')
        plt.title('NCC similarity')
        plt.xlabel('iteration')
        plt.ylabel('NCC')
        plt.savefig(item+'_demons_result.jpg')

        dst = cv.imread(item+'_demons_result.jpg',cv.IMREAD_GRAYSCALE)
        cv.imshow(item+"_demons result",dst)
        cv.waitKey(0) 
        cv.destroyAllWindows() 

    def show_voxel(self):
        t1=time.time()
        model = VXM_basic(train=False,model_path='brain_2d_smooth.h5')
        out = model.Predict(self.source,self.target)
        out = np.int8(out)
        self.out = out
        t2 = time.time()
        print('spend time:', t2-t1)
    
        diff =  (np.int32(out)-np.int32(self.target))+130
        diff = np.int8(diff)

        out = QImage(out.copy(), out.shape[1], out.shape[0],out.shape[1], QImage.Format_Grayscale8)
        out = QPixmap(out).scaled(self.label_7.width(), self.label_7.height(),Qt.KeepAspectRatio)
        self.label_7.setPixmap(out)

        diff = QImage(diff.copy(), diff.shape[1], diff.shape[0],diff.shape[1], QImage.Format_Grayscale8)
        diff = QPixmap(diff).scaled(self.label_8.width(), self.label_8.height(),Qt.KeepAspectRatio)
        self.label_8.setPixmap(diff)
        print('voxel配准完成')
        self.label_1.setText('voxel 结果')


    def save_image(self):
        '''
        保存图片
        '''
        try:
            cv.imwrite('base.jpg',self.normalize(self.target))
            print('成功保存基准图像到：'+os.getcwd()+'\\base.jpg')
        except:
            print('图片不存在！')
        try:
            cv.imwrite('float.jpg',self.normalize(self.source))
            print('成功保存浮动图像到：'+os.getcwd()+'\\float.jpg')
        except:
            print('图片不存在！')
        try:
            cv.imwrite('out.jpg',self.out)
            print('成功保存输出图像到：'+os.getcwd()+'\\out.jpg')
        except:
            print('图片不存在！')


class ChoseForm(QWidget):
    '''
    选择demons算法的具体算法窗口
    '''
    
    def __init__(self):
        super().__init__()
        self.item=''
        items=('Alpha','Active','Inertial','Symmetric','Diffeomorphic')
        item, ok=QInputDialog.getItem(self, "选择算法", '选择一种算法', items, 0, False)
        if ok and item:
            self.item = item

class ChoseSim(QWidget):
    '''
    选择相似度度量方法窗口
    '''
    
    def __init__(self):
        super().__init__()
        self.item=''
        items=('SSD','NCC')
        item, ok=QInputDialog.getItem(self, "选择算法", '选择一种相似度', items, 0, False)
        if ok and item:
            self.item = item
        



if __name__ == '__main__':
    # np.random.seed(1234)
    target = cv.imread("base2.jpg" ,cv.IMREAD_GRAYSCALE) 
    source = cv.imread('float2.jpg',cv.IMREAD_GRAYSCALE)
    target = cv.resize(target, source.shape[::-1])
    rows = target.shape[0]
    cols = target.shape[1]
    app = QApplication(sys.argv)
    ex = work()
    sys.exit(app.exec_())
