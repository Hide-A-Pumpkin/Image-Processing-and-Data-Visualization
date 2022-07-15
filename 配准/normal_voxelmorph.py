import cv2
import voxelmorph as vxm 
import neurite as ne 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def vxm_data_generator(x_data, batch_size=32):
    """
    数据生成器,将输入的数据随机匹配成输入图像和输出图像的格式

    Inputs:  浮动图像: [bs, H, W, 1], 参考图像: [bs, H, W, 1]
    Outputs: 目标图像: [bs, H, W, 1], 全0梯度图像: [bs, H, W, 2]
    """

    # 获取每张图大小
    vol_shape = x_data.shape[1:] 
    #获取图像数
    ndims = len(vol_shape)
    
    # 准备一个全0的梯度数组
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    
    while True:
        # 随机选取batch_size个数组索引，作为浮动图像
        idx1 = np.random.randint(0, x_data.shape[0], size=batch_size)

        #提取索引在idx1的图像，并将他们扩招一个维度
        moving_images = x_data[idx1, ..., np.newaxis]

        # 随机选取batch_size个数组索引，作为参考图像
        idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)

        #提取索引在idx2的图像，并将他们扩招一个维度
        fixed_images = x_data[idx2, ..., np.newaxis]

        #输入为浮动图像和参考图像对
        inputs = [moving_images, fixed_images]
        
        # 输出为参考图像和初始化变形场
        outputs = [fixed_images, zero_phi]
        
        yield (inputs, outputs)

def plot_history(hist, loss_name='loss'):
    # Simple function to plot training history.
    plt.figure()
    plt.plot(hist.epoch, hist.history[loss_name], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()




class VXM_basic():
    def __init__(self,train=False,model_path='brain_2d_smooth.h5') :

        '''
        VoxelMorph框架基本模型，使用Unet架构，训练集和测试集选用提供的脑部数据集
        Inputs: 是否训练: False ,不训练的话导入的模型地址:model_path

        '''

        npz = np.load('data.npz')
        x_train = npz['train']
        vol_shape = x_train.shape[1:]

        # Unet特征
        nb_features = [
            [32, 32, 32, 32],         # encoder 特征
            [32, 32, 32, 32, 32, 16]  # decoder 特征
        ]

        # 初始化训练网络
        vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0)


        # 设置损失为MSE和梯度L2损失
        losses = ['mse', vxm.losses.Grad('l2').loss]

        #设置两个损失的权重
        loss_weights = [1, 0.01]

        #模型预编译，优化器选择Adam，初始学习率为1e-4
        vxm_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=losses, loss_weights=loss_weights)

        if train:
            #生成训练数据生成器
            train_generator = vxm_data_generator(x_train)
    
            #训练模型，epoch为20，每个epoch的step为100
            nb_epochs = 20
            steps_per_epoch = 100
            hist = vxm_model.fit_generator(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=2)
            plot_history(hist)

        else:
            vxm_model.load_weights(model_path)

        self.model=vxm_model

    def Predict(self,S,M):
        # 用训练好的模型进行预测
        #规范化图片到0-1之间
        M = np.float32((M-np.min(M))/(np.max(M)-np.min(M)))
        S = np.float32((S-np.min(S))/(np.max(S)-np.min(S)))

        #处理输入
        A=np.array([M,S])
        val_input=[A[[0], ..., np.newaxis],A[[1], ..., np.newaxis]]
        val_pred = self.model.predict(val_input)


        # 可视化预测结果，val_input里包含浮动图和参考图，val_pred里包含配准图和变形场
        #images = [img[0, :, :, 0] for img in val_input + val_pred] 
        #titles = ['moving', 'fixed', 'moved', 'flow']
        #ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)

        # 可视化变形场
        #flow = val_pred[1].squeeze()[::3,::3]
        #ne.plot.flow([flow], width=5)
        return np.float32((val_pred[0]-np.min(val_pred[0]))/(np.max(val_pred[0])-np.min(val_pred[0]))*255)[0,...,0]

    def Compare(self,val_input):
        # 使用 MSE + smoothness loss 的模型
        self.model.load_weights('brain_2d_smooth.h5')
        our_val_pred = self.model.predict(val_input)

        # 只使用 MSE  loss 的模型
        self.model.load_weights('brain_2d_no_smooth.h5')
        mse_val_pred = self.model.predict(val_input)

        # 可视化使用 MSE + smoothness loss 的模型预测结果
        images = [img[0, ..., 0] for img in [val_input[1], *our_val_pred]]
        titles = ['fixed', 'MSE + smoothness', 'flow']
        ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)

        # 可视化只使用 MSE 的模型预测结果
        images = [img[0, ..., 0] for img in [val_input[1], *mse_val_pred]]
        titles = ['fixed', 'MSE only', 'flow']
        ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)

        ne.plot.flow([img[1].squeeze()[::3, ::3] for img in [our_val_pred, mse_val_pred]], width=10)



if __name__ == '__main__':
    # 生成测试集生成器
    img1=cv2.imread('figure/brain0.jpeg',cv2.IMREAD_GRAYSCALE)
    img2=cv2.imread('figure/brain2.jpeg',cv2.IMREAD_GRAYSCALE)

    #VXM 基本模型，使用Unet架构
    model=VXM_basic(train=False,model_path='brain_2d_smooth.h5')
    out=model.Predict(img1,img2)