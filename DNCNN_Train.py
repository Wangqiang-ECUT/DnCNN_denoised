"""
一维去噪 #音频大地电磁
 DnCNN
 """
import torch  # 包 torch 包含了多维张量的数据结构以及基于其上的多种数学操作。
import matplotlib.pyplot as plt  # 画图
import numpy as np  # NumPy(Numerical Python) 是 Python 语言的一个扩展程序库，支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。

# torch.nn，nn就是neural network的缩写，这是一个专门为深度学习而设计的模块。torch.nn的核心数据结构是Module，这是一个抽象的概念，
# 既可以表示神经网络中的某个层（layer），也可以表示一个包含很多层的神经网络。在实际使用中，最常见的做法是继承nn.Module，撰写自己的网络/层。
import torch.nn as nn

import torch.optim as optim  # torch.optim是一个实现了各种优化算法的库。
import time  # 记录整个训练时间
import scipy.io as sio  # 加载.mat文件
import math  # 数学运算
from torch.utils.data import TensorDataset, DataLoader  # DataLoader类的作用就是实现数据以什么方式输入到什么网络中。
from d2 import DnCNN  # 导入DnCNN模型
# from d3 import ResNet18
import gc  # 垃圾回收，请理内存
from torchsummary import summary

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 判断gpu是否能用
# net = DnCNN().to(device)  # 设置网络对象
net = DnCNN().to(device)
summary(net, input_size=(1, 75))
# summary(ResNet18(), (1, 255, 255))
# summary(DnCNN, input_size=[(1, 255 ,255)], batch_size=128, device="cpu")
### 超参数设置 ###
start = time.process_time()
EPOCH = 100 # 遍历数据集次数 100
BATCH_SIZE_s = 128  # 批处理尺寸(batch_size)
BATCH_SIZE_v = 128  # 128
LR = 0.0001  # 0.000001 学习率
rate = 0.9  # 学习率衰变
iteration = 10  # 每10次衰减
###########################################################

Origin_s = r'train_good.mat'
Origin_s = sio.loadmat(Origin_s)
Origin_s = Origin_s['dataMatrix'] #假设文件中存有字符变量是matrix，
Origin_s = Origin_s[:,np.newaxis,:]
MTsignal_s = r'train_noise.mat'
MTsignal_s = sio.loadmat(MTsignal_s)
MTsignal_s = MTsignal_s['dataMatrix'] #假设文件中存有字符变量是matrix，
MTsignal_s = MTsignal_s[:, np.newaxis, :]
Origin_s = Origin_s[0:5000, :, :]
MTsignal_s = MTsignal_s[0:5000:, :]
Ls = 5000
Origin_v = r'test_good.mat'
Origin_v = sio.loadmat(Origin_v)
Origin_v = Origin_v['dataMatrix'] #假设文件中存有字符变量是matrix，
Origin_v = Origin_v[:, np.newaxis, :]
MTsignal_v = r'test_noise.mat'
MTsignal_v = sio.loadmat(MTsignal_v)
MTsignal_v = MTsignal_v['dataMatrix'] #假设文件中存有字符变量是matrix，
MTsignal_v = MTsignal_v[:, np.newaxis, :]
Origin_v = Origin_v[0:500, :, :]
MTsignal_v = MTsignal_v[0:500, :, :]
Lv = 500
#############################################################
###数据预处理###
# 归一化
def normalization(data, _range):
    return data / _range

# 训练集归一化
matirx_s = np.row_stack((Origin_s, MTsignal_s))
range_s = np.max(abs(matirx_s))
norm_s = normalization(matirx_s, range_s)
np.save('range_s', range_s)
Origin_s = norm_s[0:Ls, :]
MTsignal_s = norm_s[Ls:Ls * 2, :]

# 验证集归一化
matirx_v = np.row_stack((Origin_v, MTsignal_v))
range_v = np.max(abs(matirx_v))
norm_v = normalization(matirx_v, range_v)
np.save('range_v', range_v)
Origin_v = norm_v[0:Lv, :]
MTsignal_v = norm_v[Lv:Lv * 2, :]

# 训练集格式转换
x1_s = torch.from_numpy(Origin_s)
x2_s = torch.from_numpy(MTsignal_s)
x1_s = x1_s.type(torch.FloatTensor)
x2_s = x2_s.type(torch.FloatTensor)

# 验证集格式转换
x1_v = torch.from_numpy(Origin_v)
x2_v = torch.from_numpy(MTsignal_v)
x1_v = x1_v.type(torch.FloatTensor)
x2_v = x2_v.type(torch.FloatTensor)

# 数据封装打乱顺序
train_data = TensorDataset(x2_s, x1_s)
val_data = TensorDataset(x2_v, x1_v)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE_s, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE_v, shuffle=True, num_workers=0, drop_last=True)

# 变量清除
#del dataset_s
del Origin_s
del MTsignal_s
#del dataset_v
del Origin_v
del MTsignal_v
del matirx_s
del norm_s
del matirx_v
del norm_v
del x1_s
del x1_v
del x2_s
del x2_v
gc.collect()

###网络训练###
# 定义是否使用GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型定义
# net = DnCNN(channels=1).to(device)
# net = SCWNet18().to(device)
criterion = nn.MSELoss()
criterion.cuda()
optimizer = optim.Adam(net.parameters(), lr=LR)
#optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

# 开始训练
Losslist_s = []
Losslist_v = []
best_loss = 600
save_path = './net.pth'
print("Start Training!")
for epoch in range(EPOCH):
    since = time.time()
    print('\nEpoch: %d' % (epoch + 1))
    if epoch % iteration == 9:
        LR = LR*rate
    loss_s = 0.0
    loss_v = 0.0
    #for i in range(Ls//BATCH_SIZE_s):
    for i, data_s in enumerate(train_loader, 0):
        net.train()
        net.zero_grad()
        optimizer.zero_grad()
        input_s, target_s = data_s
        input_s = input_s.to(device)
        output_s = net(input_s)
        target_s = target_s.to(device)
        loss_s0 = criterion(output_s, target_s)
        loss_s0.backward()
        optimizer.step()
        loss_s += loss_s0.item()
    Losslist_s.append(loss_s/(Ls//BATCH_SIZE_s))
    net.eval()
    with torch.no_grad():
        #for j in range(Lv//BATCH_SIZE_v):
        for j, data_v in enumerate(val_loader, 0):
            input_v, target_v = data_v
            input_v = input_v.to(device)
            output_v = net(input_v)    #前向算法
            target_v = target_v.to(device)
            loss_v0 = criterion(output_v, target_v)
            loss_v += loss_v0.item()
        Losslist_v.append(loss_v/(Lv//BATCH_SIZE_v))
        if loss_v0 < best_loss:
            best_loss = loss_v0
            torch.save(net.state_dict(), save_path)
    if (epoch+1) % 1 == 0:
        print('train loss: {:.10f}'.format(loss_s/(Ls//BATCH_SIZE_s)))
        print('val loss: {:.10f}'.format(loss_v/(Lv//BATCH_SIZE_v)))   #length
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('finished training')

###绘图###
# 格式转换
input_v = input_v.cpu()
input_v = input_v.detach().numpy()
output_v = output_v.cpu()
output_v = output_v.detach().numpy()
target_v = target_v.cpu()
target_v = target_v.detach().numpy()

# Loss变化
x = range(1, EPOCH + 1)
y_s = Losslist_s
y_v = Losslist_v
plt.semilogy(x, y_s, 'b.-')
plt.semilogy(x, y_v, 'r.-')
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.show()
#plt.savefig("accuracy_loss.jpg")
torch.save(net, r'D:\Qiang_wang\TCN-KSVD\样本库制作\仿真数据测试\test_4\DnCNN\DnCNN_quzao_GPU.pth')
#plt.plot(epoch, LR)  # 描述lr和epoch的关系
#plt.show()
#torch.load(net, 'net.pth')
# 去噪前
origSignal = target_v
errorSignal = target_v - input_v
signal_2 = sum(origSignal.flatten() ** 2)
noise_2 = sum(errorSignal.flatten() ** 2)
SNRValues3 = 10 * math.log10(signal_2 / noise_2)
print(SNRValues3)

# 去噪后
origSignal = target_v
errorSignal = target_v - output_v
signal_2 = sum(origSignal.flatten() ** 2)
noise_2 = sum(errorSignal.flatten() ** 2)
SNRValues4 = 10 * math.log10(signal_2 / noise_2)
print(SNRValues4)

y1_list = []
y2_list = []
y3_list = []
col = 0  # 显示第几个数据去噪效果
while col < 10:
      x = range(0, len(target_v[0, 0, :]))
      y1 = target_v[col, 0, :]
      y2 = input_v[col, 0, :]
      y3 = output_v[col, 0, :]
      plt.plot(x, y1, 'b.-')
      plt.plot(x, y2, 'r.-')
      plt.plot(x, y3, 'g.-')
      plt.xlabel('Time')
      plt.ylabel('Ampulitude')

      origSignal = y1
      errorSignal = y1 - y2
      signal_2 = sum(origSignal.flatten() ** 2)  # **为幂计算，2**3=2的3次方，sum为求和，可叠代，flatten为降维为1维，按行的方向
      noise_2 = sum(errorSignal.flatten() ** 2)  # 噪声
      SNRValues1 = 10 * math.log10(signal_2 / noise_2)  # 信噪比计算公式
      print(SNRValues1)  # 打印出信噪比的值

      # 去噪后
      origSignal = y1
      errorSignal = y1 - y3
      signal_2 = sum(origSignal.flatten() ** 2)  # 信号
      noise_2 = sum(errorSignal.flatten() ** 2)  # 噪声
      SNRValues2 = 10 * math.log10(signal_2 / noise_2)  # 信噪比计算公式
      print(SNRValues2)  # 输出去噪后信噪比的值
      plt.show()
      col += 1
      y1_list.append(y1)
      y2_list.append(y2)
      y3_list.append(y3)

end = time.process_time()  # 记录时间的函数
print(end - start)  # 打印出开始到结束的消耗时间
