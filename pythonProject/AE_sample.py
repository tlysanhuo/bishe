import torch
from torch import nn
from scipy import io
import numpy as np
from d2l import torch as d2l
from torch.utils.data import TensorDataset, DataLoader
import torch.autograd
import random
import time
from pandas import read_csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import hiddenlayer as h
import math

'''
上面的都是掉包，torch深度学习包，numpy数组包，
pandas包用来读excel或csv数据的，这里我数据的标准化和多采样率的预处理放在另一个文件，所以你看不到
d2l是我给你发的教材的老师自己建的包，辅助用的
from torch.utils.data import TensorDataset,DataLoader
import torch.autograd
import random
这三别动，第三个是让你找最好模型的随机数的
time计时包
matplotlib画图的包
'''


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


'''
这个是每次随机数固定，和下面的abcdefg配合，比如现在是54，那么网络每次训练出来的结果都是54号种子下的固定结果
这是方便统计所有结果的平均值和找寻最优结果的
'''

pppppppppppp1 = list()
pppppppppppp2 = list()
pppppppppppp3 = list()
pppppppppppp4 = list()
for abcdefg in range(54, 55):  # 121

    # 设置随机数种子
    setup_seed(abcdefg)

    trainXdata = torch.load('trainXdata-SAE.pth')
    trainYdata_double = torch.load('trainYdata_double-SAE.pth')
    testXdatafinalSAE1 = torch.load('testXdatafinalSAE1.pth')
    testYdatafinalSAE1 = torch.load('testYdatafinalSAE1.pth')
    testXdatafinalSAE2 = torch.load('testXdatafinalSAE2.pth')
    testYdatafinalSAE2 = torch.load('testYdatafinalSAE2.pth')
    testXdatafinalSAE3 = torch.load('testXdatafinalSAE3.pth')
    testYdatafinalSAE3 = torch.load('testYdatafinalSAE3.pth')
    testXdatafinalSAE4 = torch.load('testXdatafinalSAE4.pth')
    testYdatafinalSAE4 = torch.load('testYdatafinalSAE4.pth')

    '''
    这些load都是读取我预先存在文件里的数据，里面每个xxxxx.pth都是一组张量
    '''

    trainXdata = trainXdata.to(torch.float32)
    trainYdata_double = trainYdata_double.to(torch.float32)
    testXdatafinalSAE1 = testXdatafinalSAE1.to(torch.float32)
    testYdatafinalSAE1 = testYdatafinalSAE1.to(torch.float32)
    testXdatafinalSAE2 = testXdatafinalSAE2.to(torch.float32)
    testYdatafinalSAE2 = testYdatafinalSAE2.to(torch.float32)
    testXdatafinalSAE3 = testXdatafinalSAE3.to(torch.float32)
    testYdatafinalSAE3 = testYdatafinalSAE3.to(torch.float32)
    testXdatafinalSAE4 = testXdatafinalSAE4.to(torch.float32)
    testYdatafinalSAE4 = testYdatafinalSAE4.to(torch.float32)

    '''
    因为torch的训练要求，所有数据都需要转为32位
    '''

    '''
    ********************下面的   几个部分    非常重要*********************
    这部分是你定义网络的部分，网络都是nn.Module类，首先在def __init__(self):里定义网络每层的维度
    比如：self.linear1 = nn.Linear(33,32)就是一个叫self.linear1的全连接层，输入维度为33，输出维度为32
         self.linear2 = nn.Linear(32，31)就是一个叫self.linear2的全连接层，输入维度为32，输出维度为31
    此时，他们之间都是单层的“积木”，他们之间没有任何联系

    而为了给每层网络构成一个有先有后的结构，就是把“积木拼起来”，让数据先进入self.linear1，再进入self.linear2，再。。。。。
    则需要def forward(self, X):    其中X是你输入的数据

    self.encoderout = self.linear1(self.Selu(X))  ，X先经过selu函数激活，再通过linear1得到一个encoderout的数据
    你要分清之间的区别，linear1是网络，他是一个f(x)，而encoderout就是一个y
    最后，大部分情况下输出的都是y   （   不过如果你要保存网络的参数的话，有时也要输出f(x)   ）
    '''


    class autoencoder(nn.Module):
        def __init__(self):
            super(autoencoder, self).__init__()
            self.linear1 = nn.Linear(33, 32)
            self.linear2 = nn.Linear(32, 31)
            self.linear3 = nn.Linear(31, 30)
            self.linear4 = nn.Linear(30, 29)

            self.linear5 = nn.Linear(29, 30)
            self.linear6 = nn.Linear(30, 31)
            self.linear7 = nn.Linear(31, 32)
            self.linear8 = nn.Linear(32, 33)

            self.Selu = nn.SELU()

        def forward(self, X):
            self.encoderout = self.linear1(self.Selu(X))
            # print(self.encoderout.shape)
            # 可在此加shortcut
            self.encoderout = self.linear2(self.Selu(self.encoderout))
            self.encoderout = self.linear3(self.Selu(self.encoderout))
            self.encoderout = self.linear4(self.Selu(self.encoderout))

            self.decoderout = self.linear5(self.Selu(self.encoderout))
            self.decoderout = self.linear6(self.Selu(self.decoderout))
            self.decoderout = self.linear7(self.Selu(self.decoderout))
            self.decoderout = self.linear8(self.Selu(self.decoderout))
            return self.decoderout, self.encoderout


    '''
        如果你熟悉AE的结构，你应该知道AE是重构输入的无监督训练方式，也就是说X和X_hat比较，但是最后需要用的是中间隐藏层提取的hidden
        这也就是为什么网络会return两个y，一个y是X_hat（self.decoderout）   ，另一个y是hidden（self.encoderout）
        '''

    '''
    重要。既然网络结构已经定义好了，接下来就是定义一些损失函数，优化器之类的东西，也就是定义训练器
    首先定义网络的初始化参数，也就是f(x)=ax+b里的a和b，不过不定义也无所谓，他自己会定义的
    然后是训练模式，一般我都是CPU跑的，网络层数不深还好

    criterion是损失函数，这里用的是nn.MSELoss()，这玩意本质上也是个网络，
    你可以点进去看一看是啥样的，你复现MR-SAE需要对这个损失函数做修改的；

    optimizer是优化器，现在用的是adam，别动它就行，影响不大

    “”“”“”关键来了“”“”“”“”
    从for开始到最后我用文字给你解释下它干了啥：
    对于你规定的每个训练周期epoch，使用网络的训练模式
        对于你训练集及其标签，（那个i没啥用，放着就好），首先优化器的梯度会清零zero_grad()
        然后取出X输入到网络中并输出结果，你可以看到这里有X_hat, HK两个输出。
            一般都是一个，但因为这是AE，网络规定了有两个，所以这里也是两个。如果你只有一个输出，会输出个元组继而报错。注意下
        然后，网络不是通过bp算法训练的吗，l = criterion(X_hat, y)就是根据你网络的输出和标签y计算梯度，
            l.backward()反向传播
            optimizer.step()参数更新

    这样一个训练器就设计好了，之后就是拿数据进行训练了
    '''


    def trainer(net, train_iter, num_epochs, lr, device):
        def init_weights(m):
            if type(m) == nn.ConvTranspose1d or type(m) == nn.Conv1d:
                nn.init.xavier_uniform_(m.weight)

        net.apply(init_weights)
        print('training on', device)
        net.to(device)
        # 定义损失函数 与 优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        for epoch in range(num_epochs):
            # 训练损失之和，训练准确率之和，样本数
            metric = d2l.Accumulator(3)
            net.train()
            for i, (X, y) in enumerate(train_iter):
                optimizer.zero_grad()
                X_hat, Y = net(X)
                l = criterion(X_hat, y)
                print(f'epo={epoch},loss = {l}')
                l.backward()
                optimizer.step()


    time_start = time.time()  # 计时用

    batch_size = 32  # batch size不解释
    lr, num_epochs = 0.005, 8  # 学习率和训练周期不解释

    trainx = trainXdata
    print(trainx.shape)
    '''
    trainx就是输入AE的重构数据，下面的操作给你解释下
    网络训练的数据都需要以一个特定的迭代器输入，
        首先train_ids = TensorDataset(trainx, trainx)，是集合你训练器里的X和y，因为AE的特性，输入输出都是X
        然后train_iter = DataLoader(dataset=train_ids, batch_size=batch_size, shuffle=True)生成迭代器，抄就行
        netpre = autoencoder()是将网络实例化，假设netpre1 = autoencoder()，netpre2 = autoencoder().
            netpre1和netpre2都是autoencoder这网络的结构，但他们是两个f(x)，
            比喻来说：就是autoencoder是猫类，netpre1就是我家养的橘猫，netpre2就是隔壁家养的狸花猫

    之前某种意义上都是准备工作
    trainer(netpre, train_iter, num_epochs, lr, d2l.try_gpu()) 这一行代表你训练网络模型了
    仔细看你就会发现是上面定义的训练器，你要用什么训练器进行训练就用什么训练器，可以是trainer1，trainer2，看你自己需求
    如此以来，netpre()这个f(x)就训练好了,后面有句代码为：X_hat, Hk = netpre(trainXdata)，就是说输入trainXdata，
    他会给输出两个值，一个是重构X_hat，另一个就是重中之重的，提取的隐藏层特征Hk
    '''

    train_ids = TensorDataset(trainx, trainx)
    train_iter = DataLoader(dataset=train_ids, batch_size=batch_size, shuffle=True)
    netpre = autoencoder()
    trainer(netpre, train_iter, num_epochs, lr, d2l.try_gpu())

    '''
    定义提取的特征作为输入，最后需要软测量的质量变量y作为输出的全连接层网络
    '''


    class onedecnn(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear22 = nn.Sequential(nn.SELU(), nn.Linear(29, 5))
            # 序列化了，意思就是linear里有两层，一层激活函数一层全连接层，无关紧要，你可以对比上下看一看，
            # 上面网络的激活函数在forward里，而这里的激活函数在__init__里，其实意思意义的
            self.linear23 = nn.Sequential(nn.SELU(), nn.Linear(5, 1))

        def forward(self, X):
            # self.layer1 = self.linear21(X)
            self.layer2 = self.linear22(X)
            self.layer3 = self.linear23(self.layer2)

            return self.layer3


    def trainer2(net, train_iter, num_epochs, lr, device, batchsize):
        print('training on', device)

        net.to(device)
        # 定义损失函数 与 优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        for epoch in range(num_epochs):
            # 训练损失之和，训练准确率之和，样本数
            net.train()
            for i, (X, y) in enumerate(train_iter):
                optimizer.zero_grad()
                X_hat = net(X)
                ############### 你看，因为这里只有一个输出，所以要重新定义训练器，
                # 之前的训练器会输出两个参数而刚才定义的网络结构就一个输出
                # 当然，你也可以再return一个无关紧要的输出，用之前定义的训练器，但为了好查错，也为了避免冲突，不推荐
                l = criterion(X_hat.reshape(-1), y)
                # 你要注意输出的数据形式，这里为啥要.reshape(-1)呢，
                # 因为有时候，X_hat的tensor形式可能为(32,1),而y的形式为(32),
                # 这时候就会因为 广播机制 出问题，所以要reshape到同样的(32)
                print(f'epo={epoch},loss = {l}')
                l.backward()
                optimizer.step()


    '''
    你看，刚才训练的模型就提取出了Hk特征，而因为是网络输出，所以带有一个梯度，
    所以需要Hk = Hk.data把数据转为不带梯度的纯数据
    '''
    X_hat, Hk = netpre(trainXdata)
    Hk = Hk.data

    lr = 0.001
    num_epochs = 10
    train_ids = TensorDataset(Hk, trainYdata_double[:, 0])
    train_iter = DataLoader(dataset=train_ids, batch_size=batch_size, shuffle=True)
    netf = onedecnn()
    trainer2(netf, train_iter, num_epochs, lr, d2l.try_gpu(), batch_size)
    print('训练结束啦')

    time_end = time.time()  # 结束计时
    time_c = time_end - time_start  # 运行所花时间
    print('time cost', time_c, 's')

    '''
    网络训练就结束了，之后都是一些测试，首先通过AE，即netpre提取特征testHk，这里是四个工况，所以提取4遍
    然后让他们.data
    然后将特征输入到训练好的全连接层网络netf，得到预测的y_test_HAT
    '''

    X_hat, testHk1 = netpre(testXdatafinalSAE1)
    X_hat, testHk2 = netpre(testXdatafinalSAE2)
    X_hat, testHk3 = netpre(testXdatafinalSAE3)
    X_hat, testHk4 = netpre(testXdatafinalSAE4)
    testHk1 = testHk1.data
    testHk2 = testHk2.data
    testHk3 = testHk3.data
    testHk4 = testHk4.data
    y_test_HAT1 = netf(testHk1)
    y_test_HAT2 = netf(testHk2)
    y_test_HAT3 = netf(testHk3)
    y_test_HAT4 = netf(testHk4)

    '''
    之后就是得到MSE和画图了，我这老眼昏花，所以每个mse我都会乘以1440，实际值要除以1440，之后的内容自己研究
    *******神经网络的本质*******
    先定义网络结构net，再定义个训练器trainer，实例化网络net_exist，将数据生成迭代器train_iter
    然后自己设置下lr学习率，num_epoch训练周期
    上面的都是准备工作，实际上网络的训练，就是：
    trainer(net_exist, train_iter, num_epochs, lr, d2l.try_gpu())    
    本质上，就是求解y=f(x)，你理解整个过程就好做了
    '''

    testYdatafinalSAE1 = testYdatafinalSAE1[:, 0].reshape(960)
    testYdatafinalSAE2 = testYdatafinalSAE2[:, 0].reshape(960)
    testYdatafinalSAE3 = testYdatafinalSAE3[:, 0].reshape(960)
    testYdatafinalSAE4 = testYdatafinalSAE4[:, 0].reshape(960)
    mse1 = 0
    mse2 = 0
    mse3 = 0
    mse4 = 0
    for i in range(960):
        mse1 += (y_test_HAT1[i] - testYdatafinalSAE1[i]) ** 2
        mse2 += (y_test_HAT2[i] - testYdatafinalSAE2[i]) ** 2
        mse3 += (y_test_HAT3[i] - testYdatafinalSAE3[i]) ** 2
        mse4 += (y_test_HAT4[i] - testYdatafinalSAE4[i]) ** 2
    print(mse1 / 960 * 1440)
    print(mse2 / 960 * 1440)
    print(mse3 / 960 * 1440)
    print(mse4 / 960 * 1440)
    print(abcdefg)
    pppppppppppp1.append(mse1 / 960 * 1440)
    pppppppppppp2.append(mse2 / 960 * 1440)
    pppppppppppp3.append(mse3 / 960 * 1440)
    pppppppppppp4.append(mse4 / 960 * 1440)
    fig = plt.figure()
    testYdatafinalSAE1 = testYdatafinalSAE2.detach().numpy()
    y_test_HAT1 = y_test_HAT2.detach().numpy()
    # print(y_test_HAT)
    y_test_HAT1 = y_test_HAT1.reshape(960)
    plt.plot(testYdatafinalSAE1, color='b', label='Yreal')
    plt.plot(y_test_HAT1, color='r', label='Ypred')
    # plt.title(f'SAE{abcdefg}')
    plt.title(f'AE1')
    plt.xlabel('data')
    plt.ylabel('output')
    plt.legend(loc='lower right')
    plt.show()
    torch.save(y_test_HAT1, 'SAE1.pth')
print('result1_1')
for i in range(100):
    print(float(pppppppppppp1[i]))
print('result1_2')
for i in range(100):
    print(float(pppppppppppp2[i]))
print('result1_3')
for i in range(100):
    print(float(pppppppppppp3[i]))
print('result1_4')
for i in range(100):
    print(float(pppppppppppp4[i]))