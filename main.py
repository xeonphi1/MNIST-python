import random
import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import time
from tensorboardX import SummaryWriter
# Hyper Parameters
EPOCH =50 # 训练整批数据多少次, 为了节约时间, 我们只训练2次
BATCH_SIZE = 2048# 批训练的子集大小
LR = 0.01  # learning rate
DOWNLOAD_MNIST = True  # 如果你已经下载好了mnist数据就写上 Fasle

train_data = torchvision.datasets.MNIST(
    root='./mnist/',  # 保存或者提取位置
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成
    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_MNIST,  # 没下载就下载, 下载了就不用再下了
)

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)

# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor) / 255.
test_y = test_data.targets

print(train_data.data.size())
print(test_data.targets.size())
print(test_data.data.size())
print(test_data.targets.size())  # 输出训练和测试集的size


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=8,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,  # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(8, 16, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(16 * 7 * 7, 10)  # fully connected layer, output 10 classes


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output

cnn = CNN()
print(cnn)  # 输出网络结构
writer = SummaryWriter('logs/7')
graph_inputs = torch.from_numpy(np.random.rand(2, 1)).type(torch.FloatTensor)
#writer.add_graph(cnn, (graph_inputs,))
if_use_gpu = 2
if if_use_gpu:
    cnn = cnn.cuda()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hottedC:\Users\xeon\PycharmProjects\BOTTLE\yolov5-master\runs\train
test_x = test_x.cuda()  # 在训练过程中进行测试时，需要提前将测试数据由tensor.cpu()——>tensor,gpu()

for epoch in range(EPOCH):
    start = time.time()
    for step, (b_x, b_y) in enumerate(train_loader):  # 分配 batch data, normalize x when iterate train_loader
        if if_use_gpu:
            b_x = b_x.cuda()
            b_y = b_y.cuda()
        output = cnn(b_x)  # cnn output
        loss = loss_func(output, b_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients



        if step % 1 == 0:  # 每100个batch训练之后，输出loss以及对应此时测试集在网络中的准确率
            test_output = cnn(test_x)
            pred_y = torch.max(test_output.cpu(), 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('step',step,'Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(), '| test accuracy: %.4f' % accuracy)
            loss2=loss.cpu().data.numpy()
            writer.add_scalar('Train_loss', loss2, (step + 1))
            writer.add_scalar('acc', accuracy, step)

    interval = time.time() - start
    print
    "Time: %.4f" % interval
