
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
import os

import sys
sys.path.append("..") 
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = 'hotdog'
os.listdir(os.path.join(data_dir, "hotdog")) # ['train', 'test']

train_imgs = ImageFolder(os.path.join(data_dir, 'hotdog/train'))
test_imgs  = ImageFolder(os.path.join(data_dir, 'hotdog/test'))

# 画出前八张后八张
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i-1][0] for i in range(8)]  #not_hotdogs文件夹里从后面读回来
print(train_imgs[1])     #'tuple' object
print(train_imgs[1][0])  #'Image' object
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)

# data transform 指定RGB三个通道的均值和方差来将图像通道归一化
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  #默认的值
train_augs = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

test_augs = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        normalize
    ])


# 加载模型
pretrained_net = models.resnet18(pretrained=False)
print(pretrained_net.fc)  #最后的输出个数等于目标数据集的类别数1000

pretrained_net.fc = nn.Linear(512, 2)
print(pretrained_net.fc)

# 设置学习率
# map函数配合匿名函数使用
# x = list(map(lambda a:a*10,range(0,10))) # 序列中的每个元素乘以10
output_params = list(map(id, pretrained_net.fc.parameters()))    #id()函数用于获取对象的内存地址
feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())  #从可迭代元素中过滤不想要的数据

lr = 0.01
optimizer = optim.SGD([{'params': feature_params},
                       {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}],
                       lr=lr, weight_decay=0.001)

import time
PATH = str(time.time())+'.pth'
print(PATH)

#batch_size 可以继续调大,得看自己电脑CUDA 内存
def train_fine_tuning(net, optimizer, batch_size=20, num_epochs=5):
    print("start to load data")
    train_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/train'), transform=train_augs),
                            batch_size, shuffle=True)
    test_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/test'), transform=test_augs),
                           batch_size)
    print("finish loading")
    loss = torch.nn.CrossEntropyLoss()
    print("start to train")
    d2l.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)
# train_fine_tuning(pretrained_net, optimizer)
scratch_net = models.resnet18(pretrained=False, num_classes=2)
lr = 0.1
optimizer = optim.SGD(scratch_net.parameters(), lr=lr, weight_decay=0.001)
train_fine_tuning(scratch_net, optimizer) 

torch.save(scratch_net.state_dict(), PATH)



import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
import os

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  #默认的值

test_augs = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            normalize
 ])


test_iter = DataLoader(ImageFolder("testimg", transform=test_augs),1)


# PATH = '1591342359.343528.pth'
# PATH = '1591345605.9707272.pth'
# model = models.resnet18(pretrained=False, num_classes=2)
# model.load_state_dict(torch.load(PATH))

# with torch.no_grad():
#     # for X, y in test_iter:
#     #     outputs = scratch_net(X.to(device))
#     #
#     #     # outputs = torch.squeeze(outputs.view(1,-1))
#     #     print(outputs)
test_acc = d2l.evaluate_accuracy(test_iter, scratch_net)
print('  test acc %.3f '% ( test_acc  ))


