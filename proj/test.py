
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


PATH = '1591349252.4838984.pth'
# PATH = '1591345605.9707272.pth'
model = models.resnet18(pretrained=False, num_classes=2)
model.load_state_dict(torch.load(PATH))

net = model
with torch.no_grad():
    device = None
    for X, y in test_iter:
        if isinstance(net, torch.nn.Module):
            net.eval()  # 评估模式, 这会关闭dropout
            outputs = net(X.to(device))
            print(outputs,'0')
            net.train()  # 改回训练模式
        else:  # 自定义的模型 
            if ('is_training' in net.__code__.co_varnames):  # 如果有is_training这个参数
                # 将is_training设置成False
                outputs = net(X, is_training=False)
                print(outputs,'1')
            else:
                outputs = net(X)
                print(outputs,'2')
    test_acc = d2l.evaluate_accuracy(test_iter, model)
    print('  test acc %.3f '% ( test_acc  ))    
 

