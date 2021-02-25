'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
import random
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--train_angle', default=0, type=int)
parser.add_argument('--test_angle', default=0, type=int)
parser.add_argument('--translate', default=0.0, type=float)
parser.add_argument('--translate_by', default=0.0, type=float)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
train_angle = args.train_angle
#test_angle = args.test_angle
test_angle = args.translate_by*180
translate = args.translate
translate_by = args.translate_by

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    #transforms.Lambda(lambda image: image.rotate(random.random() * train_angle * 2 - train_angle)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Lambda(lambda image: image.rotate(random.random() * test_angle * 2 - test_angle)),
    transforms.RandomAffine(degrees=0, translate=(translate_by, translate_by)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
net = VGG('VGG16')
#net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
#net = RegNetX_200MF()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

def test_save_pred():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    total_pred = torch.LongTensor()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total_pred = torch.cat((total_pred, predicted.cpu()), 0)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    np.save("./saved_data/stdcnn_vgg16_cifar10_"+str(translate)+"_trans_rot_"+str(translate_by)+"_test_pred.npy", total_pred.numpy())

milestones = [50, 100, 140]
scheduler = MultiStepLR(optimizer, milestones, gamma=0.1)

net = torch.load("./saved_models/stdcnn_vgg16_cifar10_"+str(translate)+"_trans_rot.pth")
test_save_pred()
