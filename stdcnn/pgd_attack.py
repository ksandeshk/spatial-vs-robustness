'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

import torchvision
import torchvision.transforms as transforms
from advertorch.context import ctx_noparamgrad_and_eval

import os
import argparse

from models import *
from utils import progress_bar
from torch.autograd import Variable
import random
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--train_angle', default=0, type=float)
parser.add_argument('--test_angle', default=0, type=int)
parser.add_argument('--translate', default=0.0, type=float)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
train_angle = args.train_angle
test_angle = args.test_angle
translate = args.translate

# Data
means = (0.4914, 0.4822, 0.4465)
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(means, (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(means, (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print('==> Building model..')
net = VGG('VGG16')

if use_cuda:
    net.cuda()
    #net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

def test_pgd_save_pred(epsilon=2/255):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    total_pred = torch.LongTensor()

    from advertorch.attacks import LinfPGDAttack
    adversary = LinfPGDAttack(
        net, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=epsilon,
        nb_iter=10, eps_iter=2/255, rand_init=True, clip_min=-1.0,
        clip_max=1.0, targeted=False)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        with ctx_noparamgrad_and_eval(net):
            inputs = adversary.perturb(inputs, targets)

        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total_pred = torch.cat((total_pred, predicted.cpu()), 0)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    np.save("./saved_data/stdcnn_vgg16_cifar10_"+str(translate)+"_trans_rot_"+str(epsilon)+"_test_pred.npy", total_pred.numpy())

net = torch.load("./saved_models/stdcnn_vgg16_cifar10_"+str(translate)+"_trans_rot.pth")
test_pgd_save_pred(2/255)
test_pgd_save_pred(4/255)
test_pgd_save_pred(8/255)

