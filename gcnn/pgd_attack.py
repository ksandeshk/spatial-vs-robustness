from time import time
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from utils import progress_bar
import argparse

from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.utils import to_var, pred_batch, test, \
    attack_over_test_data

from models.resnet import *
import random

use_cuda = torch.cuda.is_available()

# Hyper-parameters
param = {
    'test_batch_size': 100,
    'epsilon': 8/255,
}

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--translate', default=0.0, type=float)
args = parser.parse_args()
trans = args.translate

means = (0.4914, 0.4822, 0.4465)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(means, (0.2023, 0.1994, 0.2010)),
])

# Data loaders
test_dataset = datasets.CIFAR10(root='../data/', train=False, download=True, transform=transform_test)
loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=param['test_batch_size'], shuffle=False)

# Setup model to be attacked
net = ResNet18()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

net = torch.load("./saved_models/gcnn_resnet18_cifar10_"+str(trans)+"_trans_rot.pth")

if torch.cuda.is_available():
    print('CUDA ensabled.')
    net.cuda()

# Adversarial attack
adversary = LinfPGDAttack(net, epsilon=2/255, k=10, a=2/255, random_start=True)
t0 = time()
attack_over_test_data(net, adversary, param, loader_test) 
print('{}s eclipsed.'.format(time()-t0))

adversary = LinfPGDAttack(net, epsilon=4/255, k=10, a=2/255, random_start=True)
t0 = time()
attack_over_test_data(net, adversary, param, loader_test) 
print('{}s eclipsed.'.format(time()-t0))

adversary = LinfPGDAttack(net, epsilon=8/255, k=10, a=2/255, random_start=True)
t0 = time()
attack_over_test_data(net, adversary, param, loader_test) 
print('{}s eclipsed.'.format(time()-t0))
