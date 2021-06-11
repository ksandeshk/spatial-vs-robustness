from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import MultiStepLR

#from models.wideresnet import *
#from models.resnet import *
from models.vgg import *
from trades import trades_loss, trades_loss_transform
import random

from advertorch.context import ctx_noparamgrad_and_eval

parser = argparse.ArgumentParser(description='PyTorch CIFAR CuSP Adversarial Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=120, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=8/255,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', default=5.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./saved_models/',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=20, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--train_angle', default=180, type=int)
parser.add_argument('--test_angle', default=0, type=int)
parser.add_argument('--translate', default=0.0, type=float)

args = parser.parse_args()
train_angle = args.train_angle
test_angle = args.test_angle
train_angle_30 = 30 
train_angle_60 = 60 
train_angle_90 = 90 
train_angle_120 = 120 
train_angle_150 = 150 
train_angle_180 = 180 

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_train_rot = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Lambda(lambda image: image.rotate(random.random() * train_angle * 2 - train_angle)),
    transforms.ToTensor(),
])

transform_train_rot_30 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Lambda(lambda image: image.rotate(random.random() * train_angle_30 * 2 - train_angle_30)),
    transforms.ToTensor(),
])

transform_train_rot_60 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Lambda(lambda image: image.rotate(random.random() * train_angle_60 * 2 - train_angle_60)),
    transforms.ToTensor(),
])

transform_train_rot_90 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Lambda(lambda image: image.rotate(random.random() * train_angle_90 * 2 - train_angle_90)),
    transforms.ToTensor(),
])

transform_train_rot_120 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Lambda(lambda image: image.rotate(random.random() * train_angle_120 * 2 - train_angle_120)),
    transforms.ToTensor(),
])

transform_train_rot_150 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Lambda(lambda image: image.rotate(random.random() * train_angle_150 * 2 - train_angle_150)),
    transforms.ToTensor(),
])

transform_train_rot_180 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Lambda(lambda image: image.rotate(random.random() * train_angle_180 * 2 - train_angle_180)),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, **kwargs)

trainset_transform = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_rot)
train_loader_transform = torch.utils.data.DataLoader(trainset_transform, batch_size=args.batch_size, shuffle=False, **kwargs)

trainset_transform_30 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_rot_30)
train_loader_transform_30 = torch.utils.data.DataLoader(trainset_transform_30, batch_size=args.batch_size, shuffle=False, **kwargs)

trainset_transform_60 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_rot_60)
train_loader_transform_60 = torch.utils.data.DataLoader(trainset_transform_60, batch_size=args.batch_size, shuffle=False, **kwargs)

trainset_transform_90 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_rot_90)
train_loader_transform_90 = torch.utils.data.DataLoader(trainset_transform_90, batch_size=args.batch_size, shuffle=False, **kwargs)

trainset_transform_120 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_rot_120)
train_loader_transform_120 = torch.utils.data.DataLoader(trainset_transform_120, batch_size=args.batch_size, shuffle=False, **kwargs)

trainset_transform_150 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_rot_150)
train_loader_transform_150 = torch.utils.data.DataLoader(trainset_transform_150, batch_size=args.batch_size, shuffle=False, **kwargs)

trainset_transform_180 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_rot_180)
train_loader_transform_180 = torch.utils.data.DataLoader(trainset_transform_180, batch_size=args.batch_size, shuffle=False, **kwargs)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss = trades_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def train_epsilon(args, model, device, train_loader, optimizer, epoch, eps):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss = trades_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=eps,
                           perturb_steps=args.num_steps,
                           beta=args.beta)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def train_transform(args, model, device, train_loader, train_loader_transform, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data_transform_iter = iter(train_loader_transform)
        data_transform,target_transform = next(data_transform_iter)
        data_transform,target_transform = data_transform.to(device), target_transform.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss = trades_loss_transform(model=model,
                           x_natural=data,
                           y=target,
                           x_transform=data_transform,
                           y_transform=target_transform,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def train_without_trades(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def train_with_pgd(args, model, device, train_loader, optimizer, epoch, eps):
    model.train()
    criterion = nn.CrossEntropyLoss()
    from advertorch.attacks import LinfPGDAttack
    adversary = LinfPGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
        nb_iter=10, eps_iter=2/255, rand_init=True, clip_min=0.0,
        clip_max=1.0, targeted=False)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        with ctx_noparamgrad_and_eval(model):
            data = adversary.perturb(data, target)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    model = VGG('VGG16').to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # CuSP with 75-90 learning rate schedule
        if epoch < 75:
            #train_epsilon(args, model, device, train_loader_transform_120, optimizer, epoch, eps=2/255)
            train_with_pgd(args, model, device, train_loader_transform_120, optimizer, epoch, eps=2/255)
        if epoch >= 75 and epoch < 90:
            #train_epsilon(args, model, device, train_loader_transform_120, optimizer, epoch, eps=4/255)
            train_with_pgd(args, model, device, train_loader_transform_150, optimizer, epoch, eps=4/255)
        if epoch >= 90:
            #train_epsilon(args, model, device, train_loader_transform_180, optimizer, epoch, eps=8/255)
            train_with_pgd(args, model, device, train_loader_transform_180, optimizer, epoch, eps=8/255)

        # evaluation on natural examples
        print('================================================================')
        eval_train(model, device, train_loader)
        eval_test(model, device, test_loader)
        print('================================================================')

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, "cifar10-vgg16-pgd-cusp-"+str(train_angle)+"-epoch{}.pt".format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, "cifar10-vgg16-opt-pgd-cusp-"+str(train_angle)+"-checkpoint_epoch{}.tar".format(epoch)))
            

if __name__ == '__main__':
    main()
