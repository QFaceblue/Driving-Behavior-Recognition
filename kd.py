from __future__ import print_function, division

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models

from ghost_net import ghost_net, ghost_net_Cifar
from utils import progress_bar
import matplotlib.pyplot as plt
import math

class MyDataset(Dataset):

    def __init__(self, names_file, transform=None):

        self.names_file = names_file
        self.transform = transform
        self.names_list = []

        if not os.path.isfile(self.names_file):
            print(self.names_file + 'does not exist!')
        with open(self.names_file, "r") as f:
            lists = f.readlines()
            for l in lists:
                self.names_list.append(l)

    def __len__(self):
        return len(self.names_list)

    def __getitem__(self, idx):
        image_path = self.names_list[idx].split(' ')[0]
        # print(image_path)
        if not os.path.isfile(image_path):
            print(image_path + 'does not exist!')
            return None
        image = Image.open(image_path).convert('RGB')  #
        if self.transform:
            image = self.transform(image)
        label = int(self.names_list[idx].split(' ')[1])

        sample = image, label

        return sample

train_transform = transforms.Compose([

    # transforms.ColorJitter(brightness=1, contrast=1, saturation=0.5, hue=0.5),
    # transforms.RandomRotation(10, resample=False, expand=False, center=None),
    # transforms.RandomResizedCrop((500,500)),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    # ToTensor()能够把灰度范围从0-255变换到0-1之间，
    # transform.Normalize()则把0-1变换到(-1,1).具体地说，对每个通道而言，Normalize执行以下操作：
    # image=(image-mean)/std
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([

    # transforms.RandomResizedCrop((500,500)),
    # transforms.CenterCrop((500,500)),
    # transforms.RandomHorizontalFlip(),
    # ToTensor()能够把灰度范围从0-255变换到0-1之间，
    # transform.Normalize()则把0-1变换到(-1,1).具体地说，对每个通道而言，Normalize执行以下操作：
    # image=(image-mean)/std
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# kaggle dataset
train_dataset = MyDataset("./data/dtrain.txt", train_transform)
val_dataset = MyDataset("./data/dval.txt", val_transform)
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=64,
                              shuffle=True,
                              num_workers=0)

val_dataloader = DataLoader(dataset=val_dataset,
                            batch_size=64,
                            shuffle=True,
                            num_workers=0)

def kdloss(outputs, teacher_outputs, labels, alpha=0.9, T=1.0):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    KD_loss = F.kl_div(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1), reduction='batchmean') * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss

def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs, dim=1)
    batch_size, class_num = outputs.shape
    onehot_targets = torch.zeros(batch_size, class_num).to(targets.device).scatter_(1, targets.view(batch_size, 1), 1)
    return -(log_softmax_outputs * onehot_targets).sum(dim=1).mean()

# Training
def train(teacher, student, opt, T=1., a=0.9, scheduler=None, stopearly=False):
    teacher.eval()
    student.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # print("inputs.shape",inputs.shape)
        with torch.no_grad():
            t_targets = teacher(inputs)
        opt.zero_grad()
        outputs = student(inputs)
        loss = kdloss(outputs, t_targets, targets, a, T)
        loss.backward()
        opt.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        average_loss = train_loss / (batch_idx + 1)
        train_acc = correct / total
        progress_bar(batch_idx, len(train_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (average_loss, 100. * train_acc, correct, total))
    if stopearly:
        scheduler.step(average_loss)
    else:
        scheduler.step()

    lr = opt.state_dict()['param_groups'][0]['lr']
    return average_loss, train_acc, lr


def val(net):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = CrossEntropy(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            average_loss = test_loss / (batch_idx + 1)
            test_acc = correct / total
            progress_bar(batch_idx, len(val_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (average_loss, 100. * test_acc, correct, total))

    return average_loss, test_acc


def lr_lambda(epoch):
    if epoch <= 32:
        return 0.01
    elif epoch <= 64:
        return 0.001
    else:
        return 0.0001

def change_lr2(epoch, T=20, factor=0.3, min=1e-4):
    mul = 1.
    if epoch < T :
        mul = mul
    elif epoch < T * 2:
        mul = mul * factor
    elif epoch < T * 3:
        mul = mul * factor * factor
    elif epoch <T * 4:
        mul = mul * factor * factor * factor
    elif epoch < T * 5:
        mul = mul * factor * factor * factor * factor
    else:
        return min
    # print(max((1 + math.cos(math.pi * (epoch % T) / T)) * mul/2, min))
    return max((1 + math.cos(math.pi * (epoch % T) / T)) * mul/2, min)

def change_lr3(epoch, T=15, factor=0.3, min=1e-4):
    mul = 1.
    if epoch < T * 3:
        mul = mul
    elif epoch < T * 7:
        mul = mul * factor
    elif epoch < T * 11:
        mul = mul * factor * factor
    else:
        return min
    # print(max((1 + math.cos(math.pi * epoch/ T)) * mul/2, min))
    return max((1 + math.cos(math.pi * epoch / T)) * mul / 2, min)

def change_lr4(epoch, T=15, factor=0.3, min=1e-4):
    mul = 1.
    if epoch < T*3:
        mul = mul
    elif epoch < T * 5:
        mul = mul * factor
    elif epoch < T * 7:
        mul = mul * factor * factor
    elif epoch < T * 9:
        mul = mul * factor * factor * factor
    else:
        return min
    # print(max((1 + math.cos(math.pi * epoch/ T)) * mul/2, min))
    return max((1 + math.cos(math.pi * epoch/ T)) * mul/2, min)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# device = 'cpu'

def kd_train():

    student = ghost_net.ghost_net_Cifar(width_mult=0.1)

    # teacher = ghost_net.ghost_net_Cifar(width_mult=1.0)
    # model_path = r"checkpoint/ghost_net/444/ghostnet_1_acc=91.6700.pth"

    teacher = SENet18()
    model_path = r"checkpoint/senet18/111/senet18_acc=94.8100.pth"
    # 多卡训练

    # 加载模型权重，忽略不同


    model_dict = teacher.state_dict()
    checkpoint = torch.load(model_path, map_location=device)
    pretrained_dict = checkpoint["net"]
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    teacher.load_state_dict(model_dict)
    # teacher.load_state_dict(pretrained_dict)
    print("loaded model with acc:{}".format(checkpoint["acc"]))

    epoches = 120
    best_acc = 0  # best test accuracy
    best_val_acc = 0  # best val accuracy

    # t = 20.0
    # a = 0.9
    # savepath = 'checkpoint/kd/cifar/ghost_net/3/000/'

    # t = 10.0
    # a = 0.9
    # savepath = 'checkpoint/kd/cifar/ghost_net/3/111/'

    # t = 5.0
    # a = 0.9
    # savepath = 'checkpoint/kd/cifar/ghost_net/3/222/'

    # t = 1.0
    # a = 0.9
    # savepath = 'checkpoint/kd/cifar/ghost_net/3/333/'

    # t = 1.0
    # a = 0.5
    # savepath = 'checkpoint/kd/cifar/ghost_net/3/444/'

    # t = 5.0
    # a = 0.5
    # savepath = 'checkpoint/kd/cifar/ghost_net/3/555/'

    # t = 10.0
    # a = 0.5
    # savepath = 'checkpoint/kd/cifar/ghost_net/3/666/'

    t = 20.0
    a = 0.5
    savepath = 'checkpoint/kd/cifar/ghost_net/3/777/'

    optimizer = optim.SGD(student.parameters(), lr=1e-1,
                          momentum=0.9, weight_decay=5e-4)
    stopearly = False
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_lr2)

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3,
    #                                                  verbose=True, threshold=1e-3, threshold_mode='rel',
    #                                                  cooldown=0, min_lr=1e-7, eps=1e-8)
    # stopearly =True
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=24, eta_min=1e-5)
    teacher.to(device)
    student.to(device)

    x = []
    lrs = []
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    start_time = time.time()
    for epoch in range(epoches):
        print('\nEpoch: %d' % epoch)
        train_l, train_a, lr = train(teacher, student, optimizer, t, a, scheduler, stopearly)
        test_l, test_a = val(student)
        if test_a > best_val_acc:
            print('Saving..')
            state = {
                'net': student.state_dict(),
                'acc': test_a,
                'epoch': epoch,
            }
            if not os.path.isdir(savepath):
                os.mkdir(savepath)
            print("best_acc:{:.4f}".format(test_a))
            torch.save(state, savepath + 'ghostcifar_01_kd_acc={:.4f}.pth'.format(test_a))
            best_val_acc = test_a
        x.append(epoch)
        lrs.append(lr)
        train_loss.append(train_l)
        test_loss.append(test_l)
        train_acc.append(train_a)
        test_acc.append(test_a)
        print("epoch={},lr={},train_loss={:.3f},test_loss={:.3f},train_acc={:.3f},test_acc={:.3f}"
              .format(epoch, lr, train_l, test_l, train_a, test_a))
        print("total train time ={}".format(format_time(time.time() - start_time)))
        # # earlystop
        if stopearly:
            if lr < 1e-4 - 1e-5:
                break

    fig = plt.figure(figsize=(16, 9))

    sub1 = fig.add_subplot(1, 3, 1)
    sub1.set_title("loss")
    sub1.plot(x, train_loss, label="train_loss")
    sub1.plot(x, test_loss, label="test_loss")
    plt.legend()

    sub2 = fig.add_subplot(1, 3, 2)
    sub2.set_title("acc")
    sub2.plot(x, train_acc, label="train_acc")
    sub2.plot(x, test_acc, label="test_acc")
    plt.legend()

    sub3 = fig.add_subplot(1, 3, 3)
    sub3.set_title("lr")
    sub3.plot(x, lrs, label="lr")
    plt.title(savepath)
    plt.legend()
    plt.savefig(savepath + 'learing.jpg')
    plt.show()


if __name__ == '__main__':
    kd_train()
