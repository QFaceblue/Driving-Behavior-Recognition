from __future__ import print_function, division
import torch
from torch.nn import init
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import math
import copy
# from tensorboardX import SummaryWriter
from utils import progress_bar, format_time
import json
from PIL import Image
from efficientnet_pytorch import EfficientNet
from ghostnet import ghostnet
from mnext import mnext
from mobilenetv3 import MobileNetV3, mobilenetv3_s
from mobilenetv3_2 import MobileNetV3_Small, MobileNetV3_Large
from mobilenet import my_mobilenext, my_mobilenext_2
from mobilenetv3_torch import mobilenet_v3_large, mobilenet_v3_small
import onnxruntime
import cv2
import json
import pandas as pd
from mobilenetv2_cbam import MobileNetV2_cbam, MobileNetV2_s, MobileNetV2, MobileNetV2_s2, \
    MobileNetV2_s3, MobileNetV2_s4, MobileNetV2_sgb, MobileNetV2_sgb_sa, MobileNetV2_sgb_sa2


def softmax_np(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
    softmax = x_exp / x_exp_row_sum
    return softmax


def softmax_flatten(x):
    x = x.flatten()
    x_row_max = x.max()
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum()
    softmax = x_exp / x_exp_row_sum
    return softmax


def imshow(inp, title=None):
    """Imshow for Tensor."""
    # 先把tensor转为numpy,然后将通道维放到最后方便广播
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    # 当inp为0-1之间的浮点数和0-255之间的整数都能显示成功
    # inp = np.clip(inp, 0, 1)*255
    # inp = inp.astype(np.int32)
    # print(inp)
    plt.imshow(inp)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    if title is not None:
        plt.title(title)
    # plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()


def showimage(dataloader, class_names):
    # 获取一批训练数据
    inputs, classes = next(iter(dataloader))
    # 批量制作网格 Make a grid of images
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])


def weightInit(model):
    # weight initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)
    return model


def weightInit_Glorot(model):
    # weight initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)
    return model


class MyDataset(Dataset):

    def __init__(self, names_file, transform=None):

        self.names_file = names_file
        self.transform = transform
        self.names_list = []

        if not os.path.isfile(self.names_file):
            print(self.names_file + 'does not exist!')
        with open(self.names_file, "r", encoding="utf-8") as f:
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
# aug5
# train_transform = transforms.Compose([
#     transforms.Resize((160, 160)),
#     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
#     transforms.RandomRotation(10, resample=False, expand=False, center=None),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])
# # #aug4
# train_transform = transforms.Compose([
#     transforms.Resize((240, 240)),
#     transforms.RandomCrop(224),
#     # transforms.Resize((224, 224)),
#     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
#     transforms.RandomRotation(10, resample=False, expand=False, center=None),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])
# aug3
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    transforms.RandomRotation(10, resample=False, expand=False, center=None),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
# # aug2
# train_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])
# # aug1
# train_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# train_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
best_acc = 0  # best test accuracy
best_val_acc = 0

# kaggle dataset
# num_classes = 10
# label_name = ["正常","右持手机","右接电话","左持手机","左接电话","操作仪器","喝水","向后侧身","整理仪容","侧视"]
# label_name = ["normal", "texting-R", "answering_R", "texting-L", "answering_L", "operating", "drinking", "leaning_back", "makeup", "side_view"]
# # 新的类别
# # # label_name = ["正常","右接电话",左接电话","低头","操作仪器","喝水","吸烟","向后侧身","整理仪容","侧视"]
# # label_name =["normal", "right to answer the phone", left to answer the phone "," head down "," operating instruments "," drinking water "," smoking "," leaning back ","makeup "," side view "]

# num_classes = 100
# num_classes = 10
num_classes = 9
# num_classes = 8
# num_classes = 7
# num_classes = 6
# net = models.resnet18(pretrained=True)
# # net = models.resnext50_32x4d(pretrained=False,num_classes=num_classes)
# net = models.resnet50(pretrained=True)
# # # net = models.resnext50_32x4d(pretrained=True)
# # # net = models.resnext101_32x8d(pretrained=True)
# #
# num_in = net.fc.in_features
# net.fc = nn.Linear(num_in, num_classes)

# net = ghostnet(num_classes=num_classes)
# net = mnext(num_classes=num_classes)

# net = mobilenet_v3_small(pretrained=True)
# net = mobilenet_v3_large(pretrained=True)
# net = mobilenet_v3_small(pretrained=False)
# net = mobilenet_v3_large(pretrained=False)
# num_in = net.classifier[3].in_features
# net.classifier[3] = nn.Linear(num_in, num_classes)

# net = MobileNetV2_s(num_classes=num_classes)
# net = MobileNetV2_s2(num_classes=num_classes)
# net = MobileNetV2_s3(num_classes=num_classes)
net = MobileNetV2_s4(num_classes=num_classes)
# net = MobileNetV2_sgb(num_classes=num_classes)
# net = MobileNetV2_sgb_sa(num_classes=num_classes)
# net = MobileNetV2_sgb_sa2(num_classes=num_classes)

# net = models.mobilenet_v2(pretrained=False, width_mult=1.0, num_classes=num_classes)

# net = models.mobilenet_v2(pretrained=True, width_mult=1.0)
# num_in = net.classifier[1].in_features
# net.classifier[1] = nn.Linear(num_in, num_classes)

# net = MobileNetV2_cbam(num_classes=num_classes, width_mult=1.0, add_location=(16,), ca=False, sa=True)
# net = MobileNetV2_cbam(num_classes=num_classes, width_mult=1.0, add_location=(24,), ca=False, sa=True)
# net = MobileNetV2_cbam(num_classes=num_classes, width_mult=1.0, add_location=(32,), ca=False, sa=True)
# net = MobileNetV2_cbam(num_classes=num_classes, width_mult=1.0, add_location=(64,), ca=False, sa=True)
# net = MobileNetV2_cbam(num_classes=num_classes, width_mult=1.0, add_location=(96,), ca=False, sa=True)
# net = MobileNetV2_cbam(num_classes=num_classes, width_mult=1.0, add_location=(160,), ca=False, sa=True)
# net = MobileNetV2_cbam(num_classes=num_classes, width_mult=1.0, add_location=(96, 160), ca=False, sa=True)
# net = MobileNetV2_cbam(num_classes=num_classes, width_mult=1.0, add_location=(64, 96, 160), ca=False, sa=True)

# net = MobileNetV2_cbam(num_classes=num_classes, width_mult=1.0, add_location=(16,), ca=True, sa=False)
# net = MobileNetV2_cbam(num_classes=num_classes, width_mult=1.0, add_location=(24,), ca=True, sa=False)
# net = MobileNetV2_cbam(num_classes=num_classes, width_mult=1.0, add_location=(64,), ca=True, sa=False)
# net = MobileNetV2_cbam(num_classes=num_classes, width_mult=1.0, add_location=(96,), ca=True, sa=False)
# net = MobileNetV2_cbam(num_classes=num_classes, width_mult=1.0, add_location=(160,), ca=True, sa=False)
# net = MobileNetV2_cbam(num_classes=num_classes, width_mult=1.0, add_location=(96, 160), ca=True, sa=False)

# net = MobileNetV2_cbam(num_classes=num_classes, width_mult=1.0, add_location=(24,))
# net = MobileNetV2_cbam(num_classes=num_classes, width_mult=1.0, add_location=(32,))
# net = MobileNetV2_cbam(num_classes=num_classes, width_mult=1.0, add_location=(64,))
# net = MobileNetV2_cbam(num_classes=num_classes, width_mult=1.0, add_location=(96,))
# net = MobileNetV2_cbam(num_classes=num_classes, width_mult=1.0, add_location=(160,))
# net = MobileNetV2_cbam(num_classes=num_classes, width_mult=1.0, add_location=(96, 160))
# net = MobileNetV2_cbam(num_classes=num_classes, width_mult=1.0, add_location=(64, 96, 160))
# net = MobileNetV2_cbam(num_classes=num_classes, width_mult=1.0, add_location=(96, 160))

# num_in = net.classifier[1].in_features
# net.classifier[1] = nn.Linear(num_in, num_classes)
# net = mobilenet_v3_small(pretrained=True)
# # net = mobilenet_v3_large(pretrained=True)
# num_in = net.classifier[3].in_features
# net.classifier[3] = nn.Linear(num_in, num_classes)

# net = models.resnext50_32x4d(pretrained=True)
# num_in = net.fc.in_features
# net.fc = nn.Linear(num_in, num_classes)


# # 加载模型权重，忽略不同
# model_path = r"checkpoint/data_12_23/mobilenetv2/000/mobilenetv2_1_12_23_acc=92.1389.pth"
# model_dict =net.state_dict()
# checkpoint = torch.load(model_path, map_location=device)
# pretrained_dict = checkpoint["net"]
# # pretrained_dict = checkpoint
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
# model_dict.update(pretrained_dict)
# net.load_state_dict(model_dict)

# 更新权重
model_path = r"weights/mobilenet_v2-b0353104.pth"
# model_path = r"weights/ghostnet_73.98.pth"
# model_path = r"weights/mnext.pth.tar"
# model_path = r"checkpoint/paper_test/ours/mobilenetv2/augment/444/mobilenetv2_acc=85.2035.pth"
model_dict = net.state_dict()
checkpoint = torch.load(model_path, map_location=device)
pretrained_dict = checkpoint
pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                   k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v)}
model_dict.update(pretrained_dict)
net.load_state_dict(model_dict)


# net = MobileNetV2_cbam(num_classes=num_classes, width_mult=1.0, add_location=16)
#
# # 更新权重 add_location=16
# model_path = r"weights/mobilenet_v2-b0353104.pth"
# model_dict =net.state_dict()
# checkpoint = torch.load(model_path, map_location=device)
# pretrained_dict = checkpoint
# new_key = list(model_dict.keys())
# pre_key = list(pretrained_dict.keys())
# ignore_num = 3
# start_index = new_key.index('features.2.fc1.weight')
# print(new_key[start_index+3], pre_key[start_index])
# for i in range(len(pre_key)):
#     if i<start_index:
#         j = i
#     else:
#         j = i+3
#     if np.shape(model_dict[new_key[j]]) == np.shape(pretrained_dict[pre_key[i]]):
#         model_dict[new_key[j]] = pretrained_dict[pre_key[i]]
# net.load_state_dict(model_dict)

# net = MobileNetV2_cbam(num_classes=num_classes, width_mult=1.0, add_location=64)
#
# # 更新权重 add_location=64
# model_path = r"weights/mobilenet_v2-b0353104.pth"
# model_dict =net.state_dict()
# checkpoint = torch.load(model_path, map_location=device)
# pretrained_dict = checkpoint
# new_key = list(model_dict.keys())
# pre_key = list(pretrained_dict.keys())
# ignore_num = 3
# start_index = new_key.index('features.11.fc1.weight')
# print(new_key[start_index+3], pre_key[start_index])
# for i in range(len(pre_key)):
#     if i<start_index:
#         j = i
#     else:
#         j = i+3
#     if np.shape(model_dict[new_key[j]]) == np.shape(pretrained_dict[pre_key[i]]):
#         model_dict[new_key[j]] = pretrained_dict[pre_key[i]]
# net.load_state_dict(model_dict)

# net = models.shufflenet_v2_x1_0(pretrained=True)
# # net = models.shufflenet_v2_x0_5(pretrained=True)
# # net = models.resnet50(pretrained=True)w
# num_in = net.fc.in_features
# net.fc = nn.Linear(num_in, num_classes)

# net = ghost_net_Cifar(num_classes=num_classes, width_mult=0.1)

# net = ghost_net(num_classes=num_classes, width_mult=1.)
# net = ghost_net(num_classes=num_classes, width_mult=0.5)
# net = ghost_net(num_classes=num_classes, width_mult=0.3)
# net = ghost_net(num_classes=num_classes, width_mult=0.1)

# net = ghostnet(num_classes=num_classes, width=1.)
# net = ghostnet(num_classes=num_classes, width=0.5)
# net = ghostnet(num_classes=num_classes, width=0.3)
# net = ghostnet(num_classes=num_classes, width=0.1)

# net = mnext(num_classes=num_classes, width_mult=1.)
# net = mnext(num_classes=num_classes, width_mult=0.5)


# num_in = net.fc.in_features
# # 创建的层默认参数需要训练
# net.fc = nn.Linear(num_in, num_classes)

# # 加载模型权重，忽略不同
# model_path = r"weights/mnext.pth.tar"
# model_dict =net.state_dict()
# checkpoint = torch.load(model_path, map_location=device)
# # pretrained_dict = checkpoint["net"]
# pretrained_dict = checkpoint
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
# model_dict.update(pretrained_dict)
# net.load_state_dict(model_dict)
# # print("loaded model with acc:{}".format(checkpoint["acc"]))


def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs, dim=1)
    batch_size, class_num = outputs.shape
    onehot_targets = torch.zeros(batch_size, class_num).to(targets.device).scatter_(1, targets.view(batch_size, 1), 1)
    return -(log_softmax_outputs * onehot_targets).sum(dim=1).mean()


def CrossEntropy_KD(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs, dim=1)
    softmax_targets = F.softmax(targets, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()


def change_lr1(epoch, T=5, factor=0.3, min=1e-3):
    mul = 1.
    if epoch < T:
        mul = mul
    elif epoch < T * 3:
        mul = mul * factor
    else:
        return min
    return max((1 + math.cos(math.pi * epoch / T)) * mul / 2, min)


def change_lr2(epoch, T=7, factor=0.3, min=1e-3):
    mul = 1.
    if epoch < T:
        mul = mul
    elif epoch < T * 3:
        mul = mul * factor
    elif epoch < T * 5:
        mul = mul * factor * factor
    else:
        return min
    return max((1 + math.cos(math.pi * epoch / T)) * mul / 2, min)


def change_lr3(epoch, T=9, factor=0.3, min=1e-3):
    mul = 1.
    if epoch < T:
        mul = mul
    elif epoch < T * 3:
        mul = mul * factor
    else:
        return min
    return max((1 + math.cos(math.pi * epoch / T)) * mul / 2, min)


def change_lr4(epoch, T=7, factor=0.3, min=1e-3):
    mul = 1.
    if epoch < T:
        mul = mul
    elif epoch < T * 3:
        mul = mul * factor
    else:
        return min
    return max((1 + math.cos(math.pi * epoch / T)) * mul / 2, min)


def change_lr5(epoch, T=9, factor=0.3, min=1e-3):
    mul = 1.
    if epoch < T:
        mul = mul
    elif epoch < T * 3:
        mul = mul * factor
    elif epoch < T * 5:
        mul = mul * factor * factor
    else:
        return min
    return max((1 + math.cos(math.pi * epoch / T)) * mul / 2, min)


def change_lr6(epoch, T=5, factor=1, min=1e-3):
    mul = 1.
    if epoch < T:
        mul = mul
    elif epoch < T * 3:
        mul = mul * factor
    else:
        return min
    return max((1 + math.cos(math.pi * epoch / T)) * mul / 2, min)


def change_lr_d1(epoch, T=5, factor=0.3, min=1e-3):
    mul = 1.
    n = epoch / T
    while n > 1:
        mul *= factor
        n -= 1
    return max((1 + math.cos(math.pi * (epoch % T) / T)) * mul / 2, min)


def change_lr_d2(epoch, T=8, factor=0.3, min=1e-3):
    mul = 1.
    n = epoch / T
    while n > 1:
        mul *= factor
        n -= 1
    return max((1 + math.cos(math.pi * (epoch % T) / T)) * mul / 2, min)

def change_0(epoch):
    return 0.1


def change_1(epoch, T=9, factor=0.1, min=1e-3):
    mul = 1.
    n = epoch / T
    while n > 1:
        mul *= factor
        n -= 1
    return max(mul, min)


def change_2(epoch, T=9, min=1e-3):
    return max((1 + math.cos(math.pi * epoch / T)) / 2, min)


def change_3(epoch, T=9, factor=0.3, min=1e-3):
    mul = 1.
    n = (epoch - T) / (2 * T)
    while n > 0:
        mul *= factor
        n -= 1
    return max((1 + math.cos(math.pi * epoch / T)) * mul / 2, min)

criterion = nn.CrossEntropyLoss()
# criterion = CrossEntropy

# epoches = 10
# optimizer = optim.Adam(net.parameters(), lr=1e-3)
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_lr_d1)

# epoches = 24
# optimizer = optim.Adam(net.parameters(), lr=1e-3)
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_lr_d2)

# epoches = 16
# optimizer = optim.Adam(net.parameters(), lr=1e-3)
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_lr1)

# lr
# epoches = 16
# # optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0, dampening=0, weight_decay=0)
# # optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0, dampening=0, weight_decay=0)
# # optimizer = optim.RMSprop(net.parameters(), lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)
# # optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
# optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-6)
# # optimizer = optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
# # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.3)
# # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_lr6)
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_lr1)

# epoches = 25
# optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-6)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 100, 150], gamma=0.1)

# epoches = 25
# optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-6)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 150], gamma=0.1)

# AUC
#
# 陈鹏
# lr=1e-4 epoches = 20 batch=16 SGD CrossEntropy
# epoches = 20
# optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0, dampening=0, weight_decay=0)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30, 40], gamma=0.1)
# MobileVGG ref57
# 训练使用随机梯度下降进行，学习率为 0.0001，衰减率为 10−6和动量值 0.9。批大小和纪元数分别设置为 64 和 100。
# epoches = 100
# optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9, dampening=0, weight_decay=1e-6)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 130, 140], gamma=0.1)

# MobileVGG
# 使用 Glorot 正常初始值设定项初始化权重。
# Adam 优化器的学习速率为 0.0001，衰减值为 0.000001，beta1 = 0.9，beta2 = 0.999。
# epoch 和批大小的数量分别设置为 500 和 32

# epoches = 500 # nopre
# epoches = 100 # nopre
# epoches = 50
# optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-6)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 1500], gamma=0.1)

epoches = 27
optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-6)
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_0)
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_1)
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_2)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_3)

# epoches = 27
# optimizer = optim.Adam(net.parameters(), lr=1e-2)
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_lr3)

# epoches = 21
# optimizer = optim.Adam(net.parameters(), lr=1e-2)
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_lr4)

# epoches = 36
# optimizer = optim.Adam(net.parameters(), lr=1e-2)
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_lr2)

# epoches = 36
# optimizer = optim.Adam(net.parameters(), lr=1e-2)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[9, 18, 27], gamma=0.3)

# epoches = 45
# optimizer = optim.Adam(net.parameters(), lr=1e-2)
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_lr5)

net.to(device)

# # all class9_crop
# train_dataset = MyDataset("data/txt_raw_crop/total_train_crop.txt", train_transform)
# val_dataset = MyDataset("data/txt_raw_crop/total_test_crop.txt", val_transform)

# # all class9
# train_dataset = MyDataset("data/txt_raw/total_train.txt", train_transform)
# val_dataset = MyDataset("data/txt_raw/total_test.txt", val_transform)

# all class8_crop
# train_dataset = MyDataset("data/txt_raw_crop/total_train_crop_8.txt", train_transform)
# val_dataset = MyDataset("data/txt_raw_crop/total_test_crop_8.txt", val_transform)

# # # all class8_crop new
# train_dataset = MyDataset("data/txt_raw_crop/total_train_crop_8_n.txt", train_transform)
# val_dataset = MyDataset("data/txt_raw_crop/total_test_crop_8_n.txt", val_transform)

# all class8
# train_dataset = MyDataset("data/txt_raw/total_train_8.txt", train_transform)
# val_dataset = MyDataset("data/txt_raw/total_test_8.txt", val_transform)

# # mix
# # all class9
# train_dataset = MyDataset("data/txt_raw/total_mix_train.txt", train_transform)
# val_dataset = MyDataset("data/txt_raw/total_mix_test.txt", val_transform)
# # all class8
# train_dataset = MyDataset("data/txt_raw/total_mix_train_8.txt", train_transform)
# val_dataset = MyDataset("data/txt_raw/total_mix_test_8.txt", val_transform)
# # all class9 crop
# train_dataset = MyDataset("data/txt_raw_crop/total_mix_train_crop.txt", train_transform)
# val_dataset = MyDataset("data/txt_raw_crop/total_mix_test_crop.txt", val_transform)
# # all class8 crop
# train_dataset = MyDataset("data/txt_raw_crop/total_mix_train_crop_8.txt", train_transform)
# val_dataset = MyDataset("data/txt_raw_crop/total_mix_test_crop_8.txt", val_transform)

# # AUCv1
# train_dataset = MyDataset("data/auc/auc_train_val_v1.txt", train_transform)
# val_dataset = MyDataset("data/auc/auc_test_v1.txt", val_transform)

# AUCv2
# train_dataset = MyDataset("data/auc2/auc2_trainVal.txt", train_transform)
# val_dataset = MyDataset("data/auc2/auc2_test.txt", val_transform)

# # stateFarm
# train_dataset = MyDataset("data/stateFarm/stateFarm_train.txt", train_transform)
# val_dataset = MyDataset("data/stateFarm/stateFarm_test.txt", val_transform)

# ours
# train_dataset = MyDataset("data/ours/224/train224.txt", train_transform)
# val_dataset = MyDataset("data/ours/224/test224.txt", val_transform)

# train_dataset = MyDataset("data/ours/224/train224_8.txt", train_transform)
# val_dataset = MyDataset("data/ours/224/test224_8.txt", val_transform)

# train_dataset = MyDataset("data/ours/224/train_crop224.txt", train_transform)
# val_dataset = MyDataset("data/ours/224/test_crop224.txt", val_transform)

# train_dataset = MyDataset("data/ours/224/train_crop224_8.txt", train_transform)
# val_dataset = MyDataset("data/ours/224/test_crop224_8.txt", val_transform)

#
# train_dataset = MyDataset("data/ours/224/mix_train224.txt", train_transform)
# val_dataset = MyDataset("data/ours/224/mix_test224.txt", val_transform)
#
# train_dataset = MyDataset("data/ours/224/mix_train_crop224.txt", train_transform)
# val_dataset = MyDataset("data/ours/224/mix_test_crop224.txt", val_transform)

# # bus
# train_dataset = MyDataset("data/bus/train.txt", train_transform)
# val_dataset = MyDataset("data/bus/test.txt", val_transform)
# train_dataset = MyDataset("data/bus/addcar_train224.txt", train_transform)
# val_dataset = MyDataset("data/bus/addcar_test224.txt", val_transform)
# train_dataset = MyDataset("data/bus/addcar_train224_8.txt", train_transform)
# val_dataset = MyDataset("data/bus/addcar_test224_8.txt", val_transform)

# class6

# train_dataset = MyDataset("data/ours/224/train224_6.txt", train_transform)
# val_dataset = MyDataset("data/ours/224/test224_6.txt", val_transform)
train_dataset = MyDataset("data/ours/224/train_crop224_6.txt", train_transform)
val_dataset = MyDataset("data/ours/224/test_crop224_6.txt", val_transform)

start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# batch_size = 64
batch_size = 32
# batch_size = 16
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)
val_dataloader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)


# Training
def train(epoch):
    print('\nEpoch: %d' % (epoch + 1))
    net.train()
    global best_acc
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # print("inputs.shape",inputs.shape)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        average_loss = train_loss / (batch_idx + 1)
        train_acc = correct / total
        progress_bar(batch_idx, len(train_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (average_loss, 100. * train_acc, correct, total))
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    scheduler.step()
    # scheduler.step(average_loss)

    return average_loss, train_acc, lr

    # # Save checkpoint.
    # acc = 100.*correct/total
    # if acc > best_acc:
    #     print('Saving..')
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint/B0'):
    #         os.mkdir('checkpoint/B0')
    #     torch.save(state, './checkpoint/B0/111/B0_acc={:.4f}.pth'.format(acc))
    #     best_acc = acc


# Save checkpoint.
# class9

# class8

# # change_lr2 nopre crop 240-》224
# savepath = 'checkpoint/paper_test/mobilenetv2/nopre/crop/000/'
# # MultiStepLR nopre crop 240-》224
# savepath = 'checkpoint/paper_test/mobilenetv2/nopre/crop/111/'
# # change_lr_d2 nopre crop 240-》224
# savepath = 'checkpoint/paper_test/mobilenetv2/nopre/crop/222/'
# # change_lr3 nopre crop 240-》224
# savepath = 'checkpoint/paper_test/mobilenetv2/nopre/crop/333/'
# # change_lr4 nopre crop 240-》224
# savepath = 'checkpoint/paper_test/mobilenetv2/nopre/crop/444/'

# # change_lr1 pre crop 240-》224
# savepath = 'checkpoint/paper_test/mobilenetv2/pre/crop/000/'
# # change_lr1 pre crop 240-》224 new
# savepath = 'checkpoint/paper_test/mobilenetv2/pre/crop/111/'
# # change_lr1 pre crop 240-》224 new
# savepath = 'checkpoint/paper_test/mobilenetv2/pre/crop/222/'
# # change_lr1 pre crop 240-》224
# savepath = 'checkpoint/paper_test/mobilenetv2/pre/crop/333/'
# adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150]
# savepath = 'checkpoint/paper_test/mobilenetv2/pre/crop/444/'

# # change_lr1 pre crop 240-》224
# savepath = 'checkpoint/paper_test/mobilenetv2/pre/normal/000/'
# # change_lr2 nopre crop 240-》224
# savepath = 'checkpoint/paper_test/mobilenetv2/pre/normal/111/'

# Mobilenetv2_cbam
# # change_lr2 nopre crop 240-》224 location=16
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/nopre/crop/000/'
# # change_lr2 nopre crop 240-》224 location=24
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/nopre/crop/111/'
# # change_lr2 nopre crop 240-》224 location=32
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/nopre/crop/222/'
# # change_lr2 nopre crop 240-》224 location=64
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/nopre/crop/333/'
# # change_lr2 nopre crop 240-》224 location=96
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/nopre/crop/444/'
# # change_lr2 nopre crop 240-》224 location=160
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/nopre/crop/555/'

# # change_lr1 nopre crop 240-》224 location=160
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/pre/crop/000/'
# # change_lr1 nopre crop 240-》224 location=96
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/pre/crop/111/'
# # change_lr1 nopre crop 240-》224 location=64
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/pre/crop/222/'
# # change_lr1 nopre crop 240-》224 location=32
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/pre/crop/333/'
# # change_lr1 nopre crop 240-》224 location=96&160
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/pre/crop/444/'
# # change_lr1 nopre crop 240-》224 location=64&96&160
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/pre/crop/555/'
# # change_lr1 nopre crop 240-》224 location=96&160 new
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/pre/crop/666/'

# Mobilenetv2_cbam sa
# # change_lr4 nopre crop 240-》224 location=16
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/nopre/crop/sa/000/'
# # change_lr4 nopre crop 240-》224 location=24
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/nopre/crop/sa/111/'
# # change_lr4 nopre crop 240-》224 location=32
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/nopre/crop/sa/222/'
# # change_lr4 nopre crop 240-》224 location=64
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/nopre/crop/sa/333/'
# # change_lr4 nopre crop 240-》224 location=96
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/nopre/crop/sa/444/'
# # change_lr4 nopre crop 240-》224 location=160
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/nopre/crop/sa/555/'
# # change_lr5 nopre crop 240-》224 location=160
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/nopre/crop/sa/666/'

# # change_lr3 pre crop 240-》224 location=160
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/pre/crop/sa/000/'
# # change_lr1 pre crop 240-》224 location=160
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/pre/crop/sa/111/'
# # change_lr1 pre crop 240-》224 location=96
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/pre/crop/sa/222/'
# # change_lr1 pre crop 240-》224 location=64
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/pre/crop/sa/333/'
# # change_lr1 pre crop 240-》224 location=32
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/pre/crop/sa/444/'
# # change_lr1 pre crop 240-》224 location=96&160
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/pre/crop/sa/555/'
# # change_lr1 pre crop 240-》224 location=96&160
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/pre/crop/sa/5551/'
# # change_lr6 pre crop 240-》224 location=96&160
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/pre/crop/sa/5552/'

# # change_lr1 pre crop 240-》224 location=64&96&160
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/pre/crop/sa/666/'
# # change_lr1 pre crop 240-》224 location=96&160 new
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/pre/crop/sa/777/'

# # Mobilenetv2_cbam ca
# # change_lr4 nopre crop 240-》224 location=16
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/nopre/crop/ca/000/'
# # change_lr4 nopre crop 240-》224 location=24
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/nopre/crop/ca/111/'
# # change_lr4 nopre crop 240-》224 location=32
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/nopre/crop/ca/222/'
# # change_lr4 nopre crop 240-》224 location=64
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/nopre/crop/ca/333/'
# # change_lr4 nopre crop 240-》224 location=96
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/nopre/crop/ca/444/'
# # change_lr4 nopre crop 240-》224 location=160
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/nopre/crop/ca/555/'
# # change_lr5 nopre crop 240-》224 location=160
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/nopre/crop/ca/666/'

# # change_lr3 pre crop 240-》224 location=160
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/pre/crop/ca/000/'
# # change_lr1 pre crop 240-》224 location=96
# savepath = 'checkpoint/paper_test/mobilenetv2_cbam/pre/crop/ca/111/'

# # change_lr1 pre crop 240-》224
# savepath = 'checkpoint/paper_test/resnext50/pre/crop/000/'
# # change_lr1 pre crop 240-》224
# savepath = 'checkpoint/paper_test/resnext101/pre/crop/000/'
# # change_lr1 pre crop 240-》224
# savepath = 'checkpoint/paper_test/resnet50/pre/crop/000/'
# # change_lr1 pre crop 240-》224
# savepath = 'checkpoint/paper_test/resnet18/pre/crop/000/'
# # change_lr1 pre crop 240-》224
# savepath = 'checkpoint/paper_test/mobilenetv_small/pre/crop/000/'
# # change_lr1 pre crop 240-》224
# savepath = 'checkpoint/paper_test/mobilenetv_small/pre/crop/111/'
# # change_lr1 pre  240-》224
# savepath = 'checkpoint/paper_test/mobilenetv_small/pre/000/'
# # change_lr1 pre  240-》224
# savepath = 'checkpoint/paper_test/mobilenetv_small/pre/111/'
# # change_lr1 pre crop 240-》224
# savepath = 'checkpoint/paper_test/mobilenetv_large/pre/crop/000/'
# # change_lr1 pre crop 240-》224
# savepath = 'checkpoint/paper_test/mobilenetv_large/pre/crop/111/'
# # change_lr1 pre  240-》224
# savepath = 'checkpoint/paper_test/mobilenetv_large/pre/000/'
# # change_lr1 pre  240-》224
# savepath = 'checkpoint/paper_test/mobilenetv_large/pre/111/'

# # change_lr1 pre  240-》224
# savepath = 'checkpoint/paper_test/ghostnet/pre/000/'

# mobilentv2 crop lr
# # MultiStepLR  adam pre crop 240-》224
# savepath = 'checkpoint/paper_test/mobilenetv2/pre/crop/lr/000/'
# # change_lr6  adam pre crop 240-》224
# savepath = 'checkpoint/paper_test/mobilenetv2/pre/crop/lr/111/'
# # change_lr1  adam pre crop 240-》224
# savepath = 'checkpoint/paper_test/mobilenetv2/pre/crop/lr/222/'
# # change_lr1  adam pre crop 240-》224 lr=1e-2
# savepath = 'checkpoint/paper_test/mobilenetv2/pre/crop/lr/333/'
# SGD
# # MultiStepLR  SGD pre crop 240-》224
# savepath = 'checkpoint/paper_test/mobilenetv2/pre/crop/lr/sgd/000/'
# # change_lr6  SGD pre crop 240-》224
# savepath = 'checkpoint/paper_test/mobilenetv2/pre/crop/lr/sgd/111/'
# # change_lr1  SGD pre crop 240-》224
# savepath = 'checkpoint/paper_test/mobilenetv2/pre/crop/lr/sgd/222/'
# # change_lr1  SGD pre crop 240-》224 lr=1e-2
# savepath = 'checkpoint/paper_test/mobilenetv2/pre/crop/lr/sgd/333/'

# rms
# # MultiStepLR  SGD pre crop 240-》224 lr=1e-2
# savepath = 'checkpoint/paper_test/mobilenetv2/pre/crop/lr/rms/000/'
# # change_lr6  SGD pre crop 240-》224 lr=1e-2
# savepath = 'checkpoint/paper_test/mobilenetv2/pre/crop/lr/rms/111/'
# # change_lr1  SGD pre crop 240-》224 lr=1e-2
# savepath = 'checkpoint/paper_test/mobilenetv2/pre/crop/lr/rms/222/'
# # change_lr1  SGD pre crop 240-》224 lr=1e-2
# savepath = 'checkpoint/paper_test/mobilenetv2/pre/crop/lr/rms/333/'

# mix
# mobilenetv2
# # change_lr1  adam pre crop 240-》224
# savepath = 'checkpoint/paper_test/mix/mobilenetv2/class9/normal/000/'
# # change_lr1  adam pre crop 240-》224
# savepath = 'checkpoint/paper_test/mix/mobilenetv2/class9/crop/000/'
# resnext50
# # change_lr1  adam pre crop 240-》224
# savepath = 'checkpoint/paper_test/mix/resnext50/class9/crop/000/'

# AUCv1
# adam lr=1e-4, weight_decay=1e-650 pre noaug
# savepath = 'checkpoint/paper_test/AUCV1/mobilenetv2/augment/000/'
# # adam lr=1e-4, weight_decay=1e-650 pre aug1
# savepath = 'checkpoint/paper_test/AUCV1/mobilenetv2/augment/111/'
# # adam lr=1e-4, weight_decay=1e-650 pre aug2
# savepath = 'checkpoint/paper_test/AUCV1/mobilenetv2/augment/222/'
# # adam lr=1e-4, weight_decay=1e-650 pre aug3
# savepath = 'checkpoint/paper_test/AUCV1/mobilenetv2/augment/333/'
# # adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150]
# savepath = 'checkpoint/paper_test/AUCV1/mobilenetv2/augment/444/'

# # change_lr1  adam pre crop 240-》224
# savepath = 'checkpoint/paper_test/AUCV1/mobilenetv2/augment/999/'

# # adam lr=1e-4, weight_decay=1e-650 pre aug3 epoches=500 milestones=[500, 1000, 1500] 1D5h
# savepath = 'checkpoint/paper_test/AUCV1/mobilenetv2/nopre/000/'
# savepath = 'checkpoint/paper_test/AUCV1/mobilenetv2_s3/nopre/000/'
# savepath = 'checkpoint/paper_test/AUCV1/mobilenetv2_s4/nopre/000/'
# savepath = 'checkpoint/paper_test/AUCV1/mobilenetv2_sgb/nopre/000/'
# savepath = 'checkpoint/paper_test/AUCV1/mobilenetv2_sgb_sa2/nopre/000/'

# # adam lr=1e-4, weight_decay=1e-650 pre aug3 epoches=50 milestones=[500, 1000, 1500] 3h
# savepath = 'checkpoint/paper_test/AUCV1/mobilenetv2/pre/000/'
# savepath = 'checkpoint/paper_test/AUCV1/mobilenetv2_s3/pre/000/'
# savepath = 'checkpoint/paper_test/AUCV1/mobilenetv2_s4/pre/000/'
# savepath = 'checkpoint/paper_test/AUCV1/mobilenetv2_sgb/pre/000/'
# savepath = 'checkpoint/paper_test/AUCV1/mobilenetv2_sgb_sa2/pre/000/'

# mobilenetv2_cbam sa 96$160
# # adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150]
# savepath = 'checkpoint/paper_test/AUCV1/mobilenetv2_cbam/000/'
# mobilenetv2_cbam sa 64$96$160
# # adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150]
# savepath = 'checkpoint/paper_test/AUCV1/mobilenetv2_cbam/111/'
# mobilenetv2_cbam sa 160
# # adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150]
# savepath = 'checkpoint/paper_test/AUCV1/mobilenetv2_cbam/222/'
# mobilenetv2_cbam cbam 160
# # adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150]
# savepath = 'checkpoint/paper_test/AUCV1/mobilenetv2_cbam/333/'
# mobilenetv2_cbam cbam 96$160
# # adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150]
# savepath = 'checkpoint/paper_test/AUCV1/mobilenetv2_cbam/444/'
# # adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150]
# savepath = 'checkpoint/paper_test/AUCV1/mobilenetv2_cbam/555/'
# # adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150]
# savepath = 'checkpoint/paper_test/AUCV1/mobilenetv2_cbam/666/'

# ghostnet
# # adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150]
# savepath = 'checkpoint/paper_test/AUCV1/ghostnet/000/'
# adam lr=1e-4, weight_decay=1e-650 pre aug3 epoches=50 milestones=[500, 1000, 1500]
# savepath = 'checkpoint/paper_test/AUCV1/ghostnet/pre/000/'
# adam lr=1e-4, weight_decay=1e-650 pre aug3 epoches=500 milestones=[500, 1000, 1500]
# savepath = 'checkpoint/paper_test/AUCV1/ghostnet/nopre/000/'
# # adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150]
# savepath = 'checkpoint/paper_test/AUCV2/ghostnet/000/'

# mnext
# # adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150]
# savepath = 'checkpoint/paper_test/AUCV1/mnext/000/'
# adam lr=1e-4, weight_decay=1e-650 pre aug3 epoches=50 milestones=[500, 1000, 1500]
# savepath = 'checkpoint/paper_test/AUCV1/mnext/pre/000/'
# adam lr=1e-4, weight_decay=1e-650 pre aug3 epoches=500 milestones=[500, 1000, 1500]
# savepath = 'checkpoint/paper_test/AUCV1/mnext/nopre/000/'
# # adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150]
# savepath = 'checkpoint/paper_test/AUCV2/mnext/000/'

# resnet18
# # adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150]
# savepath = 'checkpoint/paper_test/AUCV1/resnet18/000/'
# # adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150]
# savepath = 'checkpoint/paper_test/AUCV2/resnet18/000/'

# resnet50
# # adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150]
# savepath = 'checkpoint/paper_test/AUCV1/resnet50/000/'

# resnet50
# # adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150]
# savepath = 'checkpoint/paper_test/AUCV1/resnet50/000/'
# # adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150]
# savepath = 'checkpoint/paper_test/AUCV2/resnet50/000/'

# mobilenetv3_s
# # adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150]
# savepath = 'checkpoint/paper_test/AUCV1/mobilenetv3_s/000/'
# # adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150]
# savepath = 'checkpoint/paper_test/AUCV2/mobilenetv3_s/000/'
# adam lr=1e-4, weight_decay=1e-650 pre aug3 epoches=50 milestones=[500, 1000, 1500]
# savepath = 'checkpoint/paper_test/AUCV1/mobilenetv3_s/pre/000/'
# adam lr=1e-4, weight_decay=1e-650 pre aug3 epoches=500 milestones=[500, 1000, 1500]
# savepath = 'checkpoint/paper_test/AUCV1/mobilenetv3_s/nopre/000/'
# mobilenetv3_l
# # adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150]
# savepath = 'checkpoint/paper_test/AUCV1/mobilenetv3_l/000/'
# # adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150]
# savepath = 'checkpoint/paper_test/AUCV2/mobilenetv3_l/000/'
# adam lr=1e-4, weight_decay=1e-650 pre aug3 epoches=50 milestones=[500, 1000, 1500]
# savepath = 'checkpoint/paper_test/AUCV1/mobilenetv3_l/pre/000/'
# adam lr=1e-4, weight_decay=1e-650 pre aug3 epoches=500 milestones=[500, 1000, 1500]
# savepath = 'checkpoint/paper_test/AUCV1/mobilenetv3_l/nopre/000/'

# mobilenetv2
# # adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150]
# savepath = 'checkpoint/paper_test/AUCV2/mobilenetv2/000/'

# ours

# savepath = 'checkpoint/paper_test/ours/mobilenetv2/pre/crop/lr/111/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/pre/crop/lr/222/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/pre/crop/lr/333/'

# savepath = 'checkpoint/paper_test/ours/mobilenetv2_s4/pre/crop/lr/111/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2_s4/pre/crop/lr/222/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2_s4/pre/crop/lr/333/'

# savepath = 'checkpoint/paper_test/ours/mobilenetv2_sgb_sa2/pre/crop/lr/111/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2_sgb_sa2/pre/crop/lr/222/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2_sgb_sa2/pre/crop/lr/333/'

# savepath = 'checkpoint/paper_test/ours/mobilenetv2/pre/class6/crop/lr/111/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/pre/class6/crop/lr/222/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/pre/class6/crop/lr/333/'

# savepath = 'checkpoint/paper_test/ours/mobilenetv2_s4/pre/class6/crop/lr/111/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2_s4/pre/class6/crop/lr/222/'
savepath = 'checkpoint/paper_test/ours/mobilenetv2_s4/pre/class6/crop/lr/333/'

# savepath = 'checkpoint/paper_test/ours/mobilenetv2_sgb_sa2/pre/class6/crop/lr/111/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2_sgb_sa2/pre/class6/crop/lr/222/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2_sgb_sa2/pre/class6/crop/lr/333/'

## lr test
# # adam lr=1e-4, weight_decay=1e-650 pre aug3 epoches=500 milestones=[500, 1000, 1500]
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/nopre/000/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/nopre/crop/000/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2_sgb_sa2/nopre/000/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2_sgb_sa2/nopre/crop/000/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2_s4/nopre/000/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2_s4/nopre/crop/000/'

# # adam lr=1e-4, weight_decay=1e-650 pre aug3 epoches=50 milestones=[500, 1000, 1500]
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/pre/000/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/pre/crop/000/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/pre/class8/crop/000/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/pre/class6/crop/000/'

# savepath = 'checkpoint/paper_test/ours/mobilenetv2_sgb_sa2/pre/000/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2_sgb_sa2/pre/crop/000/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2_sgb_sa2/pre/class8/crop/000/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2_sgb_sa2/pre/class6/crop/000/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2_sgb_sa2/pre/class6/000/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2_sgb_sa2/pre/class6/111/'

# savepath = 'checkpoint/paper_test/ours/mobilenetv2_s4/pre/000/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2_s4/pre/crop/000/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2_s4/pre/class6/000/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2_s4/pre/class6/crop/000/'

# # adam lr=1e-4, weight_decay=1e-650 pre aug4 epoches=50 milestones=[500, 1000, 1500]
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/pre/crop/111/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/pre/crop/222/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2_sgb_sa2/pre/crop/111/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2_sgb_sa2/pre/crop/222/'

# savepath = 'checkpoint/paper_test/ours/mobilenetv2/pre/crop/lr/000/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/pre/crop/lr/111/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/pre/crop/lr/222/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/pre/crop/lr/333/'

# savepath = 'checkpoint/paper_test/ours/mobilenetv2/pre/lr/000/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/pre/lr/111/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/pre/lr/222/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/pre/lr/333/'

# savepath = 'checkpoint/paper_test/ours/mobilenetv2_sgb_sa2/pre/lr/000/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2_sgb_sa2/pre/lr/111/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2_sgb_sa2/pre/lr/222/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2_sgb_sa2/pre/lr/333/'

# # adam lr=1e-4, weight_decay=1e-650 pre aug3 epoches=50 milestones=[500, 1000, 1500]
# savepath = 'checkpoint/paper_test/ours/ghostnet/pre/000/'
# savepath = 'checkpoint/paper_test/ours/ghostnet/pre/111/'
# savepath = 'checkpoint/paper_test/ours/ghostnet/pre/crop/000/'
# savepath = 'checkpoint/paper_test/ours/mnext/pre/000/'
# savepath = 'checkpoint/paper_test/ours/mnext/pre/111/'
# savepath = 'checkpoint/paper_test/ours/mnext/pre/crop/000/'

# savepath = 'checkpoint/paper_test/ours/mobilenetv3_s/pre/000/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv3_s/nopre/000/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv3_s/pre/crop/000/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv3_l/pre/000/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv3_l/nopre/000/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv3_l/pre/crop/000/'

# adam lr=1e-4, weight_decay=1e-650 pre noaug milestones=[50, 100, 150]
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/augment/000/'
# adam lr=1e-4, weight_decay=1e-650 pre aug1 epoches=25 milestones=[20, 100, 150]
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/augment/111/'
# adam lr=1e-4, weight_decay=1e-650 pre aug2 epoches=25 milestones=[20, 100, 150]
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/augment/222/'
# adam lr=1e-4, weight_decay=1e-650 pre aug3 epoches=25 milestones=[20, 100, 150]
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/augment/333/'
# adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150]
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/augment/444/'

# ours class8
# adam lr=1e-4, weight_decay=1e-650 pre noaug milestones=[50, 100, 150]
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/class8/augment/000/'
# adam lr=1e-4, weight_decay=1e-650 pre aug1 epoches=25 milestones=[20, 100, 150]
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/class8/augment/111/'
# adam lr=1e-4, weight_decay=1e-650 pre aug2 epoches=25 milestones=[20, 100, 150]
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/class8/augment/222/'
# adam lr=1e-4, weight_decay=1e-650 pre aug3 epoches=25 milestones=[20, 100, 150]
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/class8/augment/333/'
# adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150]
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/class8/augment/444/'

# outs crop
# adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150]
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/crop/augment/444/'

# outs crop class8
# adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150]
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/crop/class8/augment/444/'
# adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=16 change_lr1
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/crop/class8/augment/555/'
# adam lr=1e-3, pre aug4 epoches=16 change_lr1
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/crop/class8/augment/666/'
# adam lr=1e-3, weight_decay=1e-650 pre aug4 epoches=16 change_lr1
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/crop/class8/augment/777/'
# adam lr=1e-3, weight_decay=1e-650 pre aug4 epoches=16 change_lr1
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/crop/class8/augment/888/'

# mobilenetv2_s
# # adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=16 change_lr1
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/crop/class8/mobilenetv2_s/000/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/crop/class8/mobilenetv2_s2/000/'
# adam lr=1e-3, weight_decay=1e-650 pre aug4 240>>224 epoches=16 change_lr1
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/crop/class8/mobilenetv2_s/111/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/crop/class8/mobilenetv2_s2/111/'
# # adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=16 change_lr1
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/crop/class8/mobilenetv2_s3/000/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/crop/class8/mobilenetv2_sgb/000/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/crop/class8/mobilenetv2_sgb_sa/000/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/crop/class8/mobilenetv2_sgb_sa2/000/'

# # adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150]
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/crop/class8/mobilenetv2_sgb_sa/111/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/crop/class8/mobilenetv2_sgb_sa2/111/'

# mix
# adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150]
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/mix/augment/444/'
# crop
# adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150]
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/mix/crop/augment/444/'

# mnext
# # adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150]
# savepath = 'checkpoint/paper_test/ours/mnext/000/'
# savepath = 'checkpoint/paper_test/ours/mnext/class8/000/'
# savepath = 'checkpoint/paper_test/ours/mnext/crop/000/'
# savepath = 'checkpoint/paper_test/ours/mnext/crop/class8/000/'
# # change_lr1  adam pre crop 240-》224
# savepath = 'checkpoint/paper_test/ours/mnext/crop/class8/111/'
# # change_lr1  adam pre crop 240-》224
# savepath = 'checkpoint/paper_test/ours/mnext/crop/class8/222/'
# # change_lr1  adam pre crop 160
# savepath = 'checkpoint/paper_test/ours/mnext/crop/class8/333/'

# outs class6
# # adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150]
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/class6/000/'
# savepath = 'checkpoint/paper_test/ours/mobilenetv2/class6/crop/000/'

# bus
# adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150] batch_size=16
# savepath = 'checkpoint/paper_test/mobilenetv2/pre/bus/000/'
# adam lr=1e-3, weight_decay=1e-650 mypre aug3 epoches=25 milestones=[10, 20, 150] batch_size=16
# savepath = 'checkpoint/paper_test/mobilenetv2/pre/bus/111/'
# adam lr=1e-3, weight_decay=1e-650 pre aug3 epoches=25 milestones=[10, 20, 150] batch_size=64
# savepath = 'checkpoint/paper_test/mobilenetv2/pre/bus/222/'
# adam lr=1e-3, weight_decay=1e-650 mypre aug3 epoches=25 milestones=[10, 20, 150] batch_size=64
# savepath = 'checkpoint/paper_test/mobilenetv2/pre/bus/333/'
# adam lr=1e-3, weight_decay=1e-650 pre aug4 epoches=25 milestones=[10, 20, 150] batch_size=64
# savepath = 'checkpoint/paper_test/mobilenetv2/pre/bus/444/'
# adam lr=1e-3, weight_decay=1e-650 pre aug4 epoches=25 milestones=[10, 20, 150] batch_size=64
# savepath = 'checkpoint/paper_test/mobilenetv2/pre/bus/555/'

def val(epoch):
    global best_val_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            average_loss = test_loss / (batch_idx + 1)
            test_acc = correct / total
            progress_bar(batch_idx, len(val_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (average_loss, 100. * test_acc, correct, total))

    acc = 100. * correct / total
    if acc >= best_val_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(savepath):
            # os.mkdir(savepath)
            os.makedirs(savepath)
        print("best_acc:{:.4f}".format(acc))

        # torch.save(state, savepath + 'mobilenetv2_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_s3_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_s4_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_s4_6_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_s4_crop_acc={:.4f}.pth'.format(acc))
        torch.save(state, savepath + 'mobilenetv2_s4_6_crop_acc={:.4f}.pth'.format(acc))

        # torch.save(state, savepath + 'mobilenetv2_sgb_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_sgb_sa2_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_crop_sgb_sa2_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_8_crop_sgb_sa2_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_6_crop_sgb_sa2_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_6_sgb_sa2_acc={:.4f}.pth'.format(acc))

        # torch.save(state, savepath + 'mobilenetv2_6_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_6_crop_acc={:.4f}.pth'.format(acc))

        # torch.save(state, savepath + 'mobilenetv2_8_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_6_crop_acc={:.4f}.pth'.format(acc))

        # torch.save(state, savepath + 'mobilenetv3_s_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv3_s_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv3_l_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv3_l_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'ghostnet_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'ghostnet_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mnext_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mnext_crop_acc={:.4f}.pth'.format(acc))

        # torch.save(state, savepath + 'mobilenetv2_s_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_s2_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_s3_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_sgb_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_sgb_sa_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_sgb_sa2_8_crop_acc={:.4f}.pth'.format(acc))

        # torch.save(state, savepath + 'mobilenetv2_cbam_l16_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_cbam_l24_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_cbam_l32_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_cbam_l64_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_cbam_l96_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_cbam_l160_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_cbam_l96&160_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_cbam_l64&96&160_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_cbam_l96&160_8_crop_n_acc={:.4f}.pth'.format(acc))

        # torch.save(state, savepath + 'mobilenetv2_sa_l16_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_sa_l24_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_sa_l32_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_sa_l64_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_sa_l96_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_sa_l160_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_sa_l96&160_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_sa_l64&96&160_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_sa_l96&160_8_crop_n_acc={:.4f}.pth'.format(acc))

        # torch.save(state, savepath + 'mobilenetv2_ca_l160_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_ca_l96&160_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_sa_l160_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_sa_l96&160_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_sa_l64$96&160_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_cbam_l160_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_cbam_l96$160_acc={:.4f}.pth'.format(acc))

        # torch.save(state, savepath + 'mobilenetv2_ca_l16_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_ca_l24_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_ca_l64_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_ca_l96_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_ca_l160_8_crop_acc={:.4f}.pth'.format(acc))

        # torch.save(state, savepath + 'resnet18_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'resnet18_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'resnet50_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'resnext50_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'resnext101_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv3_s_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv3_s_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv3_s_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv3_s_8_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv3_l_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv3_l_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv3_l_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv3_l_8_acc={:.4f}.pth'.format(acc))

        # torch.save(state, savepath + 'resnet18_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'resnet50_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mnext_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mnext_8_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mnext_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mnext_8_crop_acc={:.4f}.pth'.format(acc))
        # # torch.save(state, savepath + 'ghostnet_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'ghostnet_8_acc={:.4f}.pth'.format(acc))

        # torch.save(state, savepath + 'resnext50_crop_acc={:.4f}.pth'.format(acc))
        best_val_acc = acc
    return average_loss, test_acc


def main(epoches=epoches):
    x = []
    lrs = []
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    start_time = time.time()
    for epoch in range(start_epoch, start_epoch + epoches):
        train_l, train_a, lr = train(epoch)
        test_l, test_a = val(epoch)
        x.append(epoch)
        lrs.append(lr)
        train_loss.append(train_l)
        test_loss.append(test_l)
        train_acc.append(train_a)
        test_acc.append(test_a)
        print("epoch={}/{},lr={},train_loss={:.3f},test_loss={:.3f},train_acc={:.3f},test_acc={:.3f}"
              .format(epoch + 1, epoches, lr, train_l, test_l, train_a, test_a))

        # # # earlystop
        # if lr < 1e-4-1e-5:
        #     break
        # if lr < 1e-6 - 1e-7:
        #     break
        print("total train time ={}".format(format_time(time.time() - start_time)))
    # 保持训练数据
    dict = {}
    dict["lr"] = lrs
    dict["train_loss"] = train_loss
    dict["test_loss"] = test_loss
    dict["train_acc"] = train_acc
    dict["test_acc"] = test_acc
    test_path = os.path.dirname(savepath)
    with open(os.path.join(test_path, "train.json"), "w", encoding='utf-8') as f:
        json.dump(dict, f)
    # 保存excel
    df = pd.DataFrame(dict, index=x)
    df.to_excel(os.path.join(test_path, 'train.xlsx'))

    # 展示训练数据
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
    # 保存图片
    plt.savefig(savepath + 'learing.jpg')
    plt.show()


def net_test():
    num_classes = 6
    # num_classes = 7
    # num_classes = 8
    # num_classes = 9
    # net = models.mobilenet_v2(pretrained=False, num_classes=num_classes, width_mult=1.0)
    net = MobileNetV2_sgb_sa2(num_classes=num_classes)
    # net = MobileNetV2_cbam(num_classes=num_classes, width_mult=1.0, add_location=(96, 160), ca=False, sa=True)
    # model_path = r"checkpoint/paper_test/mobilenetv2/pre/bus/555/mobilenetv2_8_acc=85.2596.pth"
    # model_path = r"checkpoint/paper_test/ours/mobilenetv2_sgb_sa2/pre/crop/000/mobilenetv2_crop_sgb_sa2_acc=88.1364.pth"
    model_path = r"checkpoint/paper_test/ours/mobilenetv2_sgb_sa2/pre/class6/crop/000/mobilenetv2_6_crop_sgb_sa2_acc=91.8470.pth"
    # 加载模型权重，忽略不同
    model_dict = net.state_dict()
    checkpoint = torch.load(model_path, map_location=device)
    pretrained_dict = checkpoint["net"]
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    print("loaded model with acc:{}".format(checkpoint["acc"]))
    net.to(device)
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # test_dataset = MyDataset("data/txt_raw_crop/total_test_crop_8.txt", test_transform)
    # test_dataset = MyDataset("data/txt_raw_crop/total_test_crop.txt", test_transform)
    # test_dataset = MyDataset("data/bus/test224.txt", test_transform)
    # test_dataset = MyDataset("data/ours/224/test224.txt", test_transform)
    # test_dataset = MyDataset("data/ours/224/test_crop224.txt", test_transform)
    test_dataset = MyDataset("data/ours/224/test_crop224_6.txt", test_transform)
    # test_dataset = MyDataset("data/bus/test224_8.txt", test_transform)
    # test_dataset = MyDataset("data/ours/224/test224_8.txt", test_transform)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=64,
                                 shuffle=True,
                                 num_workers=0)

    # Confusion_matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            # print(targets, predicted)
            for i in range(targets.shape[0]):
                cm[targets[i]][predicted[i]] += 1
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            average_loss = test_loss / (batch_idx + 1)
            test_acc = correct / total
            progress_bar(batch_idx, len(test_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (average_loss, 100. * test_acc, correct, total))
    # print(average_loss, test_acc)
    print("test_acc: ", test_acc)
    if num_classes == 9:
        labels = ["正常", "侧视", "喝水", "吸烟", "操作中控", "玩手机", "侧身拿东西", "整理仪容", "接电话"]
    elif num_classes == 6:
        labels = ["正常", "喝水", "吸烟", "操作中控", "玩手机", "接电话"]
    elif num_classes == 7:
        labels = ["正常", "喝水", "吸烟", "操作中控", "玩手机", "接电话", "其他"]
    else:
        labels = ["正常", "侧视", "喝水", "吸烟", "操作中控", "玩手机", "侧身拿东西", "接电话"]
    print(labels)
    print("row:target   col:predict")
    print(cm)
    true_label = np.zeros((num_classes,), dtype=np.int)
    predicted_label = np.zeros((num_classes,), dtype=np.int)
    total = 0
    for i in range(num_classes):
        for j in range(num_classes):
            true_label[i] += cm[i][j]
            predicted_label[i] += cm[j][i]
            total += cm[i][j]
    print("true label:", true_label)
    print("predicted label:", predicted_label)
    TP = np.zeros((num_classes,), dtype=np.int)
    FP = np.zeros((num_classes,), dtype=np.int)
    FN = np.zeros((num_classes,), dtype=np.int)
    TN = np.zeros((num_classes,), dtype=np.int)
    Accuracy = np.zeros((num_classes,), dtype=np.float)
    Precision = np.zeros((num_classes,), dtype=np.float)
    Recall = np.zeros((num_classes,), dtype=np.float)
    F1 = np.zeros((num_classes,), dtype=np.float)
    for i in range(num_classes):
        TP[i] = cm[i][i]
        FP[i] = true_label[i] - TP[i]
        FN[i] = predicted_label[i] - TP[i]
        TN[i] = total - true_label[i] - FN[i]
        Accuracy[i] = (TP[i] + TN[i]) / total
        Precision[i] = TP[i] / predicted_label[i]
        Recall[i] = TP[i] / true_label[i]
        F1[i] = Precision[i] * Recall[i] / (Precision[i] + Recall[i]) * 2
    print("TP:", TP)
    print("TN:", TN)
    print("FP:", FP)
    print("FN:", FN)
    # 准确率，对于给定的测试数据集，分类器正确分类的样本数与总样本数之比。
    print("Accuracy:", Accuracy)
    # 精准率是检索出相关文档数与检索出的文档总数的比率（正确分类的正例个数占分类为正例的实例个数的比例），衡量的是检索系统的查准率。
    print("Precision:", Precision)
    # 召回率是指检索出的相关文档数和文档库中所有的相关文档数的比率（正确分类的正例个数占实际正例个数的比例），衡量的是检索系统的查全率。
    print("Recall:", Recall)
    # F1综合了P和R的结果，当F1较高时则能说明试验方法比较有效
    print("F1:", F1)
    test_path = os.path.dirname(model_path)
    dict = {}
    dict["Accuracy"] = Accuracy.tolist()  # 准确率 样本被分类正确的概率, 包括TP和TF
    dict["Precision"] = Precision.tolist()  # 精确率 样本识别正确的概率，
    dict["Recall"] = Recall.tolist()  # 召回率 样本被正确识别出的概率，检出率
    dict["F1-score"] = F1.tolist()
    df = pd.DataFrame(dict, index=labels)
    df.to_excel(os.path.join(test_path, 'test.xlsx'))
    dict["cm"] = cm.tolist()
    with open(os.path.join(test_path, "test.json"), "w", encoding='utf-8') as f:
        json.dump(dict, f)
    # 保存excel
    df2 = pd.DataFrame(cm, index=labels)
    df2.to_excel(os.path.join(test_path, 'cm.xlsx'))

# 配置环境变量
# cd D:\Program Files (x86)\Intel\openvino_2020.3.341\bin
# setupvars.bat
# cd D:\code\EfficientNet-PyTorch-master
def net_test_onnx():
    # Load dataset
    dataset_path = r"data/txt/12_23_12_addpre_test224.txt"
    with open(dataset_path) as f:
        datasets = [c.strip() for c in f.readlines()]
    path = r"checkpoint/data_12_23/mobilenetv2/888/mobilenetv2_1_12_23_acc=91.6275.onnx"
    # num_classes = 9
    num_classes = 6
    onnx_session = onnxruntime.InferenceSession(path, None)
    dnn_net = cv2.dnn.readNetFromONNX(path)
    # xml_path = r"checkpoint/data_12_23/mobilenetv2/888/mobilenetv2_1_12_23_acc=91.6275.xml"
    # bin_path = r"checkpoint/data_12_23/mobilenetv2/888/mobilenetv2_1_12_23_acc=91.6275.bin"
    # # # FP16
    # # xml_path = r"checkpoint/data_12_23/mobilenetv2/8888/mobilenetv2_1_12_23_acc=91.6275.xml"
    # # bin_path = r"checkpoint/data_12_23/mobilenetv2/8888/mobilenetv2_1_12_23_acc=91.6275.bin"
    # dnn_net = cv2.dnn.readNet(xml_path, bin_path)
    # dnn_net = cv2.dnn.readNetFromModelOptimizer(xml_path, bin_path)
    # dnn_net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
    # dnn_net.setPreferableTarget(cv2.dnn.DNN_BACKEND_HALIDE)
    # dnn_net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
    model = models.mobilenet_v2(pretrained=False, num_classes=num_classes, width_mult=1.0)
    # 加载模型参数
    path = r"checkpoint/data_12_23/mobilenetv2/888/mobilenetv2_1_12_23_acc=91.6275.pth"
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["net"])
    model.cpu()
    model.eval()

    total = 0
    right = 0
    dnn_right = 0
    model_right = 0
    for data in datasets:

        img_path = data.split(" ")[0]
        label = int(data.split(" ")[1])
        src = cv2.imread(img_path)
        # print(src.shape) # height,weight,channel
        src2 = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        image = cv2.resize(src2, (224, 224))
        # print(image.shape)
        image = np.float32(image) / 255.0
        image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
        image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
        blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224), (0, 0, 0), False)
        dnn_net.setInput(blob)
        start = time.time()
        dnn_probs = dnn_net.forward()
        print("dnn inference time:", time.time() - start)
        dnn_index = np.argmax(
            dnn_probs)  # By default, the index is into the flattened array, otherwise along the specified axis.
        # dnn_softmax = softmax_np(dnn_probs)
        # print(dnn_index, dnn_softmax.max())
        # print(image.shape)
        image = image.transpose(2, 0, 1)  # 转换轴，pytorch为channel first
        image = image.reshape(1, 3, 224, 224)  # barch,channel,height,weight

        # image = []
        # image.append(image)
        # image = np.asarray(image)

        inputs = {onnx_session.get_inputs()[0].name: image}
        probs = onnx_session.run(None, inputs)
        probs = np.array(probs)
        # print(probs.shape)
        index = np.argmax(probs)
        print(index)

        # softmax = softmax_np(probs)
        softmax = softmax_flatten(probs)
        print(index, softmax.max())
        model_image = torch.from_numpy(image)
        output = model(model_image)
        model_index = np.argmax(output.detach().numpy())
        # print("dnn_probs:{},probs:{},output:{}".format(dnn_probs, probs, output))
        # print("dnn_index:{},index:{},model_index:{},label:{}".format(dnn_index, index, model_index, label))
        total += 1
        if index == label:
            right += 1

        if dnn_index == label:
            dnn_right += 1

        if model_index == label:
            model_right += 1
    print("acc:{},dnn_acc:{},model_acc :{}".format(right / total, dnn_right / total, model_right / total))


if __name__ == '__main__':
    main()
    # net_test()
    # net_test_onnx()
