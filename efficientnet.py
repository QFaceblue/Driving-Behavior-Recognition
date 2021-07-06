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
from ghost_net import ghost_net, ghost_net_Cifar
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
from mobilenetv2_cbam import MobileNetV2_cbam
def softmax_np(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
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

def efficientnet_test():
    model = EfficientNet.from_pretrained('efficientnet-b0')

    # Preprocess image
    tfms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
    img = tfms(Image.open('./data/imgs/elephant.jpg')).unsqueeze(0)
    print(img.shape)  # torch.Size([1, 3, 224, 224])

    # Load ImageNet class names
    labels_map = json.load(open('.\data\labels_map.txt'))
    labels_map = [labels_map[str(i)] for i in range(1000)]

    # Classify
    model.eval()
    with torch.no_grad():
        outputs = model(img)

    # Print predictions
    print('-----')
    for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
        prob = torch.softmax(outputs, dim=1)[0, idx].item()
        print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob * 100))
        # print('{label:<75} ({p:.2f}%)'.format(label=idx, p=prob * 100))


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


# #一个通用的展示少量预测图片的函数
# def visualize_model(model, dataloader,device):
#     model.eval()
#     model.to(device)
#     with torch.no_grad():
#         inputs, labels = next(iter(dataloader))
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#         outputs = model(inputs)
#         _, preds = torch.max(outputs, 1)
#         out = torchvision.utils.make_grid(inputs).cpu()
#         # title = "predect/label"
#         title = ""
#         for i,label in enumerate(labels):
#             # title+="  {}/{}  ".format(preds[i],label)
#             title += "    {}    ".format(label_name[preds[i]])
#         imshow(out, title=title)

# def visualize_pred():
#
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model = EfficientNet.from_name('efficientnet-b0',num_classes=10)
#     # 加载模型参数
#     # path = r"checkpoint/B0/000/B0_acc=99.8528.pth"
#     path = r"checkpoint\B0\111\B0_acc=99.5540.pth"
#     checkpoint = torch.load(path)
#     model.load_state_dict(checkpoint["net"])
#     print("loaded model with acc:{}".format(checkpoint["acc"]))
#     # data_path = r"E:\Datasets\state-farm-distracted-driver-detection\imgs\train"
#     # # data_path = r".\data\hymenoptera_data\train"
#     # train_dataset = datasets.ImageFolder(root=data_path, transform=data_transform)
#     # train_dataloader = DataLoader(dataset=train_dataset,
#     #                               batch_size=4,
#     #                               shuffle=True,
#     #                               num_workers=0)
#     val_dataset = MyDataset("./data/dval.txt", data_transform)
#     val_dataloader = DataLoader(dataset=val_dataset,
#                                 batch_size=4,
#                                 shuffle=True,
#                                 num_workers=0)
#     # visualize_model(model,train_dataloader,device)
#     visualize_model(model, val_dataloader, device)

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

# #
# train_transform = transforms.Compose([
#
#     transforms.RandomResizedCrop((224, 224),  scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.), ),
#     # transforms.RandomResizedCrop((320, 320),  scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.), ),
#     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
#     # transforms.RandomRotation(10, resample=False, expand=False, center=None),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])
#
# val_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     # transforms.Resize((320, 320)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# train_transform = transforms.Compose([
#
#     transforms.Resize((224, 224)),
#     # transforms.ColorJitter(brightness=0.8, contrast=0.5, saturation=0.5, hue=0.1),
#     # transforms.RandomRotation(10, resample=False, expand=False, center=None),
#     # transforms.RandomCrop(224, padding=16),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])
# train_transform = transforms.Compose([
#
#     transforms.Resize((224, 224)),
#     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
#     # transforms.ColorJitter(brightness=0.8, contrast=0.5, saturation=0.5, hue=0.1),
#     # transforms.RandomRotation(20, resample=False, expand=False, center=None),
#     # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     transforms.RandomRotation(10, resample=False, expand=False, center=None),
#     # transforms.RandomHorizontalFlip(p=0.5),
#     # # transforms.RandomVerticalFlip(p=0.5),
#     # # ToTensor()能够把灰度范围从0-255变换到0-1之间，
#     # # transform.Normalize()则把0-1变换到(-1,1).具体地说，对每个通道而言，Normalize执行以下操作：
#     # # image=(image-mean)/std
#     # transforms.RandomResizedCrop((224,224)),
#     # transforms.Resize((224, 224)),
#     transforms.RandomCrop(224, padding=16),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])
#

train_transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.RandomCrop(224),
    # transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    # transforms.RandomRotation(20, resample=False, expand=False, center=None),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(10, resample=False, expand=False, center=None),
    # transforms.RandomHorizontalFlip(p=0.5),
    # # transforms.RandomVerticalFlip(p=0.5),
    # # ToTensor()能够把灰度范围从0-255变换到0-1之间，
    # # transform.Normalize()则把0-1变换到(-1,1).具体地说，对每个通道而言，Normalize执行以下操作：
    # # image=(image-mean)/std
    # transforms.RandomResizedCrop((224,224)),
    # transforms.Resize((224, 224)),
    # transforms.RandomCrop(224, padding=16),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# train_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     # transforms.RandomCrop(320),
#     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
#     transforms.RandomRotation(10, resample=False, expand=False, center=None),
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


# train_transform = transforms.Compose([
#     transforms.Resize((320, 320)),
#     # transforms.RandomCrop(320),
#     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
#     transforms.RandomRotation(10, resample=False, expand=False, center=None),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])
#
# val_transform = transforms.Compose([
#     transforms.Resize((320, 320)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# train_transform = transforms.Compose([
#     transforms.Resize((340, 340)),
#     transforms.RandomCrop(320),
#     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
#     transforms.RandomRotation(10, resample=False, expand=False, center=None),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])
#
# val_transform = transforms.Compose([
#     transforms.Resize((320, 320)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# train_transform = transforms.Compose([
#     transforms.Resize((160, 160)),
#     # transforms.RandomCrop(320),
#     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
#     transforms.RandomRotation(10, resample=False, expand=False, center=None),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])
#
# val_transform = transforms.Compose([
#     transforms.Resize((160, 160)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# train_transform = transforms.Compose([
#
#     transforms.Resize((160, 160)),
#     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
#     # transforms.RandomRotation(20, resample=False, expand=False, center=None),
#     # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     transforms.RandomRotation(10, resample=False, expand=False, center=None),
#     # transforms.RandomHorizontalFlip(p=0.5),
#     # # transforms.RandomVerticalFlip(p=0.5),
#     # # ToTensor()能够把灰度范围从0-255变换到0-1之间，
#     # # transform.Normalize()则把0-1变换到(-1,1).具体地说，对每个通道而言，Normalize执行以下操作：
#     # # image=(image-mean)/std
#     # # transforms.RandomResizedCrop((224,224)),
#     # # transforms.Resize((224, 224)),
#     transforms.RandomCrop(160, padding=16),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
best_acc = 0  # best test accuracy
best_val_acc = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
# kaggle dataset
# num_classes = 10
# label_name = ["正常","右持手机","右接电话","左持手机","左接电话","操作仪器","喝水","向后侧身","整理仪容","侧视"]
# label_name = ["normal", "texting-R", "answering_R", "texting-L", "answering_L", "operating", "drinking", "leaning_back", "makeup", "side_view"]
# # 新的类别
# # # label_name = ["正常","右接电话",左接电话","低头","操作仪器","喝水","吸烟","向后侧身","整理仪容","侧视"]
# # label_name =["normal", "right to answer the phone", left to answer the phone "," head down "," operating instruments "," drinking water "," smoking "," leaning back ","makeup "," side view "]
# # mydataset
# classes_path = r"data/drive_classes.txt"
# with open(classes_path) as f:
#     label_name = [c.strip() for c in f.readlines()]
# num_classes = len(label_name)

# num_classes = 100
# num_classes = 10
num_classes = 9
# num_classes = 8
# num_classes = 7
# num_classes = 6
# net = EfficientNet.from_pretrained('efficientnet-b0',num_classes=num_classes)

# net = models.resnet18(pretrained=True)
# net = models.resnext50_32x4d(pretrained=True)
# net = models.resnext50_32x4d(pretrained=False,num_classes=num_classes)
# net = models.resnet50(pretrained=False,num_classes=num_classes)
# net = models.mobilenet_v2(pretrained=True)
# net = models.mobilenet_v2(pretrained=False, num_classes=num_classes, width_mult=1.0)
# net = models.mobilenet_v2(pretrained=False, num_classes=num_classes, width_mult=0.5)
# net = models.mobilenet_v2(pretrained=False, num_classes=num_classes, width_mult=0.3)
# net = models.mobilenet_v2(pretrained=False, num_classes=num_classes, width_mult=0.1)

# net = models.mobilenet_v2(pretrained=False, width_mult=1.0)

net = models.mobilenet_v2(pretrained=True, width_mult=1.0)
num_in = net.classifier[1].in_features
net.classifier[1] = nn.Linear(num_in, num_classes)

# net = mobilenet_v3_small(pretrained=True)
# # net = mobilenet_v3_large(pretrained=True)
# num_in = net.classifier[3].in_features
# net.classifier[3] = nn.Linear(num_in, num_classes)

#
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

# net = my_mobilenext_2(num_classes=num_classes, width_mult=1.)
# num_in = net.fc.in_features
# # 创建的层默认参数需要训练
# net.fc = nn.Linear(num_in, num_classes)

# net = MobileNetV3(n_class=num_classes, mode="small", dropout=0.2, width_mult=1.0)
# net = MobileNetV3(n_class=num_classes, mode="large", dropout=0.2, width_mult=1.0)

# net = MobileNetV3_Small(num_classes=num_classes)
# net = MobileNetV3(n_class=num_classes, mode="large", dropout=0.2, width_mult=1.0)

# # 加载模型权重，忽略不同
# # model_path = r"checkpoint/imagenet/imagenet100/mobilenetv2/111/mobilenetv2_1_imagenet_acc=68.9234.pth"
# # model_path = r"checkpoint/kaggle/v1/mobilenetv2/pre/000/mobilenetv2_1_kg1_acc=85.2244.pth"
# # model_path = r"checkpoint/imagenet/imagenet100/ghostnet/000/ghostnet_1_imagenet_acc=63.0497.pth"
# # model_path = r"checkpoint/imagenet/imagenet100/mnext/000/mnext_1_imagenet_acc=65.5769.pth"
# model_path = r"weights/mnext.pth.tar"
# model_dict =net.state_dict()
# checkpoint = torch.load(model_path, map_location=device)
# # pretrained_dict = checkpoint["net"]
# pretrained_dict = checkpoint
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
# model_dict.update(pretrained_dict)
# net.load_state_dict(model_dict)
# # print("loaded model with acc:{}".format(checkpoint["acc"]))

# # # # # 预加载
# # path = r"checkpoint\resnet18\000\B0_acc=83.4532.pth"
# # path = r"checkpoint/mobilenetv2/000/mv2_acc=82.7338.pth"
# path = r"checkpoint/resnext50/333/resnext50_my_acc=72.6619.pth"
# checkpoint = torch.load(path)
# net.load_state_dict(checkpoint["net"],strict=False) # 模型参数大小不一样仍然报错！可能因为其通过参数名确定是否加载，但参数名相同默认参数大小一样，而这里刚好不一样故报错
# print("loaded model with acc:{}".format(checkpoint["acc"]))


# num_in = net.fc.in_features
# # 创建的层默认参数需要训练
# net.fc = nn.Linear(num_in, num_classes)

# 按照dim，将index指定位置的值取出
# gather() For a 3-D tensor the output is specified by:
# out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
# out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
# out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
# 按照dim，将值放入index指定的位置
# scatter_() For a 3-D tensor, self is updated as:
# self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
# self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
# self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs, dim=1)
    batch_size, class_num = outputs.shape
    onehot_targets = torch.zeros(batch_size, class_num).to(targets.device).scatter_(1, targets.view(batch_size, 1), 1)
    return -(log_softmax_outputs * onehot_targets).sum(dim=1).mean()


def CrossEntropy_KD(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs, dim=1)
    softmax_targets = F.softmax(targets, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()


def change_lr2(epoch, T=20, factor=0.3, min=1e-4):
    mul = 1.
    if epoch < T:
        mul = mul
    elif epoch < T * 2:
        mul = mul * factor
    elif epoch < T * 3:
        mul = mul * factor * factor
    elif epoch < T * 4:
        mul = mul * factor * factor * factor
    elif epoch < T * 5:
        mul = mul * factor * factor * factor * factor
    else:
        return min
    # print(max((1 + math.cos(math.pi * (epoch % T) / T)) * mul/2, min))
    return max((1 + math.cos(math.pi * (epoch % T) / T)) * mul / 2, min)


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


def change_lr4(epoch, T=10, factor=0.3, min=1e-4):
    mul = 1.
    if epoch < T * 3:
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
    return max((1 + math.cos(math.pi * epoch / T)) * mul / 2, min)


def change_lr5(epoch, T=10, factor=0.3, min=1e-3):
    mul = 1.
    if epoch < T * 3:
        mul = mul
    elif epoch < T * 5:
        mul = mul * factor
    elif epoch < T * 7:
        mul = mul * factor * factor
    else:
        return min
    # print(max((1 + math.cos(math.pi * epoch/ T)) * mul/2, min))
    return max((1 + math.cos(math.pi * epoch / T)) * mul / 2, min)


def change_lr6(epoch, T=6, factor=0.3, min=1e-3):
    mul = 1.
    if epoch < T * 3:
        mul = mul
    elif epoch < T * 5:
        mul = mul * factor
    elif epoch < T * 7:
        mul = mul * factor * factor
    else:
        return min
    # print(max((1 + math.cos(math.pi * epoch/ T)) * mul/2, min))
    return max((1 + math.cos(math.pi * epoch / T)) * mul / 2, min)


def change_lr7(epoch, T=8, factor=0.3, min=1e-3):
    mul = 1.
    if epoch < T * 3:
        mul = mul
    elif epoch < T * 5:
        mul = mul * factor
    else:
        return min
    return max((1 + math.cos(math.pi * epoch / T)) * mul / 2, min)


# 注意 new_lr = lr * mul
def change_lr8(epoch, T=6, factor=0.3, min=1e-2):
    mul = 1.
    if epoch < T * 3:
        mul = mul
    elif epoch < T * 5:
        mul = mul * factor
    elif epoch < T * 7:
        mul = mul * factor * factor
    else:
        return min
    # print(max((1 + math.cos(math.pi * epoch/ T)) * mul/2, min))
    return max((1 + math.cos(math.pi * epoch / T)) * mul / 2, min)

def change_lr9(epoch, T=6, factor=0.3, min=1e-3):
    mul = 1.
    if epoch < T * 3:
        mul = mul
    elif epoch < T * 7:
        mul = mul * factor
    else:
        return min
    return max((1 + math.cos(math.pi * epoch / T)) * mul / 2, min)

def change_lr10(epoch, T=5, factor=0.3, min=1e-2):
    mul = 1.
    if epoch < T * 3:
        mul = mul
    elif epoch < T * 5:
        mul = mul * factor
    else:
        return min
    # print(max((1 + math.cos(math.pi * epoch/ T)) * mul/2, min))
    return max((1 + math.cos(math.pi * epoch / T)) * mul / 2, min)

def change_lr11(epoch, T=8, min=1e-3):
    mul = 1.
    if epoch < T:
        mul = mul
    else:
        return min
    # print(max((1 + math.cos(math.pi * epoch/ T)) * mul/2, min))
    return max((1 + math.cos(math.pi * epoch / T)) * mul / 2, min)

def change_lr12(epoch, T=6, factor=0.3, min=1e-3):
    mul = 1.
    if epoch < T:
        mul = mul
    elif epoch < T * 3:
        mul = mul * factor
    else:
        return min
    return max((1 + math.cos(math.pi * epoch / T)) * mul / 2, min)

criterion = nn.CrossEntropyLoss()
criterion = CrossEntropy
# epoches = 48
# epoches = 30
# epoches = 30
# epoches = 16
# epoches = 30
# optimizer = optim.Adam(net.parameters(), lr=5e-3)
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_lr9)
# optimizer = optim.Adam(net.parameters(), lr=1e-2)
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_lr9)
# optimizer = optim.Adam(net.parameters(), lr=1e-3)
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_lr9)
# optimizer = optim.Adam(net.parameters(), lr=1e-3)
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_lr10)
# optimizer = optim.SGD(net.parameters(), lr=1e-3)
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_lr9)
# optimizer = optim.Adam(net.parameters(), lr=1e-2)
# optimizer = optim.Adam(net.parameters(), lr=1e-3)
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_lr6)
# optimizer = optim.SGD(net.parameters(), lr=1e-1,
#                       momentum=0.9, weight_decay=5e-4)
# optimizer = optim.SGD(net.parameters(), lr=1e-2,
#                       momentum=0.9, weight_decay=5e-4)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 8], gamma=0.1)
#
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3,
#                                                  verbose=True, threshold=1e-4, threshold_mode='rel',
#                                                  cooldown=0, min_lr=1e-7, eps=1e-8)

# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_lr4)

# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_lr7)
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_lr8)

# epoches = 48
# optimizer = optim.Adam(net.parameters(), lr=1e-3)
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_lr9)
# epoches = 30
# optimizer = optim.Adam(net.parameters(), lr=1e-3)
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_lr10)
epoches = 18
optimizer = optim.Adam(net.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_lr12)
# epoches = 16
# optimizer = optim.Adam(net.parameters(), lr=1e-3)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6, 12], gamma=0.1)
# epoches = 8
# optimizer = optim.Adam(net.parameters(), lr=1e-3)
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_lr11)

net.to(device)

# data_path = r"E:\Datasets\state-farm-distracted-driver-detection\imgs\train"
# data_path = r".\data\hymenoptera_data\train"

# datasets.ImageFolder读取图片文件夹，要求图片按照类别分文件夹存放
# root：在root指定的路径下寻找图片
# transform：对PIL Image进行的转换操作，transform的输入是使用loader读取图片的返回对象
# target_transform：对label的转换
# loader：给定路径后如何读取图片，默认读取为RGB格式的PIL Image对象


# # kaggle dataset 100
# train_dataset = MyDataset("data/imagenet/imagenet2012_100_train.txt", train_transform)
# val_dataset = MyDataset("data/imagenet/imagenet2012_100_val.txt", val_transform)
# # train_dataloader = DataLoader(dataset=train_dataset,
# #                               batch_size=128,
# #                               shuffle=True,
# #                               num_workers=0)
# #
# #
# # val_dataloader = DataLoader(dataset=val_dataset,
# #                             batch_size=128,
# #                             shuffle=True,
# #                             num_workers=0)
#
# train_dataloader = DataLoader(dataset=train_dataset,
#                               batch_size=96,
#                               shuffle=True,
#                               num_workers=0)
#
#
# val_dataloader = DataLoader(dataset=val_dataset,
#                             batch_size=96,
#                             shuffle=True,
#                             num_workers=0)


# # # kaggle dataset
# # train_dataset = datasets.ImageFolder(root=data_path, transform=data_transform)
# train_dataset = MyDataset("data/txt/kg_train224.txt", train_transform)
# val_dataset = MyDataset("data/txt/kg_val224.txt", val_transform)
# train_dataloader = DataLoader(dataset=train_dataset,
#                               batch_size=64,
#                               shuffle=True,
#                               num_workers=0)
#
# val_dataloader = DataLoader(dataset=val_dataset,
#                             batch_size=64,
#                             shuffle=True,
#                             num_workers=0)
#
# kaggle dataset 2
# train_dataset = MyDataset("data/txt/kg_train2.txt", train_transform)
# val_dataset = MyDataset("data/txt/kg_val2.txt", val_transform)
# train_dataloader = DataLoader(dataset=train_dataset,
#                               batch_size=64,
#                               shuffle=True,
#                               num_workers=0)
#
#
# val_dataloader = DataLoader(dataset=val_dataset,
#                             batch_size=64,
#                             shuffle=True,
#                             num_workers=0)

# train_dataset = MyDataset("data/txt/kg_train2_224.txt", train_transform)
# val_dataset = MyDataset("data/txt/kg_val2_224.txt", val_transform)
# train_dataloader = DataLoader(dataset=train_dataset,
#                               batch_size=64,
#                               shuffle=True,
#                               num_workers=0)
#
#
# val_dataloader = DataLoader(dataset=val_dataset,
#                             batch_size=64,
#                             shuffle=True,
#                             num_workers=0)
# # AUC v1 dataset
# train_dataset = MyDataset("data/txt/aucv1_trainVal224.txt", train_transform)
# val_dataset = MyDataset("data/txt/aucv1_test224.txt", val_transform)
# train_dataloader = DataLoader(dataset=train_dataset,
#                               batch_size=64,
#                               shuffle=True,
#                               num_workers=0)
# val_dataloader = DataLoader(dataset=val_dataset,
#                             batch_size=64,
#                             shuffle=True,
#                             num_workers=0)

# AUC v2 dataset
# train_dataset = MyDataset("data/txt/auc_trainVal224.txt", train_transform)
# val_dataset = MyDataset("data/txt/auc_test224.txt", val_transform)
# train_dataloader = DataLoader(dataset=train_dataset,
#                               batch_size=64,
#                               shuffle=True,
#                               num_workers=0)
# val_dataloader = DataLoader(dataset=val_dataset,
#                             batch_size=64,
#                             shuffle=True,
#                             num_workers=0)
# # AUC v2 dataset
# train_dataset = MyDataset("data/txt/auc_trainVal224.txt", train_transform)
# val_dataset = MyDataset("data/txt/auc_test224.txt", val_transform)
# train_dataloader = DataLoader(dataset=train_dataset,
#                               batch_size=64,
#                               shuffle=True,
#                               num_workers=0)
# val_dataloader = DataLoader(dataset=val_dataset,
#                             batch_size=64,
#                             shuffle=True,
#                             num_workers=0)

# drive119
# train_dataset = MyDataset("./data/train11_9.txt", train_transform)
# train_dataloader = DataLoader(dataset=train_dataset,
#                               batch_size=32,
#                               shuffle=True,
#                               num_workers=0)
# val_dataset = MyDataset("./data/val11_9.txt", val_transform)
# val_dataloader = DataLoader(dataset=val_dataset,
#                               batch_size=32,
#                               shuffle=True,
#                               num_workers=0)

# # drive224 将drive119图片预处理为224,提高速度
# train_dataset = MyDataset("./data/train224.txt", train_transform)
# train_dataloader = DataLoader(dataset=train_dataset,
#                               batch_size=32,
#                               shuffle=True,
#                               num_workers=0)
# train_dataset = MyDataset("./data/kgAddmy_add.txt", train_transform)
# train_dataloader = DataLoader(dataset=train_dataset,
#                               batch_size=64,
#                               shuffle=True,
#                               num_workers=0)
# val_dataset = MyDataset("./data/val224.txt", val_transform)
# val_dataset = MyDataset("./data/test11_9s.txt", val_transform)
# val_dataloader = DataLoader(dataset=val_dataset,
#                               batch_size=32,
#                               shuffle=True,
#                               num_workers=0)
#
# test_dataset = MyDataset("./data/test11_9s.txt", val_transform)
# test_dataloader = DataLoader(dataset=test_dataset,
#                               batch_size=32,
#                               shuffle=True,
#                               num_workers=0)

# # dataset 11_16
# # train_dataset = MyDataset("data/kgAddmy_add.txt", train_transform)
# # train_dataset = MyDataset("data/total_train.txt", train_transform)
# # train_dataset = MyDataset("data/train224_116_119.txt", train_transform)
# # train_dataset = MyDataset("data/txt/116_119trainAddcrop224.txt", train_transform)
# train_dataset = MyDataset("data/txt/116_119trainAddcrop224_kg.txt", train_transform)
# # train_dataset = MyDataset("data/txt/116_traincrop224.txt", train_transform)
# # train_dataset = MyDataset("data/txt/116_119traincrop224.txt", train_transform)
# # train_dataset = MyDataset("data/txt/116_119traincrop224_kg.txt", train_transform)
# # train_dataset = MyDataset("data/txt/116_119traincrop224_kg_auc2.txt", train_transform)
# # train_dataset = MyDataset("data/train224_11_16_train.txt", train_transform)
# train_dataloader = DataLoader(dataset=train_dataset,
#                               batch_size=64,
#                               # batch_size=32,
#                               shuffle=True,
#                               num_workers=0)
# val_dataset = MyDataset("data/test224_11_16.txt", val_transform)
# # val_dataset = MyDataset("data/txt/116_testcrop224.txt", val_transform)
# # val_dataset = MyDataset("data/train224_11_16_val.txt", val_transform)
# val_dataloader = DataLoader(dataset=val_dataset,
#                             batch_size=64,
#                             shuffle=True,
#                             num_workers=0)


# test_dataset = MyDataset("data/test224_11_16.txt", val_transform)
# test_dataloader = DataLoader(dataset=test_dataset,
#                               batch_size=32,
#                               shuffle=True,
#                               num_workers=0)
#
# # # dataset kg_total
# train_dataset = MyDataset("data/kg_total_add_t.txt", train_transform)
# train_dataloader = DataLoader(dataset=train_dataset,
#                               batch_size=128,
#                               shuffle=True,
#                               num_workers=0)
# val_dataset = MyDataset("data/test224_11_16.txt", val_transform)
# # val_dataset = MyDataset("data/kg_total_add_v.txt", val_transform)
# val_dataloader = DataLoader(dataset=val_dataset,
#                               batch_size=128,
#                               shuffle=True,
#                               num_workers=0)

# # dataset 12_23
# train_dataset = MyDataset("data/txt/12_23_2_train224.txt", train_transform)
# val_dataset = MyDataset("data/txt/12_23_2_test224.txt", val_transform)
# train_dataset = MyDataset("data/txt/12_23_1_train224.txt", train_transform)
# val_dataset = MyDataset("data/txt/12_23_1_test224.txt", val_transform)
# train_dataset = MyDataset("data/txt/12_23_12_train224.txt", train_transform)
# val_dataset = MyDataset("data/txt/12_23_12_test224.txt", val_transform)
# train_dataset = MyDataset("data/txt/12_23_12_addpre_train224.txt", train_transform)
# val_dataset = MyDataset("data/txt/12_23_12_addpre_test224.txt", val_transform)
# train_dataset = MyDataset("data/txt/12_23_12_addpre_train224_kg2my.txt", train_transform)
# val_dataset = MyDataset("data/txt/12_23_12_addpre_test224.txt", val_transform)
# train_dataset = MyDataset("data/txt/12_23_12_addpre_train224_kg2my_aucv2_my.txt", train_transform)
# val_dataset = MyDataset("data/txt/12_23_12_addpre_test224.txt", val_transform)

# crop 12_23
# train_dataset = MyDataset("data/txt/12_23_12_addpre_train_crop224.txt", train_transform)
# train_dataset = MyDataset("data/txt/12_23_12_addpre_train_crop224_kg2my.txt", train_transform)
# train_dataset = MyDataset("data/txt/12_23_12_addpre_train_crop224_kg2my_aucv2_my.txt", train_transform)
# val_dataset = MyDataset("data/txt/12_23_12_addpre_test_crop224.txt", val_transform)
# train_dataset = MyDataset("data/txt/12_23_12_addpre_train224_addcrop.txt", train_transform)
# train_dataset = MyDataset("data/txt/12_23_12_addpre_train224_kg2my_aucv2_my_addcrop.txt", train_transform)
# val_dataset = MyDataset("data/txt/12_23_12_addpre_test224_addcrop.txt", val_transform)


# # class6
# train_dataset = MyDataset("data/txt6/12_23_12_addpre_train224_6.txt", train_transform)
# val_dataset = MyDataset("data/txt6/12_23_12_addpre_test224_6.txt", val_transform)
# train_dataset = MyDataset("data/txt6/12_23_12_addpre_train224_addcrop_6.txt", train_transform)
# val_dataset = MyDataset("data/txt6/12_23_12_addpre_test224_addcrop_6.txt", val_transform)
# train_dataset = MyDataset("data/txt6/12_23_12_addpre_train224_kg2my_aucv2_my_addcrop_6.txt", train_transform)
# val_dataset = MyDataset("data/txt6/12_23_12_addpre_test224_addcrop_6.txt", val_transform)
# train_dataset = MyDataset("data/txt6/12_23_12_addpre_train224_kg2my_aucv2_my_6.txt", train_transform)
# val_dataset = MyDataset("data/txt6/12_23_12_addpre_test224_6.txt", val_transform)

# class7
# train_dataset = MyDataset("data/txt7/12_23_12_addpre_train224_7.txt", train_transform)
# val_dataset = MyDataset("data/txt7/12_23_12_addpre_test224_7.txt", val_transform)
# train_dataset = MyDataset("data/txt7/12_23_12_addpre_train224_addcrop_7.txt", train_transform)
# val_dataset = MyDataset("data/txt7/12_23_12_addpre_test224_addcrop_7.txt", val_transform)
# train_dataset = MyDataset("data/txt7/12_23_12_addpre_train224_kg2my_aucv2_my_addcrop_7.txt", train_transform)
# val_dataset = MyDataset("data/txt7/12_23_12_addpre_test224_addcrop_7.txt", val_transform)
# train_dataset = MyDataset("data/txt7/12_23_12_addpre_train224_kg2my_aucv2_my_7.txt", train_transform)
# val_dataset = MyDataset("data/txt7/12_23_12_addpre_test224_7.txt", val_transform)

# txt_raw
# train_dataset = MyDataset("data/txt_raw/total_train.txt", train_transform)
# val_dataset = MyDataset("data/txt_raw/total_test.txt", val_transform)

# train_dataset = MyDataset("data/txt_raw/total_train_c6.txt", train_transform)
# val_dataset = MyDataset("data/txt_raw/total_test_c6.txt", val_transform)
#
# train_dataset = MyDataset("data/txt_raw/total_train_c7.txt", train_transform)
# val_dataset = MyDataset("data/txt_raw/total_test_c7.txt", val_transform)



#3_23
# train_dataset = MyDataset("data/txt_3_23/bus_train.txt", train_transform)
# val_dataset = MyDataset("data/txt_3_23/bus_test.txt", val_transform)

# train_dataset = MyDataset("data/txt_3_23/he_train.txt", train_transform)
# val_dataset = MyDataset("data/txt_3_23/he_test.txt", val_transform)

# train_dataset = MyDataset("data/txt_3_23/wen_train.txt", train_transform)
# val_dataset = MyDataset("data/txt_3_23/wen_test.txt", val_transform)

# train_dataset = MyDataset("data/txt_3_23/he_wen_train.txt", train_transform)
# val_dataset = MyDataset("data/txt_3_23/he_wen_test.txt", val_transform)

# # 3_25
# train_dataset = MyDataset("data/txt_3_25/train325.txt", train_transform)
# val_dataset = MyDataset("data/txt_3_25/test325.txt", val_transform)

# # 3_25 crop
# train_dataset = MyDataset("data/txt_3_25/train325_crop.txt", train_transform)
# val_dataset = MyDataset("data/txt_3_25/test325_crop.txt", val_transform)

# 3-25_all
# train_dataset = MyDataset("data/txt_3_25/train325_all.txt", train_transform)
# val_dataset = MyDataset("data/txt_3_25/test325_all.txt", val_transform)

# # 3-25_all crop
# train_dataset = MyDataset("data/txt_raw_crop/total_train_crop.txt", train_transform)
# val_dataset = MyDataset("data/txt_raw_crop/total_test_crop.txt", val_transform)

# all class7_2
# train_dataset = MyDataset("data/txt_raw/total_train_c7_2.txt", train_transform)
# val_dataset = MyDataset("data/txt_raw/total_test_c7_2.txt", val_transform)

# all class72_crop
# train_dataset = MyDataset("data/txt_raw_crop/total_train_crop_72.txt", train_transform)
# val_dataset = MyDataset("data/txt_raw_crop/total_test_crop_72.txt", val_transform)

# # all class73_crop
# train_dataset = MyDataset("data/txt_raw_crop/total_train_crop_73.txt", train_transform)
# val_dataset = MyDataset("data/txt_raw_crop/total_test_crop_73.txt", val_transform)

# # all class8
# train_dataset = MyDataset("data/txt_raw/total_train_8.txt", train_transform)
# val_dataset = MyDataset("data/txt_raw/total_test_8.txt", val_transform)

# # all class8_crop
# train_dataset = MyDataset("data/txt_raw_crop/total_train_crop_8.txt", train_transform)
# val_dataset = MyDataset("data/txt_raw_crop/total_test_crop_8.txt", val_transform)

# all class9
# train_dataset = MyDataset("data/txt_raw/total_train.txt", train_transform)
# val_dataset = MyDataset("data/txt_raw/total_test.txt", val_transform)

# # # all class9_crop
train_dataset = MyDataset("data/txt_raw_crop/total_train_crop.txt", train_transform)
val_dataset = MyDataset("data/txt_raw_crop/total_test_crop.txt", val_transform)

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=64,
                              shuffle=True,
                              num_workers=0)
val_dataloader = DataLoader(dataset=val_dataset,
                              batch_size=64,
                              shuffle=True,
                              num_workers=0)

# train_dataloader = DataLoader(dataset=train_dataset,
#                               batch_size=32,
#                               shuffle=True,
#                               num_workers=0)
# val_dataloader = DataLoader(dataset=val_dataset,
#                               batch_size=32,
#                               shuffle=True,
#                               num_workers=0)


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

# kaggle v1
# savepath = 'checkpoint/kaggle/v1/mobilenetv2/pre/000/' # randcrop 16 rotation 10 colorjit 0.5

# savepath = 'checkpoint/kaggle/v1/mobilenetv2/nopre/000/' # randcrop 16 rotation 10 colorjit 0.5

# kaggle v2
# savepath = 'checkpoint/kaggle/v2/mobilenetv2/000/' # no  augment
# savepath = 'checkpoint/kaggle/v2/mobilenetv2/111/' # rotation 10
# savepath = 'checkpoint/kaggle/v2/mobilenetv2/222/' # randcrop 16
# savepath = 'checkpoint/kaggle/v2/mobilenetv2/333/' # randcrop 16 rotation 10
# savepath = 'checkpoint/kaggle/v2/mobilenetv2/444/' # randcrop 16 rotation 10 colorjit 0.5
# savepath = 'checkpoint/kaggle/v2/mobilenetv2/555/' # randcrop 16 rotation 10 colorjit 0.2
# savepath = 'checkpoint/kaggle/v2/mobilenetv2/666/' # randcrop 16 rotation 30 colorjit 0.5
# savepath = 'checkpoint/kaggle/v2/mobilenetv2/777/' # randcrop 16 rotation 20 colorjit 0.5
# savepath = 'checkpoint/kaggle/v2/mobilenetv2/888/' #  randcrop 16 rotation 10 colorjit 0.5

# savepath = 'checkpoint/kaggle/v2/mobilenetv2/pre/000/' #  randcrop 16 rotation 10 colorjit 0.5
# savepath = 'checkpoint/kaggle/v2/mobilenetv2/pre/111/' #  randcrop 16 rotation 20 colorjit 0.5

# AUC v2
# savepath = 'checkpoint/AUC/v2/mobilenetv2/000/'
# savepath = 'checkpoint/AUC/v2/mobilenetv2/111/'
# savepath = 'checkpoint/AUC/v2/mobilenetv2/222/'
# savepath = 'checkpoint/AUC/v2/mobilenetv2/333/'
# savepath = 'checkpoint/AUC/v2/mobilenetv2/444/'

# savepath = 'checkpoint/AUC/v2/mobilenetv2/pre/444/' #  randcrop 16 rotation 10 colorjit 0.5
# savepath = 'checkpoint/AUC/v2/mobilenetv2/pre/000/' #  randcrop 16 rotation 10 colorjit 0.5
# savepath = 'checkpoint/AUC/v2/mnext/000/'

# savepath = 'checkpoint/AUC/v2/resnet50/000/'
# savepath = 'checkpoint/AUC/v2/resnet50/111/'

# AUC v1
# savepath = 'checkpoint/AUC/v1/mobilenetv2/000/'
# savepath = 'checkpoint/AUC/v1/mobilenetv2/111/'
# savepath = 'checkpoint/AUC/v1/mobilenetv2/222/'
# savepath = 'checkpoint/AUC/v1/mobilenetv2/333/'
# savepath = 'checkpoint/AUC/v1/mobilenetv2/444/'

# savepath = 'checkpoint/AUC/v1/mnext/000/'

# savepath = 'checkpoint/AUC/v1/ghostnet/000/'

# savepath = 'checkpoint/AUC/v1/resnet50/000/'

# 11_16
# savepath = 'checkpoint/data_11_16/mobilenetv2/nopre/333/' #  randcrop 16 rotation 10 colorjit 0.5

# savepath = 'checkpoint/data_11_16/mobilenetv2/pre/000/' #  randcrop 16 rotation 10 colorjit 0.5
# savepath = 'checkpoint/data_11_16/mobilenetv2/pre/111/' #  randcrop 16 rotation 10 colorjit 0.5
# savepath = 'checkpoint/data_11_16/mobilenetv2/pre/222/' #  randcrop 16 rotation 10 colorjit 0.5
# savepath = 'checkpoint/data_11_16/mobilenetv2/pre/333/' #  randcrop 16 rotation 10 colorjit 0.5



# savepath = 'checkpoint/data_11_16/ghostnet/pre/000/' #  randcrop 16 rotation 10 colorjit 0.5  change_lr8 1e-3
# savepath = 'checkpoint/data_11_16/ghostnet/pre/111/' #  randcrop 16 rotation 10 colorjit 0.5  change_lr6 1e-3
# savepath = 'checkpoint/data_11_16/ghostnet/pre/222/' #  randcrop 16 rotation 10 colorjit 0.5  change_lr6 1e-3 addcrop
# savepath = 'checkpoint/data_11_16/ghostnet/pre/333/' #  randcrop 16 rotation 10 colorjit 0.5  change_lr6 1e-3 addcrop

# savepath = 'checkpoint/data_11_16/mnext/pre/000/' #  randcrop 16 rotation 10 colorjit 0.5 change_lr8
# savepath = 'checkpoint/data_11_16/mnext/pre/111/' #  randcrop 16 rotation 10 colorjit 0.5 change_lr8
# savepath = 'checkpoint/data_11_16/mnext/pre/222/' #  randcrop 16 rotation 10 colorjit 0.5  change_lr6 1e-3 addcrop

# crop224
# savepath = 'checkpoint/data_11_16/mobilenetv2/pre/444/' #  randcrop 16 rotation 10 colorjit 0.5
# savepath = 'checkpoint/data_11_16/mobilenetv2/pre/555/' #  randcrop 16 rotation 10 colorjit 0.5
# savepath = 'checkpoint/data_11_16/mobilenetv2/pre/666/' #  randcrop 16 rotation 10 colorjit 0.5 160
# savepath = 'checkpoint/data_11_16/mobilenetv2/pre/777/' #  randcrop 16 rotation 10 colorjit 0.5  add kg
# savepath = 'checkpoint/data_11_16/mobilenetv2/pre/888/' #  randcrop 16 rotation 10 colorjit 0.5  add kg 160
# savepath = 'checkpoint/data_11_16/mobilenetv2/pre/999/' #  randcrop 16 rotation 10 colorjit 0.5  add kg auc2

# savepath = 'checkpoint/data_11_16/mobilenetv2/pre/0/000/' #  16 rotation 10 colorjit 0.5  224 116_119
# savepath = 'checkpoint/data_11_16/mobilenetv2/pre/0/111/' #  randcrop 16 rotation 10 colorjit 0.5  224 116_119 add crop
# savepath = 'checkpoint/data_11_16/mobilenetv2/pre/0/222/'  # add andcrop 16 rotation 10 colorjit 0.5  224 116_119 add crop kg
# savepath = 'checkpoint/data_11_16/mobilenetv2/pre/0/333/' #  randcrop 16 rotation 10 colorjit 0.5
# savepath = 'checkpoint/data_11_16/mobilenetv2/pre/0/444/' #  randcrop 16 rotation 10 colorjit 0.5
# savepath = 'checkpoint/data_11_16/mobilenetv2/pre/0/555/' #  randcrop 16 rotation 10 colorjit 0.5
# savepath = 'checkpoint/data_11_16/mobilenetv2/pre/0/666/' #  randcrop 16 rotation 10 colorjit 0.5

# savepath = 'checkpoint/data_11_16/shufflenetv2/pre/000/'  # add andcrop 16 rotation 10 colorjit 0.5  224 116_119 add crop kg

# savepath = 'checkpoint/data_11_16/shufflenetv2/pre/000/' #  randcrop 16 rotation 10 colorjit 0.5
# savepath = 'checkpoint/data_11_16/shufflenetv2/pre/111/' #  randcrop 16 rotation 10 colorjit 0.5
# savepath = 'checkpoint/data_11_16/shufflenetv2/pre/222/' #  randcrop 16 rotation 10 colorjit 0.5
# savepath = 'checkpoint/data_11_16/shufflenetv2/pre/333/' #  randcrop 16 rotation 10 colorjit 0.5
# savepath = 'checkpoint/data_11_16/shufflenetv2/pre/444/' #  randcrop 16 rotation 10 colorjit 0.5
# savepath = 'checkpoint/data_11_16/shufflenetv2/pre/555/' #  randcrop 16 rotation 10 colorjit 0.5

# imagenet
# savepath = 'checkpoint/imagenet/imagenet100/mobilenetv2/000/' #  randcrop 16 rotation 10 colorjit 0.5
# savepath = 'checkpoint/imagenet/imagenet100/mobilenetv2/111/' #  randcrop 16 rotation 10 colorjit 0.5

# savepath = 'checkpoint/imagenet/imagenet100/ghostnet/000/' #  randcrop 16 rotation 10 colorjit 0.5

# savepath = 'checkpoint/imagenet/imagenet100/mnext/000/' #  randcrop 16 rotation 10 colorjit 0.5

# savepath = 'checkpoint/imagenet/imagenet100/my_mnextv2/000/' #  randcrop 16 rotation 10 colorjit 0.5

# dataset 12_23
# savepath = 'checkpoint/data_12_23/mobilenetv2/000/' #  randcrop 16 rotation 10 colorjit 0.5 12_23_2 change_lr6 1e-3
# savepath = 'checkpoint/data_12_23/mobilenetv2/111/' #  randcrop 16 rotation 10 colorjit 0.5 12_23_1 change_lr6 1e-3
# savepath = 'checkpoint/data_12_23/mobilenetv2/222/' #  randcrop 16 rotation 10 colorjit 0.5 12_23_12 change_lr6 1e-3
# savepath = 'checkpoint/data_12_23/mobilenetv2/333/' #  randcrop 16 rotation 10 colorjit 0.5 12_23_12_addpre change_lr6 1e-3
# savepath = 'checkpoint/data_12_23/mobilenetv2/444/' #  randcrop 16 rotation 10 colorjit 0.5 12_23_12_addpre kg2my change_lr6 1e-3
# savepath = 'checkpoint/data_12_23/mobilenetv2/555/' #  randcrop 16 rotation 10 colorjit 0.5 12_23_12_addpre change_lr9 5e-3
# savepath = 'checkpoint/data_12_23/mobilenetv2/666/' #  randcrop 16 rotation 10 colorjit 0.5 12_23_12_addpre change_lr9 1e-3
# savepath = 'checkpoint/data_12_23/mobilenetv2/777/' #  randcrop 16 rotation 10 colorjit 0.5 12_23_12_addpre kg2my change_lr9 1e-3
# savepath = 'checkpoint/data_12_23/mobilenetv2/888/' #  randcrop 16 rotation 10 colorjit 0.5 12_23_12_addpre kg2my aucv2 change_lr9 1e-3
# savepath = 'checkpoint/data_12_23/mobilenetv2/0/999/' #  randcrop 16 rotation 10 colorjit 0.5 12_23_12_addpre kg2my aucv2 change_lr9 sgd 1e-3
# savepath = 'checkpoint/data_12_23/mobilenetv2/0/000/' #  randcrop 16 rotation 10 colorjit 0.5 12_23_12_addpre kg2my aucv2 change_lr9 sgd 1e-1
# savepath = 'checkpoint/data_12_23/mobilenetv2/0/111/' #  randcrop 16 rotation 10 colorjit 0.5 12_23_12_addpre kg2my aucv2 change_lr9
# savepath = 'checkpoint/data_12_23/mobilenetv2/0/222/' #  flip randcrop 16 rotation 10 colorjit 0.5 12_23_12_addpre change_lr9
# savepath = 'checkpoint/data_12_23/mobilenetv2/0/333/' #  randcrop 16 rotation 10 colorjit 0.5 12_23_12_addpre kg2my aucv2 change_lr9
# savepath = 'checkpoint/data_12_23/mobilenetv2/0/444/' # change_lr10 brightness=0.8 mypre
# savepath = 'checkpoint/data_12_23/mobilenetv2/0/555/' # change_lr10 brightness=0.8 mypre

# savepath = 'checkpoint/data_12_23/mobilenetv2/nopre/000/' #  nopre randcrop 16 rotation 10 colorjit 0.5 12_23_12_change_lr9 1e-3
# savepath = 'checkpoint/data_12_23/mobilenetv2/nopre/111/' #  pre randcrop 16 rotation 10 colorjit 0.5 12_23_12_change_lr9 1e-3

# savepath = 'checkpoint/data_12_23/mobilenetv2/nopre/0/000/' #  nopre randcrop 16 rotation 10 colorjit 0.5 12_23_12_change_lr9 1e-2 cbam c=64
# savepath = 'checkpoint/data_12_23/mobilenetv2/nopre/0/111/' #  nopre randcrop 16 rotation 10 colorjit 0.5 12_23_12_change_lr9 1e-3 cbam c=64

# savepath = 'checkpoint/data_12_23/mobilenetv2/nopre/1/000/' #  nopre randcrop 16 rotation 10 colorjit 0.5 12_23_12_change_lr9 1e-2 cbam c=16
# savepath = 'checkpoint/data_12_23/mobilenetv2/nopre/1/111/' #  pre randcrop 16 rotation 10 colorjit 0.5 12_23_12_change_lr9 1e-2 cbam c=16
# savepath = 'checkpoint/data_12_23/mobilenetv2/nopre/1/222/' #  pre randcrop 16 rotation 10 colorjit 0.5 12_23_12_ addpre kg2my aucv2 change_lr9 1e-2 cbam c=16
# savepath = 'checkpoint/data_12_23/mobilenetv2/nopre/1/333/' #  pre randcrop 16 rotation 10 colorjit 0.5 12_23_12_ addpre kg2my aucv2 change_lr9 1e-2 cbam c=64

# savepath = 'checkpoint/data_12_23/mnext/000/' #  randcrop 16 rotation 10 colorjit 0.5 12_23_12_addpre kg2my aucv2 change_lr9
# savepath = 'checkpoint/data_12_23/mnext/111/' #  randcrop 16 rotation 10 colorjit 0.5 12_23_12_addpre change_lr9


# dataset 12_23
# savepath = 'checkpoint/data_12_23/mobilenetv2/crop/000/' #  crop randcrop 16 rotation 10 colorjit 0.5 12_23_2 change_lr6 1e-3
# savepath = 'checkpoint/data_12_23/mobilenetv2/crop/111/' #  crop randcrop 16 rotation 10 colorjit 0.5 12_23_2 change_lr6 1e-3 addpre kg2my
# savepath = 'checkpoint/data_12_23/mobilenetv2/crop/222/' #  crop randcrop 16 rotation 10 colorjit 0.5 12_23_2 change_lr6 1e-3 addpre kg2my aucv2
# savepath = 'checkpoint/data_12_23/mobilenetv2/crop/333/' #  randcrop 16 rotation 10 colorjit 0.5 12_23_2 change_lr6 1e-3 addpre addcrop
# savepath = 'checkpoint/data_12_23/mobilenetv2/crop/444/' #  randcrop 16 rotation 10 colorjit 0.5 12_23_2 change_lr6 1e-3 addpre kg2my aucv2 addcrop



# # dataset class6
# # savepath = 'checkpoint/data_12_23/class6/mobilenetv2/000/' #  randcrop 16 rotation 10 colorjit 0.5 12_23_2 change_lr9 1e-3
# savepath = 'checkpoint/data_12_23/class6/mobilenetv2/111/' #  randcrop 16 rotation 10 colorjit 0.5 12_23_2 change_lr9 1e-3 addcrop
# savepath = 'checkpoint/data_12_23/class6/mobilenetv2/222/' #  randcrop 16 rotation 10 colorjit 0.5 12_23_2 change_lr9 1e-3 addpre kg2my aucv2 addcrop
# savepath = 'checkpoint/data_12_23/class6/mobilenetv2/333/' #  randcrop 16 rotation 10 colorjit 0.5 12_23_2 change_lr9 1e-3 a ddpre kg2my aucv2

# dataset class7
# savepath = 'checkpoint/data_12_23/class7/mobilenetv2/000/' #  randcrop 16 rotation 10 colorjit 0.5 12_23_2 change_lr9 1e-3
# savepath = 'checkpoint/data_12_23/class7/mobilenetv2/111/' #  randcrop 16 rotation 10 colorjit 0.5 12_23_2 change_lr91e-3 addcrop
# savepath = 'checkpoint/data_12_23/class7/mobilenetv2/222/' #  randcrop 16 rotation 10 colorjit 0.5 12_23_2 change_lr9 1e-3 addpre kg2my aucv2 addcrop
# savepath = 'checkpoint/data_12_23/class7/mobilenetv2/333/' #  randcrop 16 rotation 10 colorjit 0.5 12_23_2 change_lr9 1e-3 a ddpre kg2my aucv2

# dataset txt_raw
# savepath = 'checkpoint/txt_raw/mobilenetv2/224/000/' # change_lr9 totol_test
# savepath = 'checkpoint/txt_raw/mobilenetv2/224/111/' # change_lr9 totol_test
# savepath = 'checkpoint/txt_raw/mobilenetv2/224/222/' # change_lr9 totol_test 240-》224
# savepath = 'checkpoint/txt_raw/mobilenetv2/224/333/' # change_lr9 totol_test
# savepath = 'checkpoint/txt_raw/mobilenetv2/224/444/' # change_lr9 totol_test 240-》224
# savepath = 'checkpoint/txt_raw/mobilenetv2/320/000/' # change_lr9 totol_test

# savepath = 'checkpoint/txt_raw/mobilenetv2/000/' # change_lr10 totol_test 240-》224
# savepath = 'checkpoint/txt_raw/mobilenetv2/111/' # change_lr10 totol_test
# savepath = 'checkpoint/txt_raw/mobilenetv2/222/' # change_lr10 totol_test brightness=0.8

# savepath = 'checkpoint/txt_raw/mobilenetv2/class6/000/' # change_lr10 totol_test 240-》224 c9
# savepath = 'checkpoint/txt_raw/mobilenetv2/class6/111/' # change_lr10 totol_test 240-》224
# savepath = 'checkpoint/txt_raw/mobilenetv2/class6/222/' # change_lr10 totol_test
# savepath = 'checkpoint/txt_raw/mobilenetv2/class6/333/' # change_lr10 totol_test brightness=0.8

# savepath = 'checkpoint/txt_raw/mobilenetv2/class7/000/' # change_lr10 totol_test 240-》224
# savepath = 'checkpoint/txt_raw/mobilenetv2/class7/111/' # change_lr10 totol_test
# savepath = 'checkpoint/txt_raw/mobilenetv2/class7/222/' # change_lr10 totol_test brightness=0.8



# 3_23
# savepath = 'checkpoint/data_3_23/mobilenetv2/000/' # change_lr10 brightness=0.8
# savepath = 'checkpoint/data_3_23/mobilenetv2/111/' # change_lr10 brightness=0.5 240>>224
# savepath = 'checkpoint/data_3_23/mobilenetv2/222/' # change_lr10 no data augment

# savepath = 'checkpoint/data_3_23/mobilenetv2/he/000/' # change_lr10 brightness=0.5 240>>224
# savepath = 'checkpoint/data_3_23/mobilenetv2/he/111/' # change_lr10 brightness=0.8

# savepath = 'checkpoint/data_3_23/mobilenetv2/wen/000/' # change_lr10 brightness=0.5 240>>224
# savepath = 'checkpoint/data_3_23/mobilenetv2/wen/111/' # change_lr10 brightness=0.8

# savepath = 'checkpoint/data_3_23/mobilenetv2/he_wen/000/' # change_lr10 brightness=0.5 240>>224
# savepath = 'checkpoint/data_3_23/mobilenetv2/he_wen/111/' # change_lr10 brightness=0.8
# savepath = 'checkpoint/data_3_23/mobilenetv2/he_wen/222/' # change_lr10 brightness=0.8 nopre
# savepath = 'checkpoint/data_3_23/mobilenetv2/he_wen/333/' # change_lr10 brightness=0.8 mypre
# savepath = 'checkpoint/data_3_23/mobilenetv2/he_wen/333/' # change_lr10 brightness=0.5 mypre
# savepath = 'checkpoint/data_3_23/mobilenetv2/he_wen/444/' # change_lr19 brightness=0.5

# 325
# savepath = 'checkpoint/data_3_25/mobilenetv2/000/' # change_lr10 brightness=0.5
# savepath = 'checkpoint/data_3_25/mobilenetv2/111/' # change_lr9 brightness=0.5

# 325 crop
# savepath = 'checkpoint/data_3_25_crop/mobilenetv2/000/' # change_lr10 brightness=0.5
# savepath = 'checkpoint/data_3_25_crop/mobilenetv2/111/' # change_lr9 brightness=0.5
# savepath = 'checkpoint/data_3_25_crop/mobilenetv2/222/' # change_lr9 brightness=0.5 nopre


# 325 all
# savepath = 'checkpoint/data_3_25_all/mobilenetv2/111/' # change_lr9 brightness=0.5
# savepath = 'checkpoint/data_3_25_all/mobilenetv2/222/' # change_lr9 brightness=0.5 nopre
# savepath = 'checkpoint/data_3_25_all/mobilenetv2/333/' # change_lr9 brightness=0.5 crop
# savepath = 'checkpoint/data_3_25_all/mobilenetv2/444/' # change_lr9 brightness=0.5  crop nopre

# savepath = 'checkpoint/data_3_25_all/mobilenetv2/555/' # change_lr9 brightness=0.5 crop 240->224
# savepath = 'checkpoint/data_3_25_all/mobilenetv2/666/' # change_lr9 brightness=0.5 crop 240->224 nopre
# savepath = 'checkpoint/data_3_25_all/mobilenetv2/777/' # change_lr9 brightness=0.5 240->224
# savepath = 'checkpoint/data_3_25_all/mobilenetv2/888/' # change_lr9 brightness=0.5 240->224 nopre

# savepath = 'checkpoint/data_3_25_all/mobilenetv2/16/000/' # 16 milestones brightness=0.5 240->224
# savepath = 'checkpoint/data_3_25_all/mobilenetv2/16/111/' # 16 milestones brightness=0.5 240->224 crop

# savepath = 'checkpoint/data_3_25_all/mobilenetv2/18/000/' # 18 change_lr12 brightness=0.5 240->224 crop

# savepath = 'checkpoint/data_3_25_all/mobilenetv2/class9/160/000/' # 18 change_lr12 milestones brightness=0.5 160 no randcrop
# savepath = 'checkpoint/data_3_25_all/mobilenetv2/class9/160/111/' # 18 change_lr12 milestones brightness=0.5 160 crop no randcrop

# savepath = 'checkpoint/data_3_25_all/mobilenetv2/class9/224/000/' # 18 change_lr12 milestones brightness=0.5 224 no randcrop
# savepath = 'checkpoint/data_3_25_all/mobilenetv2/class9/224/111/' # 18 change_lr12 milestones brightness=0.5 224 crop no randcrop
# savepath = 'checkpoint/data_3_25_all/mobilenetv2/class9/224/222/' # 18 change_lr12 milestones brightness=0.5 224 no randcrop
# savepath = 'checkpoint/data_3_25_all/mobilenetv2/class9/224/333/' # 18 change_lr12 milestones brightness=0.5 224 crop no randcrop

# savepath = 'checkpoint/data_3_25_all/mobilenetv2/class9/224/444/' # 18 change_lr12 milestones brightness=0.5 224 crop 240->224
savepath = 'checkpoint/data_3_25_all/mobilenetv2/class9/224/555/' # 18 change_lr12 milestones brightness=0.5 224 crop 240->224

# savepath = 'checkpoint/data_3_25_all/mobilenetv2/class9/320/000/' # 18 change_lr12 milestones brightness=0.5 320 no randcrop
# savepath = 'checkpoint/data_3_25_all/mobilenetv2/class9/320/111/' # 18 change_lr12 milestones brightness=0.5 320 crop no randcrop

# class7_2
# savepath = 'checkpoint/data_3_25_all/mobilenetv2/class7_2/000/' # 16 milestones brightness=0.5 240->224
# savepath = 'checkpoint/data_3_25_all/mobilenetv2/class7_2/111/' # 16 milestones brightness=0.5 240->224 crop
# savepath = 'checkpoint/data_3_25_all/mobilenetv2/class7_2/222/' # change_lr10  brightness=0.5 240->224 crop
# savepath = 'checkpoint/data_3_25_all/mobilenetv2/class7_2/333/' # change_lr9  brightness=0.5 240->224 crop

# class7_3
# savepath = 'checkpoint/data_3_25_all/mobilenetv2/class7_3/000/' # 16 milestones brightness=0.5 240->224 crop
# savepath = 'checkpoint/data_3_25_all/mobilenetv2/class7_3/111/' # change_lr10  brightness=0.5 240->224 crop

# class8
# savepath = 'checkpoint/data_3_25_all/mobilenetv2/class8/000/' # 16 milestones brightness=0.5 240->224 crop
# savepath = 'checkpoint/data_3_25_all/mobilenetv2/class8/111/' # change_lr12  brightness=0.5 240->224 crop
# savepath = 'checkpoint/data_3_25_all/mobilenetv2/class8/222/' # 16 milestones brightness=0.5 240->224
# savepath = 'checkpoint/data_3_25_all/mobilenetv2/class8/333/' # 18 change_lr12 milestones brightness=0.5 240->224 crop
# savepath = 'checkpoint/data_3_25_all/mobilenetv2/class8/444/' # 18 change_lr12 milestones brightness=0.5 240->224
# savepath = 'checkpoint/data_3_25_all/mobilenetv2/class8/555/' # 18 change_lr12 milestones brightness=0.5 240->224 crop no randcrop
# savepath = 'checkpoint/data_3_25_all/mobilenetv2/class8/666/' # 18 change_lr12 milestones brightness=0.5 240->224 crop
# savepath = 'checkpoint/data_3_25_all/mobilenetv2/class8/777/' # 18 change_lr12 milestones brightness=0.5 240->224 crop nopre

# class8 320
# savepath = 'checkpoint/data_3_25_all/mobilenetv2/class8/320/000/' # 18 change_lr12 milestones brightness=0.5 320 crop no randcrop
# savepath = 'checkpoint/data_3_25_all/mobilenetv2/class8/320/111/' # 18 change_lr12 milestones brightness=0.5 320 randcrop

# class8 160
# savepath = 'checkpoint/data_3_25_all/mobilenetv2/class8/160/000/' # 18 change_lr12 milestones brightness=0.5 160 crop no randcrop
# savepath = 'checkpoint/data_3_25_all/mobilenetv2/class8/160/111/' # 18 change_lr12 milestones brightness=0.5 160 nocrop no randcrop


# resnext50
# savepath = 'checkpoint/data_3_25_all/resnext50/224/class8/000/' # 18 change_lr12 milestones brightness=0.5 240->224 crop

# mobilenetv3_small
# savepath = 'checkpoint/data_3_25_all/mobilenetv3_s/224/class8/000/' # 18 change_lr12 milestones brightness=0.5 240->224 crop
# savepath = 'checkpoint/data_3_25_all/mobilenetv3_s/224/class8/111/' # 18 change_lr12 milestones brightness=0.5 240->224

# savepath = 'checkpoint/data_3_25_all/mobilenetv3_s/224/class9/000/' # 18 change_lr12 milestones brightness=0.5 240->224 crop
# savepath = 'checkpoint/data_3_25_all/mobilenetv3_s/224/class9/111/' # 18 change_lr12 milestones brightness=0.5 240->224

# mobilenetv3_LARGE
# savepath = 'checkpoint/data_3_25_all/mobilenetv3_L/224/class8/000/' # 18 change_lr12 milestones brightness=0.5 240->224 crop
# savepath = 'checkpoint/data_3_25_all/mobilenetv3_L/224/class8/111/' # 18 change_lr12 milestones brightness=0.5 240->224

# savepath = 'checkpoint/data_3_25_all/mobilenetv3_L/320/class8/000/' # 18 change_lr12 milestones brightness=0.5 320 crop

# savepath = 'checkpoint/data_3_25_all/mobilenetv3_L/224/class9/000/' # 18 change_lr12 milestones brightness=0.5 240->224 crop
# savepath = 'checkpoint/data_3_25_all/mobilenetv3_L/224/class9/111/' # 18 change_lr12 milestones brightness=0.5 240->224

# savepath = 'checkpoint/data_3_25_all/mobilenetv3_L/320/class9/111/' # 18 change_lr12 milestones brightness=0.5 320

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
        # torch.save(state, savepath + 'mobilenetv2_1_imagenet_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'ghostnet_1_imagenet_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mnext_1_imagenet_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'my_mnextv2_1_imagenet_acc={:.4f}.pth'.format(acc))

        # torch.save(state, savepath + 'mobilenetv2_1_my_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'shufflenetv2_1_my_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'shufflenetv2_05_my_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'ghostnet_1_my_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mnext_1_my_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_1_kg1_acc={:.4f}.pth'.format(acc))

        # torch.save(state, savepath + 'mobilenetv2_1_kg2_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'resnet50_1_kg2_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mnext_1_kg2_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv3s_1_kg2_acc={:.4f}.pth'.format(acc))

        # torch.save(state, savepath + 'mobilenetv2_1_aucv2_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mnext_1_aucv2_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv3s_1_aucv2_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'resnet50_aucv2_acc={:.4f}.pth'.format(acc))

        # torch.save(state, savepath + 'mobilenetv2_1_aucv1_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mnext_1_aucv1_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv3s_1_aucv1_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'resnet50_1_aucv1_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'ghostnet_1_aucv1_acc={:.4f}.pth'.format(acc))

        # torch.save(state, savepath + 'mobilenetv2_1_12_23_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_cbam_1_12_23_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mnext_1_12_23_acc={:.4f}.pth'.format(acc))

        # torch.save(state, savepath + 'mobilenetv2_1_crop_acc={:.4f}.pth'.format(acc))

        # torch.save(state, savepath + 'mobilenetv2_1_c6_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_1_c7_acc={:.4f}.pth'.format(acc))

        # torch.save(state, savepath + 'mobilenetv2_224_crop_72_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_224_crop_73_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_224_crop_8_acc={:.4f}.pth'.format(acc))

        # torch.save(state, savepath + 'mobilenetv2_160_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_160_crop_acc={:.4f}.pth'.format(acc))

        # torch.save(state, savepath + 'mobilenetv2_224_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_224_crop_acc={:.4f}.pth'.format(acc))

        # torch.save(state, savepath + 'mobilenetv2_320_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_320_crop_acc={:.4f}.pth'.format(acc))

        # torch.save(state, savepath + 'mobilenetv2_320_crop_8_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_160_crop_8_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_160_8_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_224_8_acc={:.4f}.pth'.format(acc))

        # torch.save(state, savepath + 'mobilenetv2_224_9_acc={:.4f}.pth'.format(acc))
        torch.save(state, savepath + 'mobilenetv2_224_9_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mobilenetv2_320_acc={:.4f}.pth'.format(acc))

        # torch.save(state, savepath + 'resnext50_224_8_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mv3_s_224_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mv3_s_224_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mv3_s_224_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mv3_l_224_8_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mv3_s_224_9_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mv3_s_224_9_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mv3_s_224_8_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mv3_l_224_8_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mv3_l_224_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mv3_l_224_crop_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mv3_l_224_9_acc={:.4f}.pth'.format(acc))
        # torch.save(state, savepath + 'mv3_l_224_9_crop_acc={:.4f}.pth'.format(acc))
        best_val_acc = acc
    return average_loss, test_acc


# B0/000  kaggle dataset without val
# B0/111  kaggle dataset with val
# B0/222  my dataset with normal transfrom
# B0/333  my dataset with randomcrop random flip
# B0/444  my dataset with random flip
# B0/555  my dataset with random flip with val

# ghost_net/000  kaggle dataset with random flip w = 1
# ghost_net/111  kaggle dataset with random flip w = 0.5
# ghost_net/222  kaggle dataset with random flip w = 0.3
# ghost_net/333  my dataset with random flip w = 0.5
# ghost_net/444  my dataset with random flip w = 1
# ghost_net/555  my dataset no val with random flip w = 0.5
# ghost_net/666  kaggle dataset with random flip w = 0.1
# ghost_net/777  kaggle dataset with random flip w = 0.1

# mobilenetv2/000  my dataset with random flip
# mobilenetv2/111  kaggle dataset with random flip
# mobilenetv2/111  kaggle dataset with random flip w =0.5

# resnet18/000  my dataset with random flip
# resnet18/111  kaggle dataset with random flip


# resnext50/000  my dataset with random flip
# resnext50/111  kaggle dataset with random flip
# resnext50/222  my dataset with random flip with 111 pretrain
# resnext50/333 my dataset with random flip without pretrain
# resnext50/444 kaggle dataset with random flip without pretrain

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

    # num_classes = 6
    # num_classes = 7
    # num_classes = 8
    num_classes = 9
    net = models.mobilenet_v2(pretrained=False, num_classes=num_classes, width_mult=1.0)
    # # model_path = r"checkpoint/data_11_16/mobilenetv2/pre/555/mobilenetv2_1_my_acc=96.1749.pth" # crop 160=0.7486338797814208
    # # model_path = r"checkpoint/data_11_16/mobilenetv2/pre/222/mobilenetv2_1_my_acc=92.3497.pth"  # 160=0.5846994535519126
    # # model_path = r"checkpoint/data_11_16/mobilenetv2/pre/666/mobilenetv2_1_my_acc=95.6284.pth"  # 160=0.9562841530054644 224=0.7486338797814208
    # # model_path = r"checkpoint/data_11_16/mobilenetv2/pre/777/mobilenetv2_1_my_acc=95.6284.pth"  # 160=0.8142076502732241
    # # model_path = r"checkpoint/data_11_16/mobilenetv2/pre/0/111/mobilenetv2_1_my_acc=93.4426.pth"  # crop=0.9398907103825137 crop_160=0.907103825136612
    #
    # # model_path = r"checkpoint/data_12_23/mobilenetv2/222/mobilenetv2_1_12_23_acc=93.6898.pth"
    # # model_path = r"checkpoint/data_12_23/mobilenetv2/333/mobilenetv2_1_12_23_acc=89.9061.pth"
    # model_path = r"checkpoint/data_12_23/mobilenetv2/888/mobilenetv2_1_12_23_acc=91.6275.pth"
    # model_path = r"checkpoint/data_12_23/mobilenetv2/0/222/mnext_1_12_23_acc=88.2629.pth" # mobilenetv2
    # model_path = r"checkpoint/data_12_23/mobilenetv2/0/333/mobilenetv2_1_12_23_acc=84.8983.pth"
    # model_path = r"checkpoint/data_12_23/mobilenetv2/crop/333/mobilenetv2_1_crop_acc=90.8059.pth"
    # model_path = r"checkpoint/data_12_23/mobilenetv2/crop/444/mobilenetv2_1_crop_acc=90.9233.pth"
    # model_path = r"checkpoint/data_3_25_all/mobilenetv2/111/mobilenetv2_224_acc=85.6154.pth"

    model_path = r"checkpoint/data_3_25_all/mobilenetv2/class8/666/mobilenetv2_224_8_acc=90.5516.pth"

    model_path = r"checkpoint/data_3_25_all/mobilenetv2/555/mobilenetv2_224_crop_acc=89.1415.pth"

    model_path = r"checkpoint/data_3_25_all/mobilenetv2/111/mobilenetv2_224_acc=85.6154.pth"
    # class6
    # model_path = r"checkpoint/data_12_23/class6/mobilenetv2/222/mobilenetv2_1_c6_acc=95.1313.pth"

    # class7
    # model_path = r"checkpoint/data_12_23/class7/mobilenetv2/000/mobilenetv2_1_c7_acc=92.0188.pth"

    # net = mnext(num_classes=num_classes, width_mult=1.)
    # model_path = r"checkpoint/data_12_23/mnext/000/mnext_1_12_23_acc=92.1753.pth"
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
    # test_dataset = MyDataset("data/txt/116_testcrop224.txt", test_transform)
    # test_dataset = MyDataset("data/txt/119_testcrop224.txt", test_transform)
    # test_dataset = MyDataset("data/test224_11_16.txt", test_transform)
    # test_dataset = MyDataset("data/test_116_119.txt", test_transform)
    # test_dataset = MyDataset("data/train224_11_16.txt", test_transform)
    # test_dataset = MyDataset("data/txt/12_23_1_test224.txt", test_transform)
    # test_dataset = MyDataset("data/txt/12_23_2_test224.txt", test_transform)
    # test_dataset = MyDataset("data/txt/12_23_12_test224.txt", test_transform)
    # test_dataset = MyDataset("data/txt/12_23_12_addpre_test224.txt", test_transform)
    # test_dataset = MyDataset("data/txt/12_23_12_addpre_test_crop224.txt", test_transform)
    # test_dataset = MyDataset("data/txt_raw/total_test.txt", test_transform)

    # class6
    # test_dataset = MyDataset("data/txt6/12_23_12_addpre_test224_6.txt", test_transform)
    # test_dataset = MyDataset("data/txt6/12_23_12_addpre_test224_addcrop_6.txt", test_transform)

    # class7
    # test_dataset = MyDataset("data/txt7/12_23_12_addpre_test224_7.txt", test_transform)
    # test_dataset = MyDataset("data/txt7/12_23_12_addpre_test224_addcrop_7.txt", test_transform)

    # class8
    # all class8
    # test_dataset = MyDataset("data/txt_raw/total_test_8.txt", test_transform)
    # test_dataset = MyDataset("data/txt_raw_crop/total_test_crop_8.txt", test_transform)
    # test_dataset = MyDataset("data/txt_raw_crop/total_test_crop.txt", test_transform)
    test_dataset = MyDataset("data/txt_raw/total_test.txt", test_transform)
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
        labels = ["正常",  "喝水", "吸烟", "操作中控", "玩手机", "接电话"]
    elif num_classes == 7:
        labels = ["正常",  "喝水", "吸烟", "操作中控", "玩手机", "接电话", "其他"]
    else:
        labels = ["正常", "侧视", "喝水", "吸烟", "操作中控", "玩手机", "侧身拿东西", "接电话"]
    print(labels)
    print("row:target   col:predict")
    print(cm)
    true_label = np.zeros((num_classes, ), dtype=np.int)
    predicted_label = np.zeros((num_classes,), dtype=np.int)
    total = 0
    for i in range(num_classes):
        for j in range(num_classes):
            true_label[i] += cm[i][j]
            predicted_label[i] += cm[j][i]
            total += cm[i][j]
    print("true label:", true_label)
    print("predicted label:", predicted_label)
    TP = np.zeros((num_classes, ), dtype=np.int)
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
        Accuracy[i] = (TP[i]+TN[i])/total
        Precision[i] = TP[i]/predicted_label[i]
        Recall[i] = TP[i]/true_label[i]
        F1[i] = Precision[i]*Recall[i]/(Precision[i]+Recall[i])*2
    print("TP:", TP)
    print("TN:", TN)
    print("FP:", FP)
    print("FN:", FN)
    print("Accuracy:", Accuracy)
    print("Precision:", Precision)
    print("Recall:", Recall)
    print("F1:", F1)
    dict = {}
    dict["准确率"] = Accuracy.tolist() # 样本被分类正确的概率, 包括TP和TF
    dict["精确率"] = Precision.tolist() # 样本识别正确的概率，
    dict["召回率"] = Recall.tolist() # 样本被正确识别出的概率，检出率
    dict["F1-score"] = F1.tolist()
    test_path = os.path.dirname(model_path)
    with open(os.path.join(test_path, "test.json"), "w", encoding='utf-8') as f:
        json.dump(dict, f)
    # 保存excel
    df = pd.DataFrame(dict, index=labels)
    df.to_excel(os.path.join(test_path, 'test.xlsx'))

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
        dnn_index = np.argmax(dnn_probs) #By default, the index is into the flattened array, otherwise along the specified axis.
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
    # 测试
    # test_efficientnet()
    # 训练
    # for epoch in range(start_epoch, start_epoch + 48):
    #     train(epoch)
    #     val(epoch)
    main()
    # net_test()
    # net_test_onnx()
    # 展示预测结果
    # visualize_pred()
