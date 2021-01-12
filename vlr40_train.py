from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import math
from utils import progress_bar, format_time

from PIL import Image
import onnxruntime
import cv2

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


train_transform = transforms.Compose([

    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    # transforms.RandomRotation(20, resample=False, expand=False, center=None),
    transforms.RandomRotation(10, resample=False, expand=False, center=None),
    # transforms.RandomHorizontalFlip(p=0.5),
    # # transforms.RandomVerticalFlip(p=0.5),
    # # transforms.RandomResizedCrop((224,224)),
    # # transforms.Resize((224, 224)),
    transforms.RandomCrop(224, padding=16),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

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
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

num_classes = 40
# net = models.mobilenet_v2(pretrained=False, num_classes=num_classes, width_mult=1.0)
net = models.mobilenet_v2(pretrained=True, width_mult=1.0)
num_in = net.classifier[1].in_features
net.classifier[1] = nn.Linear(num_in, num_classes)


# net = models.shufflenet_v2_x1_0(pretrained=True)
# # net = models.shufflenet_v2_x0_5(pretrained=True)
# # net = models.resnet50(pretrained=True)w
# num_in = net.fc.in_features
# net.fc = nn.Linear(num_in, num_classes)


# # 加载模型权重，忽略不同
# model_path = r"checkpoint/imagenet/imagenet100/mobilenetv2/111/mobilenetv2_1_imagenet_acc=68.9234.pth"
# model_dict =net.state_dict()
# checkpoint = torch.load(model_path, map_location=device)
# pretrained_dict = checkpoint["net"]
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
# model_dict.update(pretrained_dict)
# net.load_state_dict(model_dict)
# print("loaded model with acc:{}".format(checkpoint["acc"]))


def change_lr(epoch, T=6, factor=0.3, min=1e-3):
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

criterion = nn.CrossEntropyLoss()
epoches = 48


optimizer = optim.Adam(net.parameters(), lr=1e-3)
# optimizer = optim.SGD(net.parameters(), lr=1e-1,
#                       momentum=0.9, weight_decay=5e-4)

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_lr)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 12], gamma=0.1)
net.to(device)


train_dataset = MyDataset("VLR40/txt/train.txt", val_transform)
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=32,
                              shuffle=True,
                              num_workers=0)

val_dataset = MyDataset("VLR40/txt/test.txt", val_transform)
val_dataloader = DataLoader(dataset=val_dataset,
                              batch_size=32,
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

# Save checkpoint.

# VLR40
savepath = 'VLR40/checkpoint/mobilenetv2/pre/000/' #  randcrop 16 rotation 10 colorjit 0.5


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
            os.makedirs(savepath)
        print("best_acc:{:.4f}".format(acc))
        torch.save(state, savepath + 'mobilenetv2_VLR_acc={:.4f}.pth'.format(acc))
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
    net = models.mobilenet_v2(pretrained=False, num_classes=num_classes, width_mult=1.0)
    model_path = r"checkpoint/data_11_16/mobilenetv2/pre/0/111/mobilenetv2_1_my_acc=93.4426.pth"
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

    test_dataset = MyDataset("VLR40/txt/test.txt", test_transform)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=64,
                                 shuffle=True,
                                 num_workers=0)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
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
    print(average_loss, test_acc)



if __name__ == '__main__':
    # 训练
    main()
    # 测试
    # net_test()

