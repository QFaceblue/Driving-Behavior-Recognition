from __future__ import print_function, division

import os
import time

import cv2
import numpy as np
import onnxruntime
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from mnext import mnext
from utils import progress_bar


def softmax_np(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax


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
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(10, resample=False, expand=False, center=None),
    # transforms.RandomHorizontalFlip(p=0.5),
    # # transforms.RandomVerticalFlip(p=0.5),
    # # ToTensor()能够把灰度范围从0-255变换到0-1之间，
    # # transform.Normalize()则把0-1变换到(-1,1).具体地说，对每个通道而言，Normalize执行以下操作：
    # # image=(image-mean)/std
    # # transforms.RandomResizedCrop((224,224)),
    # # transforms.Resize((224, 224)),
    transforms.RandomCrop(224, padding=16),
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



def net_test():
    # net = models.mobilenet_v2(pretrained=False, num_classes=num_classes, width_mult=1.0)
    # # model_path = r"checkpoint/data_11_16/mobilenetv2/pre/555/mobilenetv2_1_my_acc=96.1749.pth" # crop 160=0.7486338797814208
    # # model_path = r"checkpoint/data_11_16/mobilenetv2/pre/222/mobilenetv2_1_my_acc=92.3497.pth"  # 160=0.5846994535519126
    # # model_path = r"checkpoint/data_11_16/mobilenetv2/pre/666/mobilenetv2_1_my_acc=95.6284.pth"  # 160=0.9562841530054644 224=0.7486338797814208
    # # model_path = r"checkpoint/data_11_16/mobilenetv2/pre/777/mobilenetv2_1_my_acc=95.6284.pth"  # 160=0.8142076502732241
    # # model_path = r"checkpoint/data_11_16/mobilenetv2/pre/0/111/mobilenetv2_1_my_acc=93.4426.pth"  # crop=0.9398907103825137 crop_160=0.907103825136612
    #
    # # model_path = r"checkpoint/data_12_23/mobilenetv2/222/mobilenetv2_1_12_23_acc=93.6898.pth"
    # # model_path = r"checkpoint/data_12_23/mobilenetv2/333/mobilenetv2_1_12_23_acc=89.9061.pth"
    # model_path = r"checkpoint/data_12_23/mobilenetv2/888/mobilenetv2_1_12_23_acc=91.6275.pth"

    net = mnext(num_classes=num_classes, width_mult=1.)
    model_path = r"checkpoint/data_12_23/mnext/000/mnext_1_12_23_acc=92.1753.pth"
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
    test_dataset = MyDataset("data/txt/12_23_12_test224.txt", test_transform)
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
    print(cm)

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

    onnx_session = onnxruntime.InferenceSession(path, None)
    # dnn_net = cv2.dnn.readNetFromONNX(path)
    xml_path = r"checkpoint/data_12_23/mobilenetv2/888/mobilenetv2_1_12_23_acc=91.6275.xml"
    bin_path = r"checkpoint/data_12_23/mobilenetv2/888/mobilenetv2_1_12_23_acc=91.6275.bin"
    # # FP16
    # xml_path = r"checkpoint/data_12_23/mobilenetv2/8888/mobilenetv2_1_12_23_acc=91.6275.xml"
    # bin_path = r"checkpoint/data_12_23/mobilenetv2/8888/mobilenetv2_1_12_23_acc=91.6275.bin"
    dnn_net = cv2.dnn.readNet(xml_path, bin_path)
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

    net_test()

