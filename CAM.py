import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from PIL import Image
import cv2
import torch.nn.functional as F
from mobilenetv2_cam import MobileNetV2_cam
train_transform = transforms.Compose([

    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    transforms.RandomRotation(10, resample=False, expand=False, center=None),
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

def softmax_np(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max # 当输入一个较大的数值时，sofmax函数将会超出限制,利用softmax(x) = softmax(x+c), 取c = - max(x)
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax

def returnCAM(feature_conv, linear_weight, class_idx, size_upsample = (224, 224)):
    # generate the class activation maps upsample to 256x256
    bz, nc, h, w = feature_conv.shape
    # weight_softmax = linear_weight
    # print(linear_weight)
    weight_softmax = F.softmax(linear_weight, 1)
    # print(weight_softmax[class_idx].shape)
    weight_softmax = weight_softmax[class_idx].reshape(-1, nc)
    # print(weight_softmax)
    # print(feature_conv.reshape((nc, h * w)).shape)
    cam = np.matmul(weight_softmax, feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - cam.min()
    cam_img = cam / cam.max()
    cam_img = np.uint8(255 * cam_img)
    print(cam_img)
    # cam_img = cv2.resize(cam_img, size_upsample)
    cam_img = cv2.resize(cam_img, size_upsample, interpolation=cv2.INTER_LINEAR)

    # print(cam_img)
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    # cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_HOT)
    cam_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)
    return cam_img


device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = 9
net = MobileNetV2_cam(num_classes=num_classes, width_mult=1.0)

# 加载模型权重，忽略不同
model_path = r"checkpoint/data_12_23/mobilenetv2/666/mobilenetv2_1_12_23_acc=91.4710.pth"
model_dict =net.state_dict()
checkpoint = torch.load(model_path, map_location=device)
pretrained_dict = checkpoint["net"]
pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
model_dict.update(pretrained_dict)
net.load_state_dict(model_dict)
print("loaded model with acc:{}".format(checkpoint["acc"]))

net.to(device)
net.eval()

# D:\drivedata\train224\0\15-35-35_2a-6_0302.jpg 0
# D:\datasets\12_23_2\dataset\test224\p1\1\p1_0070.jpg 1
# D:\datasets\12_23_2\dataset\test224\p4\2\p4_0953.jpg 2
# D:\datasets\12_23_2\dataset\train224\p9\3\p9_0199.jpg 3
# D:\datasets\12_23_2\dataset\test224\p4\4\p4_1085.jpg 4
# D:\datasets\12_23_2\dataset\train224\p9\5\p9_0102.jpg 5
# D:\datasets\12_23_2\dataset\train224\p9\6\p9_0571.jpg 6
# D:\datasets\12_23_2\dataset\train224\p9\7\p9_0675.jpg 7
# D:\datasets\12_23_2\dataset\test224\p4\8\p4_0284.jpg 8

image_path = r"D:\datasets\12_23_2\dataset\train224\p9\7\p9_0675.jpg"
label = 0
raw_image = Image.open(image_path).convert('RGB')  #
# print(image.size)
# img = np.array(image)
# print(img.shape)
image = val_transform(raw_image)
print(image.shape)
input = torch.unsqueeze(image, 0)
print(input.shape)
input = input.to(device)

grad_list = []
def get_grad(grad):
    print(grad.shape)
    grad_list.append(grad)

net.classifier[1].weight.register_hook(get_grad)
output, features = net(input)
# one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
# one_hot[0][label] = 1
# one_hot = torch.from_numpy(one_hot).requires_grad_(True)
# # one_hot = torch.nn.Parameter(one_hot)
# # one_hot.cuda()
one_hot = torch.zeros((1, output.size()[-1]), device=device, requires_grad=False)
one_hot[0][label] = 1

out = torch.sum(one_hot * output)
net.zero_grad()
out.backward(retain_graph=True)
print(grad_list[0].shape)
# # print(output.shape)
# print(output)
# _, predicted = output.max(1)
# print(predicted)
# softmax = F.softmax(output, 1)
# print(softmax.max())
# softmax = softmax_np(output.cpu().detach().numpy())
# print(softmax.max())
#
# print(net.classifier[1].weight.shape)
print(net)
# print(features.shape)
# print(net.classifier[1].weight.shape)


# cam_img = returnCAM(features.cpu().detach(), net.classifier[1].weight.cpu().detach(), label)
cam_img = returnCAM(features.cpu().detach(), grad_list[0].cpu(), label)
raw_img = np.array(raw_image)
print(cam_img.shape, raw_img.shape)
result = np.uint8((cam_img *0.3+ raw_img*0.7))

fig = plt.figure(figsize=(16, 9))
sub1 = fig.add_subplot(1, 3, 1)
sub1.set_title("raw_image")
sub1.imshow(raw_img)
sub2 = fig.add_subplot(1, 3, 2)
sub2.set_title("cam_image")
sub2.imshow(cam_img)
sub3 = fig.add_subplot(1, 3, 3)
sub3.set_title("result")
sub3.imshow(result)
plt.show()