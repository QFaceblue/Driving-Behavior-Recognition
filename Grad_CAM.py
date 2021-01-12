import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from PIL import Image
import cv2
import torch.nn.functional as F
from mobilenetv2_cam import MobileNetV2_grad_cam
from mobilenetv2_cbam import MobileNetV2_cbam
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

def returnGrad_CAM(feature_conv, grad, size_upsample = (224, 224)):
    # generate the class activation maps upsample to 256x256
    bz, nc, h, w = feature_conv.shape
    out, inc, gh, gw = grad.shape
    c_w = torch.mean(grad, (2, 3))
    # print(c_w.shape)
    c_w = c_w.sum(0)
    # print(c_w.shape)

    weight_softmax = F.softmax(c_w, 0)
    if weight_softmax.shape != torch.Size([1]):
        weight_softmax = weight_softmax.reshape(-1, nc)
        # print(weight_softmax)
        cam = np.matmul(weight_softmax, feature_conv.reshape((nc, h * w)))
    else:
        cam = feature_conv.sum(1)
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
net = MobileNetV2_grad_cam(num_classes=num_classes, width_mult=1.0)
# net = MobileNetV2_cbam(num_classes=num_classes, width_mult=1.0, add_location=16)
# 加载模型权重，忽略参数形状不同
model_path = r"checkpoint/data_12_23/mobilenetv2/666/mobilenetv2_1_12_23_acc=91.4710.pth"
# model_path = r"checkpoint/data_12_23/mobilenetv2/nopre/1/111/mobilenetv2_1_12_23_acc=90.9091.pth"
model_dict =net.state_dict()
checkpoint = torch.load(model_path, map_location=device)
pretrained_dict = checkpoint["net"]
pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
model_dict.update(pretrained_dict)
net.load_state_dict(model_dict)
print("loaded model with acc:{}".format(checkpoint["acc"]))

# # 加载模型权重，忽略不同
# model_path = r"checkpoint/data_12_23/mobilenetv2/666/mobilenetv2_1_12_23_acc=91.4710.pth"
# model_dict =net.state_dict()
# checkpoint = torch.load(model_path, map_location=device)
# pretrained_dict = checkpoint["net"]
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v)}
# model_dict.update(pretrained_dict)
# net.load_state_dict(model_dict)
# print("loaded model with acc:{}".format(checkpoint["acc"]))

# # 更新权重 mobilenetv2->mobilenetv2_cbam_16
# model_path = r"checkpoint/data_12_23/mobilenetv2/666/mobilenetv2_1_12_23_acc=91.4710.pth"
# model_dict =net.state_dict()
# checkpoint = torch.load(model_path, map_location=device)
# pretrained_dict = checkpoint["net"]
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
# print("loaded model with acc:{}".format(checkpoint["acc"]))

net.to(device)
net.eval()
# print(net)
# D:\drivedata\train224\0\15-35-35_2a-6_0302.jpg 0
# D:\datasets\12_23_2\dataset\test224\p1\1\p1_0070.jpg 1
# D:\datasets\12_23_2\dataset\test224\p4\2\p4_0953.jpg 2
# D:\datasets\12_23_2\dataset\train224\p9\3\p9_0199.jpg 3
# D:\datasets\12_23_2\dataset\test224\p4\4\p4_1085.jpg 4
# D:\datasets\12_23_2\dataset\train224\p9\5\p9_0102.jpg 5
# D:\datasets\12_23_2\dataset\train224\p9\6\p9_0571.jpg 6
# D:\datasets\12_23_2\dataset\train224\p9\7\p9_0675.jpg 7
# D:\datasets\12_23_2\dataset\test224\p4\8\p4_0284.jpg 8

image_path = r"D:\datasets\12_23_2\dataset\test224\p4\8\p4_0284.jpg"
label = 8
raw_image = Image.open(image_path).convert('RGB')  #
# print(image.size)
# img = np.array(image)
# print(img.shape)
image = val_transform(raw_image)
print("input size:", image.shape)
input = torch.unsqueeze(image, 0)
# print(input.shape)
input = input.to(device)

input_list = []
output_list = []
grad_list = []

# 展示中间特征
def show(input,col = 8,max=None):
    imgs = input[0]
    nc, h, w = imgs.shape
    if max is not None and nc > max:
        nc = max
    row = nc//col + 1
    plt.figure(figsize=(16, 9))
    for i in range(nc):
        plt.subplot(row, col, i+1)
        plt.imshow(imgs[i].cpu().detach().numpy())
        plt.axis('off')
    plt.show()

# 参考 https://blog.csdn.net/dreaming_coder/article/details/104522332
# 注意不同hook函数的使用,以下三个对module使用
def forward_hook(module, data_input, data_output):
    print("forward hook input:{}".format(data_input[0].shape))
    print("forward hook output:{}".format(data_output.shape))
    input_list.append(data_input[0])
    output_list.append(data_output)

def forward_pre_hook(module, data_input):
    print("forward_pre_hook input:{}".format(data_input))


def backward_hook(module, grad_input, grad_output):
    print("backward hook input:{}".format(grad_input))
    print("backward hook output:{}".format(grad_output))
    # 前两个梯度可能是batchnorm参数的梯度
    grad_list.append(grad_input)
    # grad_list.append(grad_output)
# 对tensor使用
def grad_hook(grad):
    print("grad hook :{}".format(grad.shape))
    # weight shape = out, in, h, w
    grad_list.append(grad)

# 给分类层加hook
# net.classifier[1].weight.register_hook(grad_hook)
# net.classifier[1].register_backward_hook(backward_hook)
# net.classifier[1].register_forward_hook(forward_hook)

# 给特征提取层加hook
# 2:torch.Size([1, 16, 112, 112])
# 4:torch.Size([1, 24, 56, 56])
# 7:torch.Size([1, 32, 28, 28])
# 11:torch.Size([1, 64, 14, 14])
# 17:torch.Size([1, 160, 7, 7])
layer_num = 17
# 注意力机制增加两层
# layer_num = layer_num + 2
# CA
# net.features[2].fc2.register_forward_hook(forward_hook)
# # SA
# net.features[3].conv1.register_forward_hook(forward_hook)

net.features[layer_num].conv[0][0].register_forward_hook(forward_hook)
# deepwise conv kernel torch.Size([32, 1, 3, 3]) out chanel number = 1
net.features[layer_num].conv[0][0].weight.register_hook(grad_hook)
output, features = net(input)
# output = net(input)

# print(len(input_list))
# print(output_list[0])
# 展示中间特征
# show(input_list[0])
# show(input_list[0], max=32)
one_hot = torch.zeros((1, output.size()[-1]), device=device, requires_grad=False)
one_hot[0][label] = 1
out = torch.sum(one_hot * output)
net.zero_grad()
out.backward(retain_graph=True)

# 前两个梯度可能是batchnorm参数的梯度
# print(grad_list[0][2].permute(1, 0).shape)
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
# print(net)
# print(features.shape)
# print(net.classifier[1].weight.shape)


# # cam_img = returnCAM(features.cpu().detach(), net.classifier[1].weight.cpu().detach(), label)
# cam_img = returnCAM(features.cpu().detach(), grad_list[0][2].permute(1,0).cpu(), label)
cam_img = returnGrad_CAM(input_list[0].cpu().detach(), grad_list[0].cpu().detach())
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