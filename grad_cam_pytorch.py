# https://github.com/jacobgil/pytorch-grad-cam
# https://zhuanlan.zhihu.com/p/269702192
# 对单个图像可视化
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from torchvision.models import resnet50, mobilenet_v2
from torchvision import transforms
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image
import torch
from mobilenetv2_cbam import MobileNetV2_cbam

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
num_classes = 9
# num_classes = 8
# 1.加载模型
# model = resnet50(pretrained=True)
# 2.选择目标层
# target_layers = [model.layer4[-1]]
# 3. 构建输入图像的Tensor形式
# image_path = './examples/simple/both.png'
# rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]   # 1是读取rgb
# rgb_img = np.float32(rgb_img) / 255
#
# # preprocess_image作用：归一化图像，并转成tensor
# input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
#                                              std=[0.229, 0.224, 0.225])   # torch.Size([1, 3, 224, 224])

model = mobilenet_v2(num_classes=num_classes, pretrained=False, width_mult=1.0)
# model = MobileNetV2_cbam(num_classes=num_classes, width_mult=1.0, add_location=(96, 160), ca=False, sa=True)
num_in = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_in, num_classes)
# print(model)
target_layers = [model.features[-1]]
# print(model.features[-1])

# 加载模型权重，忽略参数形状不同
# model_path = r"checkpoint/data_12_23/mobilenetv2/666/mobilenetv2_1_12_23_acc=91.4710.pth"
model_path = r"checkpoint/paper_test/ours/mobilenetv2/pre/000/mobilenetv2_acc=83.7205.pth"
model_path = r"checkpoint/paper_test/ours/mobilenetv2/pre/crop/000/mobilenetv2_crop_acc=87.8234.pth"
model_dict = model.state_dict()
checkpoint = torch.load(model_path, map_location="cpu")
pretrained_dict = checkpoint["net"]
pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                   k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v)}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
print("loaded model with acc:{}".format(checkpoint["acc"]))

# Create an input tensor image for your model..
# Note: input_tensor can be a batch tensor with several images!
# test dataset
# image_path = r"D:\datasets\3_23\cam_wen\dataset\test_crop\p23\8\270.jpg"
# label = 7
# image_path = r"D:\datasets\3_23\cam_wen\dataset\test_crop\p23\6\334.jpg"
# label = 6
# image_path = r"D:\datasets\3_23\cam_wen\dataset\test_crop\p23\5\238.jpg"
# label = 5
# image_path = r"D:\datasets\3_23\cam_wen\dataset\test_crop\p23\4\166.jpg"
# label = 4
# image_path = r"D:\datasets\3_23\cam_wen\dataset\test_crop\p23\3\135.jpg"
# label = 3
# image_path = r"D:\datasets\3_23\cam_wen\dataset\test_crop\p23\2\109.jpg"
# label = 2
# image_path = r"D:\datasets\3_23\cam_wen\dataset\test_crop\p23\1\19.jpg"
# label = 1
# image_path = r"D:\datasets\3_23\cam_wen\dataset\test_crop\p23\0\3.jpg"
# label = 0
# train dataset
# image_path = r"D:\datasets\3_23\cam_he\dataset\train_crop\p33\8\263.jpg"
# label = 7
# image_path = r"D:\datasets\3_23\cam_wen\dataset\train_crop\p9\6\255.jpg"
# label = 6
# image_path = r"D:\datasets\3_23\cam_wen\dataset\train_crop\p9\5\191.jpg"
# label = 5
# image_path = r"D:\datasets\3_23\cam_wen\dataset\train_crop\p9\4\161.jpg"
# label = 4
# image_path = r"D:\datasets\3_23\cam_wen\dataset\train_crop\p9\3\130.jpg"
# label = 3
# image_path = r"D:\datasets\3_23\cam_wen\dataset\train_crop\p9\2\98.jpg"
# label = 2
# image_path = r"D:\datasets\3_23\cam_wen\dataset\train_crop\p9\1\32.jpg"
# label = 1
# image_path = r"D:\datasets\3_23\cam_wen\dataset\train_crop\p9\0\71.jpg"
# label = 0
image_path = r"D:\车联网\3_25数据集示例\0.jpg"
label = 0
image_path1 = r"D:\车联网\3_25数据集示例\1.jpg"
label1 = 1
image_path2 = r"D:\车联网\3_25数据集示例\2.jpg"
label2 = 2
image_path3 = r"D:\车联网\3_25数据集示例\3.jpg"
label3 = 3

image_path4 = r"C:\Users\win10\Desktop\开题\大论文\图片\数据集示例\0_crop.png "
label4 = 0
def getResult(imagePath, lbl):
    raw_image = Image.open(imagePath).convert('RGB').resize((224, 224))  #
    rgb_img = np.float32(raw_image) / 255
    # print(image.size)
    # img = np.array(image)
    # print(img.shape)
    image = val_transform(raw_image)
    # print("input size:", image.shape)
    input = torch.unsqueeze(image, 0)
    # print(input.shape)
    input_tensor = input.to('cpu')
    # Construct the CAM object once, and then re-use it on many images:
    # 4.初始化GradCAM，包括模型，目标层以及是否使用cuda
    gradCAM = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    # If target_category is None, the highest scoring category
    # will be used for every image in the batch.
    # target_category can also be an integer, or a list of different integers
    # for every image in the batch.
    # 5.选定目标类别，如果不设置，则默认为分数最高的那一类
    # target_category = None # 281
    target_category = lbl
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    # 6. 计算cam
    grayscale_cam = gradCAM(input_tensor=input_tensor, target_category=target_category)  # [batch, 224,224]
    grayscale_cam = grayscale_cam[0]
    visualization = show_cam_on_image(rgb_img, grayscale_cam)  # (224, 224, 3)
    result = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
    return result


raw_image = Image.open(image_path).convert('RGB').resize((224, 224))  #
rgb_img = np.float32(raw_image) / 255
# print(image.size)
# img = np.array(image)
# print(img.shape)
image = val_transform(raw_image)
print("input size:", image.shape)
input = torch.unsqueeze(image, 0)
# print(input.shape)
input_tensor = input.to('cpu')

# Construct the CAM object once, and then re-use it on many images:
# 4.初始化GradCAM，包括模型，目标层以及是否使用cuda
gradCAM = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
gradCAMPP = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=False)
xgradCAM = XGradCAM(model=model, target_layers=target_layers, use_cuda=False)
eigenCAM = EigenCAM(model=model, target_layers=target_layers, use_cuda=False)
# If target_category is None, the highest scoring category
# will be used for every image in the batch.
# target_category can also be an integer, or a list of different integers
# for every image in the batch.
# 5.选定目标类别，如果不设置，则默认为分数最高的那一类
# target_category = None # 281
target_category = label
# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
# 6. 计算cam
grayscale_cam = gradCAM(input_tensor=input_tensor, target_category=target_category)  # [batch, 224,224]
# grayscale_cam1 = gradCAMPP(input_tensor=input_tensor, target_category=target_category)
# grayscale_cam2 = xgradCAM(input_tensor=input_tensor, target_category=target_category)
# grayscale_cam3 = eigenCAM(input_tensor=input_tensor, target_category=target_category)
# In this example grayscale_cam has only one image in the batch:
# 7.展示热力图并保存, grayscale_cam是一个batch的结果，只能选择一张进行展示
grayscale_cam = grayscale_cam[0]
visualization = show_cam_on_image(rgb_img, grayscale_cam)  # (224, 224, 3)
result = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
# grayscale_cam1 = grayscale_cam1[0]
# visualization1 = show_cam_on_image(rgb_img, grayscale_cam1)  # (224, 224, 3)
# result1 = cv2.cvtColor(visualization1, cv2.COLOR_BGR2RGB)
# grayscale_cam2 = grayscale_cam2[0]
# visualization2 = show_cam_on_image(rgb_img, grayscale_cam2)  # (224, 224, 3)
# result2 = cv2.cvtColor(visualization2, cv2.COLOR_BGR2RGB)
# grayscale_cam3 = grayscale_cam3[0]
# visualization3 = show_cam_on_image(rgb_img, grayscale_cam3)  # (224, 224, 3)
# result3 = cv2.cvtColor(visualization3, cv2.COLOR_BGR2RGB)
# heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
# heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
# heatmap = np.float32(heatmap) / 255

# fig = plt.figure(figsize=(16, 9))
# sub1 = fig.add_subplot(2, 3, 1)
# sub1.set_title("raw_image")
# sub1.imshow(rgb_img)
# sub2 = fig.add_subplot(2, 3, 2)
# sub2.set_title("gradcam")
# sub2.imshow(result)
# sub3 = fig.add_subplot(2,3, 3)
# sub3.set_title("gradcam++")
# sub3.imshow(result1)
# sub3 = fig.add_subplot(2, 3, 5)
# sub3.set_title("xgradcam")
# sub3.imshow(result2)
# sub3 = fig.add_subplot(2, 3, 6)
# sub3.set_title("eigencam")
# sub3.imshow(result3)
# plt.show()

# fig = plt.figure()
# sub1 = fig.add_subplot(1, 2, 1)
# sub1.set_title("(a)", y=-0.1)
# sub1.imshow(rgb_img)
# plt.axis('off')
# sub2 = fig.add_subplot(1, 2, 2)
# sub2.set_title("(b)", y=-0.1)
# sub2.imshow(result)
# plt.axis('off')
# plt.show()
# cv2.imwrite('./examples/simple/cam_dog.jpg', visualization)

# result = getResult(image_path, label)
# result1 = getResult(image_path1, label1)
# result2 = getResult(image_path2, label2)
# result3 = getResult(image_path3, label3)
#
# fig = plt.figure()
# sub1 = fig.add_subplot(2, 2, 1)
# sub1.set_title("(a)", y=-0.15)
# sub1.imshow(result)
# plt.axis('off')
# sub2 = fig.add_subplot(2, 2, 2)
# sub2.set_title("(b)", y=-0.15)
# sub2.imshow(result1)
# plt.axis('off')
# sub2 = fig.add_subplot(2, 2, 3)
# sub2.set_title("(c)", y=-0.15)
# sub2.imshow(result2)
# plt.axis('off')
# sub2 = fig.add_subplot(2, 2, 4)
# sub2.set_title("(d)", y=-0.15)
# sub2.imshow(result3)
# plt.axis('off')
# plt.show()

result4 = getResult(image_path4, label4)
raw4 = Image.open(image_path4).convert('RGB').resize((224, 224))
fig = plt.figure()
sub1 = fig.add_subplot(1, 2, 1)
sub1.set_title("(a)", y=-0.15)
sub1.imshow(raw4)
plt.axis('off')
sub2 = fig.add_subplot(1, 2, 2)
sub2.set_title("(b)", y=-0.15)
sub2.imshow(result4)
plt.axis('off')
plt.show()