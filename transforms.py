from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import math

# img_path = r"D:\datasets\11_16\dataset\train224\8\20201116_155514_tp00008_4018.jpg "
img_path = r"C:\Users\win10\Desktop\开题\大论文\图片\数据集示例\0_crop.png "

img = Image.open(img_path)
# img.show()
# print(img)
# img = img.resize((256, 256), Image.BILINEAR)
# img = np.array(img)
# print(img.shape)
transform = transforms.Compose([
    # transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.), ),
    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomRotation(10, resample=False, expand=False, center=None),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.Resize((224, 224)),
    # transforms.RandomCrop(224, padding=16),

])
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    transforms.RandomRotation(10, resample=False, expand=False, center=None),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
# trans_img = transform(img).numpy()
# trans_img = transform(img)
# trans_img.show()
# print(type(trans_img))
# trans_img = trans_img.transpose(1, 2, 0)
# plt.imshow(img)
# plt.imshow(trans_img)
# plt.axis('off')
# plt.show()

# fig = plt.figure(figsize=(16, 9))
# sub1 = fig.add_subplot(2, 3, 1)
# trans_img = transform(img)
# sub1.imshow(trans_img)
# plt.axis('off')
# sub2 = fig.add_subplot(2, 3, 2)
# trans_img = transform(img)
# sub2.imshow(trans_img)
# plt.axis('off')
# sub3 = fig.add_subplot(2, 3, 3)
# trans_img = transform(img)
# sub3.imshow(trans_img)
# plt.axis('off')
# sub4 = fig.add_subplot(2, 3, 4)
# trans_img = transform(img)
# sub4.imshow(trans_img)
# plt.axis('off')
# sub5 = fig.add_subplot(2, 3, 5)
# trans_img = transform(img)
# sub5.imshow(trans_img)
# plt.axis('off')
# sub6 = fig.add_subplot(2, 3, 6)
# trans_img = transform(img)
# sub6.imshow(trans_img)
# plt.axis('off')
# plt.show()

fig = plt.figure(figsize=(16, 9))
sub1 = fig.add_subplot(1, 3, 1)
sub1.imshow(img)
plt.axis('off')
sub1.set_title("(a)", y=-0.15)
sub2 = fig.add_subplot(1, 3, 2)
trans_img = transform(img)
sub2.imshow(trans_img)
plt.axis('off')
sub2.set_title("(b)", y=-0.15)
sub3 = fig.add_subplot(1, 3, 3)
trans_img = transform(img)
sub3.imshow(trans_img)
plt.axis('off')
sub3.set_title("(c)", y=-0.15)
plt.show()