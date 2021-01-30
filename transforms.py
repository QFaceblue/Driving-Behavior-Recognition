from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import math

img_path = r"D:\datasets\11_16\dataset\train224\8\20201116_155514_tp00008_4018.jpg "

img = Image.open(img_path)
# img.show()
# print(img)
# img = img.resize((256, 256), Image.BILINEAR)
# img = np.array(img)
# print(img.shape)
transform = transforms.Compose([
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.), ),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomRotation(10, resample=False, expand=False, center=None),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.Resize((224, 224)),
    # transforms.RandomCrop(224, padding=16),

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
fig = plt.figure(figsize=(16, 9))
sub1 = fig.add_subplot(2, 3, 1)
trans_img = transform(img)
sub1.imshow(trans_img)
plt.axis('off')
sub2 = fig.add_subplot(2, 3, 2)
trans_img = transform(img)
sub2.imshow(trans_img)
plt.axis('off')
sub3 = fig.add_subplot(2, 3, 3)
trans_img = transform(img)
sub3.imshow(trans_img)
plt.axis('off')
sub4 = fig.add_subplot(2, 3, 4)
trans_img = transform(img)
sub4.imshow(trans_img)
plt.axis('off')
sub5 = fig.add_subplot(2, 3, 5)
trans_img = transform(img)
sub5.imshow(trans_img)
plt.axis('off')
sub6 = fig.add_subplot(2, 3, 6)
trans_img = transform(img)
sub6.imshow(trans_img)
plt.axis('off')
plt.show()