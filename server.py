import io
import json
import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from efficientnet_pytorch import EfficientNet

app = Flask(__name__)

# kaggle dataset
# num_classes = 10
# label_name = ["正常","右持手机","右接电话","左持手机","左接电话","操作仪器","喝水","向后侧身","整理仪容","侧视"]
# # 新的类别
# # # label_name = ["正常","右接电话",左接电话","低头","操作仪器","喝水","吸烟","向后侧身","整理仪容","侧视"]
# mydataset
classes_path = r"data/drive_classes.txt"
with open(classes_path) as f:
    label_name = [c.strip() for c in f.readlines()]
num_classes = len(label_name)

## model: efficientnet  dataset: kaggle
# model = EfficientNet.from_name('efficientnet-b0',num_classes=num_classes)
# # 加载模型参数
# path = r"checkpoint/B0/000/B0_acc=99.8528.pth"

## model: resnet18  dataset: mydataset
model = models.resnet18(pretrained=False,num_classes=num_classes)
# 加载模型参数
path = r"checkpoint/resnet18/000/B0_acc=84.8921.pth"
checkpoint = torch.load(path)
model.load_state_dict(checkpoint["net"])
print("loaded model with acc:{}".format(checkpoint["acc"]))
model.eval()

def transform_image(image_bytes):
    # my_transforms = transforms.Compose([transforms.Resize(255),
    #                                     transforms.CenterCrop(224),
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize(
    #                                         [0.485, 0.456, 0.406],
    #                                         [0.229, 0.224, 0.225])])
    my_transforms = transforms.Compose([

        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    # _, y_hat = outputs.max(1) # max_value,max_index
    # predicted_idx = str(y_hat.item())
    # return imagenet_class_index[predicted_idx]
    pred_index = int(torch.argmax(outputs, 1).cpu().detach().numpy())
    return pred_index,label_name[pred_index]

@app.route('/')
def hello():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})

if __name__ == '__main__':
    app.run()