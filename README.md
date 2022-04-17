# Driving-Behavior-Recognition
训练驾驶行为识别模型
# 环境
python 3.7  
torch 1.7  
torchvision 0.8  
onnxruntime 1.7  
opencv-python 4.5
# 介绍
该项目包含训练驾驶行为识别模型的所有相关处理代码。包括数据集处理、模型训练、模型转换、模型测试等
# 公开数据集
[Driver Behavior Dataset](https://www.kaggle.com/datasets/robinreni/revitsone-5class)  
[AUC Distracted Driver Dataset](https://abouelnaga.io/projects/auc-distracted-driver-dataset/)  
使用ceateTxt.py创建数据集标签文件 
# 自定义数据集
使用convertVideoToImages.py将采集视频转化为图片  
使用打标签工具[actionlabel](https://github.com/QFaceblue/actionlabel)进行数据标注  
使用ceateTxt.py创建数据集标签文件 
# 模型训练
train.py是模型训练代码，使用时注意修改相关参数  
vlr40_train.py是精简版训练代码
# 模型转换
torchToOnnx.py将pth格式权重转化为通用onnx格式权重
# 模型测试
predict.py使用模型进行检测