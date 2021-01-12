import torch
from efficientnet_pytorch import EfficientNet
# import onnx # 环境问题
from torchvision import datasets, models, transforms
import json
from PIL import Image
import cv2
import numpy as np
from timeit import default_timer as timer
import time
import onnxruntime
from ghost_net import ghost_net

# mydataset
classes_path = r"data/drive_classes.txt"
with open(classes_path) as f:
    label_name = [c.strip() for c in f.readlines()]
num_classes = len(label_name)

# # efficientnet b0
# model = EfficientNet.from_name('efficientnet-b0',num_classes=num_classes)
# path = r"checkpoint/B0/444/B0_acc=84.8921.pth"
## model: resnet18  dataset: mydataset
# model = models.resnet18(pretrained=False,num_classes=num_classes)
# # 加载模型参数
# path = r"checkpoint/resnet18/000/B0_acc=84.8921.pth"
#
# # # mobilenetv2 dataset mydataset
# # model = models.mobilenet_v2(pretrained=False,num_classes=num_classes)
# # # 加载模型参数
# # path = r"checkpoint/mobilenetv2/000/mv2_acc=82.7338.pth"

path = r"checkpoint/ghost_net/333/ghostnet_05_kg_acc=68.3453.pth"
model = ghost_net(num_classes=num_classes, width_mult=0.5)
checkpoint = torch.load(path)
model.load_state_dict(checkpoint["net"])
print("loaded model with acc:{}".format(checkpoint["acc"]))
model.eval()
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device ='cpu'
model.to(device)

# onnx
# path = r"checkpoint/resnet18/000/B0_acc=84.8921_2.onnx"
path = r"checkpoint/resnet18/111/resnet18_kg_acc=99.3310.onnx"
path = r"checkpoint/data_11_16/mobilenetv2/pre/0/111/mobilenetv2_1_my_224.onnx"
net = cv2.dnn.readNetFromONNX(path)
# xml_path = r"checkpoint/resnet18/000/B0_acc=84.8921.xml"
# bin_path = r"checkpoint/resnet18/000/B0_acc=84.8921.bin"
# xml_path = r"checkpoint/resnet18/111/resnet18_kg_acc=99.3310.xml"
# bin_path = r"checkpoint/resnet18/111/resnet18_kg_acc=99.3310.bin"
xml_path = r"checkpoint/data_11_16/mobilenetv2/pre/0/111/mobilenetv2_1_my_224.xml"
bin_path = r"checkpoint/data_11_16/mobilenetv2/pre/0/111/mobilenetv2_1_my_224.bin"
# net = cv2.dnn.readNet(xml_path, bin_path)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
# net = cv2.dnn.readNetFromModelOptimizer(xml_path,bin_path)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
net = cv2.dnn.readNetFromModelOptimizer(xml_path,bin_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
onnx_session = onnxruntime.InferenceSession(path,None)
# # 使用openvino后端
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
# # openvino
# xml_path = r"checkpoint/resnet18/000/B0_acc=84.8921.xml"
# bin_path = r"checkpoint/resnet18/000/B0_acc=84.8921.bin"
# net_openvino = cv2.dnn.readNetFromModelOptimizer(xml_path,bin_path)
# # 使用openvino后端
# net_openvino.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
# net_openvino.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

data_transform = transforms.Compose([

    # transforms.RandomResizedCrop((500,500)),
    # transforms.RandomHorizontalFlip(),
    # ToTensor()能够把灰度范围从0-255变换到0-1之间，
    # transform.Normalize()则把0-1变换到(-1,1).具体地说，对每个通道而言，Normalize执行以下操作：
    # image=(image-mean)/std
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def detect_image_path(img_path,draw=True):
    src = cv2.imread(img_path)
    image = cv2.resize(src, (224, 224))
    image = np.float32(image) / 255.0
    image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
    image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image).unsqueeze(0).to(device)
    outputs = model.forward(image)
    pred_index = int(torch.argmax(outputs, 1).cpu().detach().numpy())
    if draw:
        cv2.putText(src, label_name[pred_index], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow("input", src)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # _, y_hat = outputs.max(1) # max_value,max_index
    # predicted_idx = str(y_hat.item())
    # return imagenet_class_index[predicted_idx]
    return pred_index, label_name[pred_index]

def detect_image(image):
    image = data_transform(image).unsqueeze(0).to(device)
    outputs = model.forward(image)
    pred_index = int(torch.argmax(outputs, 1).cpu().detach().numpy())
    return pred_index, label_name[pred_index]


# total time:1175.555784702301
def detect_video(video_path,output_path=""):

    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(
            video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()

        if not return_value:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        pre_t = timer()
        # opencv读取的图片不管是视频帧还是图片都是矩阵形式，即np.array，转PIL.Image格式用PIL.Image.fromarray()函数即可。
        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame2)
        index,label_name = detect_image(image)
        # print('class_id: {}  class_name: {}'.format(index,label_name))
        result = np.asarray(image)
        curr_t = timer()
        infer_t = curr_t - pre_t
        print("inferrence time:{}".format(infer_t))
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        text ="label:{}   {}".format(label_name,fps)
        cv2.putText(frame, text=text, org=(150, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0, 0, 255), thickness=3)
        # cv2.putText(result, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", frame)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# total time:476.4733600616455
def detect_video_dnn(video_path,output_path=""):

    vid = cv2.VideoCapture(video_path)
    # print("buffersize:",vid.get(cv2.CAP_PROP_BUFFERSIZE))
    # # 设置缓冲区
    # vid.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    # print("buffersize:", vid.get(cv2.CAP_PROP_BUFFERSIZE))
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    print("FourCC:{}  fps:{}  size:{}  output:{}".format(video_FourCC,video_fps,video_size,isOutput))
    if isOutput:
        print("!!! TYPE:", type(output_path), type(
            video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()

        if not return_value:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(frame2, (224, 224))
        image = np.float32(image) / 255.0
        image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
        image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
        pre_t = timer()
        blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224), (0, 0, 0), False)
        net.setInput(blob)
        probs = net.forward()
        index = np.argmax(probs)
        index = 0
        curr_t = timer()
        infer_t = curr_t - pre_t
        print("inferrence time:{}".format(infer_t))
        # print('class_id: {}  class_name: {}'.format(index,label_name[index]))
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        text ="label:{}   {}".format(label_name[index],fps)
        # print(text)
        cv2.putText(frame, text=text, org=(150, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=2
                    , color=(0, 0, 255), thickness=3)
        # cv2.putText(result, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", frame)
        if isOutput:
            out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# total time:476.4733600616455
def detect_video_onnx(video_path,output_path=""):

    vid = cv2.VideoCapture(video_path)
    # print("buffersize:",vid.get(cv2.CAP_PROP_BUFFERSIZE))
    # # 设置缓冲区
    # vid.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    # print("buffersize:", vid.get(cv2.CAP_PROP_BUFFERSIZE))
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    print("FourCC:{}  fps:{}  size:{}  output:{}".format(video_FourCC,video_fps,video_size,isOutput))
    if isOutput:
        print("!!! TYPE:", type(output_path), type(
            video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()

        if not return_value:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(frame2, (224, 224))
        image = np.float32(image) / 255.0
        image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
        image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))

        image = image.reshape(1, 3, 224, 224)

        pre_t = timer()
        inputs = {onnx_session.get_inputs()[0].name: image}
        probs = onnx_session.run(None, inputs)
        index = np.argmax(probs)
        # index = 0
        curr_t = timer()
        infer_t = curr_t - pre_t
        print("inferrence time:{}".format(infer_t))
        # print('class_id: {}  class_name: {}'.format(index,label_name[index]))
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        text ="label:{}   {}".format(label_name[index],fps)
        # print(text)
        cv2.putText(frame, text=text, org=(150, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1
                    , color=(0, 0, 255), thickness=3)
        # cv2.putText(result, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        # cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", frame)
        if isOutput:
            out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 初始化openvino，必须在指定目录运行
# C:
# cd C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\bin
# setupvars.bat
# d:
# total time:414.77942538261414
# def detect_video_openvino(video_path,output_path=""):
#
#     vid = cv2.VideoCapture(video_path)
#     if not vid.isOpened():
#         raise IOError("Couldn't open webcam or video")
#     video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
#     video_fps = vid.get(cv2.CAP_PROP_FPS)
#     video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
#                   int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#     isOutput = True if output_path != "" else False
#     if isOutput:
#         print("!!! TYPE:", type(output_path), type(
#             video_FourCC), type(video_fps), type(video_size))
#         out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
#     accum_time = 0
#     curr_fps = 0
#     fps = "FPS: ??"
#     prev_time = timer()
#     while True:
#         return_value, frame = vid.read()
#         if not return_value:
#             print("Can't receive frame (stream end?). Exiting ...")
#             break
#         frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image = cv2.resize(frame2, (224, 224))
#         image = np.float32(image) / 255.0
#         image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
#         image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
#         blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224), (0, 0, 0), False)
#         net_openvino.setInput(blob)
#         probs = net_openvino.forward()
#         index = np.argmax(probs)
#         # print('class_id: {}  class_name: {}'.format(index,label_name[index]))
#         curr_time = timer()
#         exec_time = curr_time - prev_time
#         prev_time = curr_time
#         accum_time = accum_time + exec_time
#         curr_fps = curr_fps + 1
#         if accum_time > 1:
#             accum_time = accum_time - 1
#             fps = "FPS: " + str(curr_fps)
#             curr_fps = 0
#         text ="label:{}   {}".format(label_name[index],fps)
#         cv2.putText(frame, text=text, org=(150, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                     fontScale=3, color=(0, 0, 255), thickness=3)
#         # cv2.putText(result, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
#         cv2.namedWindow("result", cv2.WINDOW_NORMAL)
#         cv2.imshow("result", frame)
#         if isOutput:
#             out.write(frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

def get_cap():
    video_path = r"E:\zdk\videos\dataset1\raw\26.1.mp4"
    # video_path = r"rtmp://58.200.131.2:1935/livetv/hunantv"
    cap = vid = cv2.VideoCapture(video_path)
    for i in range(20):
        print(i,cap.get(i))

if __name__ == '__main__':
    # img_path = r"data/imgs/drink.jpg"
    # i,l = detect_image_path(img_path,False)
    # print('class_id: {}  class_name: {}'.format(i,l))

    # video_path = r"./data/video/26.1.mp4"
    video_path = r"rtmp://58.200.131.2:1935/livetv/hunantv"
    video_path = r"./data/video/drive.avi"
    # video_path = r"./data/video/output.avi"
    # video_path = "rtmp://202.115.17.6:8002/live/test3"
    # video_path ="rtsp://admin:admin@202.115.17.6:554/h265/ch1/main/av_stream"
    # video_path = "rtmp://202.115.17.6:8002/live/test2"
    start = time.time()
    # detect_video(video_path)

    detect_video_dnn(video_path)

    # detect_video_onnx(video_path)

    # detect_video_openvino(video_path)
    end = time.time()
    total = end -start
    print("total time:{}".format(total))
    # get_cap()