import torch
from efficientnet_pytorch import EfficientNet
# import onnx # 环境问题
import onnxruntime
import onnx
from torchvision import datasets, models, transforms
import json
from PIL import Image
import cv2
import numpy as np
from timeit import default_timer as timer
from ghost_net import ghost_net
from mobilenetv3 import MobileNetV3
from ghostnet import ghostnet
from mnext import mnext

# RuntimeError: ONNX export failed: Couldn't export Python operator SwishImplementation
# 解决办法 model.set_swish(memory_efficient=False)

# 转化onnx模型 为opencvDNN 中间模型
# 再全局环境中执行
# cd D:\Program Files (x86)\Intel\openvino_2020.3.341\bin
# setupvars.bat
# cd D:\Program Files (x86)\Intel\openvino_2020.3.341\deployment_tools\model_optimizer
# --data_type {FP16,FP32,half,float}
# python mo_onnx.py --input_model D:\code\EfficientNet-PyTorch-master\checkpoint\data_12_23\mobilenetv2\888\mobilenetv2_1_12_23_acc=91.6275.onnx --output_dir D:\code\EfficientNet-PyTorch-master\checkpoint\data_12_23\mobilenetv2\888\ --input_shape [1,3,224,224]
# python mo_onnx.py --input_model D:\code\EfficientNet-PyTorch-master\checkpoint\data_12_23\mobilenetv2\888\mobilenetv2_1_12_23_acc=91.6275.onnx --output_dir D:\code\EfficientNet-PyTorch-master\checkpoint\data_12_23\mobilenetv2\8888\ --input_shape [1,3,224,224] --data_type FP16
# python mo_onnx.py --input_model D:\code\EfficientNet-PyTorch-master\checkpoint\B0\000\B0_acc=99.8528.onnx --output_dir D:\code\EfficientNet-PyTorch-master\checkpoint\B0\000\ --input_shape [1,3,224,224]
# python mo_onnx.py --input_model D:\code\EfficientNet-PyTorch-master\checkpoint\resnet18\111\resnet18_kg.onnx --output_dir D:\code\EfficientNet-PyTorch-master\checkpoint\resnet18\111\ --input_shape [1,3,224,224]
def convert():

    model = EfficientNet.from_name('efficientnet-b0',num_classes=10)
    model.set_swish(memory_efficient=False)
    # 加载模型参数
    path = r"checkpoint/B0/000/B0_acc=99.8528.pth"
    to_path = r"checkpoint/B0/000/B0_acc=99.8528.onnx"

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["net"])
    print("loaded model with acc:{}".format(checkpoint["acc"]))
    model.cuda()
    # dummy_input = torch.randn(10, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
    torch.onnx.export(model, dummy_input, to_path, verbose=True)

def convert_resnet18():

    model = models.resnet18(pretrained=False,num_classes=9)
    # 加载模型参数
    path = r"checkpoint/resnet18/000/B0_acc=84.8921.pth"
    to_path = r"checkpoint/resnet18/000/B0_acc=84.8921_2.onnx"

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["net"])
    print("loaded model with acc:{}".format(checkpoint["acc"]))
    model.cuda()
    # dummy_input = torch.randn(10, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
    torch.onnx.export(model, dummy_input, to_path, verbose=True)

def convert_resnet18_kg():

    model = models.resnet18(pretrained=False,num_classes=10)
    # 加载模型参数
    path = r"checkpoint/resnet18/111/resnet18_kg_acc=99.3310.pth"
    to_path = r"checkpoint/resnet18/111/resnet18_kg_acc=99.3310.onnx"

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["net"])
    print("loaded model with acc:{}".format(checkpoint["acc"]))
    model.cuda()
    # dummy_input = torch.randn(10, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
    torch.onnx.export(model, dummy_input, to_path, verbose=True)

def convert_ghostnet_1_kg():

    model =  ghost_net(num_classes=10, width_mult=1.0)
    # 加载模型参数
    path = r"checkpoint/ghost_net/000/ghostnet_kg_acc=97.8591.pth"
    to_path = r"checkpoint/ghost_net/000/ghostnet_kg_acc=97.8591.onnx"

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["net"])
    print("loaded model with acc:{}".format(checkpoint["acc"]))
    model.cuda()
    # dummy_input = torch.randn(10, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
    torch.onnx.export(model, dummy_input, to_path, verbose=True)
# 转化onnx模型
# cd D:\Program Files (x86)\Intel\openvino_2021.1.110\deployment_tools\model_optimizer
# python mo_onnx.py --input_model D:\code\EfficientNet-PyTorch-master\checkpoint\data_11_16\ghostnet\222\ghostnet_1_drive.onnx --output_dir D:\code\EfficientNet-PyTorch-master\checkpoint\data_11_16\ghostnet\222\ --input_shape [1,3,224,224]
def convert_ghostnet_1_my():

    # model =  ghost_net(num_classes=9, width_mult=1.0)
    model = ghostnet(num_classes=9, width=1.)
    # 加载模型参数
    path = r"checkpoint/data_11_16/ghostnet/nopre/222/ghostnet_1_drive=96.5665.pth"
    to_path = r"checkpoint/data_11_16/ghostnet/nopre/222/ghostnet_1_drive.onnx"

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["net"])
    print("loaded model with acc:{}".format(checkpoint["acc"]))
    model.cuda()
    # dummy_input = torch.randn(10, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
    torch.onnx.export(model, dummy_input, to_path, verbose=True)


def convert_ghostnet_0_5_kg():

    model =  ghost_net(num_classes=9, width_mult=0.5)
    # 加载模型参数
    path = r"checkpoint/ghost_net/333/ghostnet_05_kg_acc=68.3453.pth"
    to_path = r"checkpoint/ghost_net/333/ghostnet_05_my_acc=68.3453.onnx"

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["net"])
    print("loaded model with acc:{}".format(checkpoint["acc"]))
    model.cuda()
    model.eval()
    # dummy_input = torch.randn(10, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
    torch.onnx.export(model, dummy_input, to_path, verbose=True)

def convert_ghostnet_0_3_kg():

    model =  ghost_net(num_classes=10, width_mult=0.3)
    # 加载模型参数
    path = r"checkpoint/ghost_net/222/ghostnet_03_kg_acc=89.4291.pth"
    to_path = r"checkpoint/ghost_net/222/ghostnet_03_kg_acc=89.4291.onnx"

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["net"])
    print("loaded model with acc:{}".format(checkpoint["acc"]))
    model.cuda()
    # dummy_input = torch.randn(10, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
    torch.onnx.export(model, dummy_input, to_path, verbose=True)
# 转化onnx模型
# python mo_onnx.py --input_model D:\code\EfficientNet-PyTorch-master\checkpoint\data_11_16\mobilenetv2\pre\0\111\mobilenetv2_1_my_224.onnx --output_dir D:\code\EfficientNet-PyTorch-master\checkpoint\data_11_16\mobilenetv2\pre\0\111\ --input_shape [1,3,224,224]
def convert_mobilenetv2():

    model = models.mobilenet_v2(pretrained=False, num_classes=9)
    # 加载模型参数
    path = r"checkpoint/data_12_23/mobilenetv2/888/mobilenetv2_1_12_23_acc=91.6275.pth"
    to_path = r"checkpoint/data_12_23/mobilenetv2/888/mobilenetv2_1_12_23_acc=91.6275.onnx"
    # to_path = r"checkpoint/data_11_16/mobilenetv2/pre/777/mobilenetv2_1_my_224c.onnx"
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["net"])
    print("loaded model with acc:{}".format(checkpoint["acc"]))
    model.cuda()
    dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
    # dummy_input = torch.randn(1, 3, 160, 160, device='cuda')
    torch.onnx.export(model, dummy_input, to_path, verbose=True)

# 转化onnx模型 为opencvDNN 中间模型
# 再全局环境中执行
# cd D:\Program Files (x86)\Intel\openvino_2020.3.341\bin
# setupvars.bat
# cd D:\Program Files (x86)\Intel\openvino_2020.3.341\deployment_tools\model_optimizer
# python mo_onnx.py --input_model D:\code\EfficientNet-PyTorch-master\checkpoint\data_12_23\mnext\000\mnext_1_12_23_acc=92.1753.onnx --output_dir D:\code\EfficientNet-PyTorch-master\checkpoint\data_12_23\mnext\000\ --input_shape [1,3,224,224]
def convert_mnext():

    model = mnext(num_classes=9, width_mult=1.)
    # 加载模型参数
    path = r"checkpoint/data_12_23/mnext/000/mnext_1_12_23_acc=92.1753.pth"
    to_path = r"checkpoint/data_12_23/mnext/000/mnext_1_12_23_acc=92.1753.onnx"
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["net"])
    print("loaded model with acc:{}".format(checkpoint["acc"]))
    model.cuda()
    dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
    # dummy_input = torch.randn(1, 3, 160, 160, device='cuda')
    torch.onnx.export(model, dummy_input, to_path, verbose=True)

def convert_shufflenetv2():
    # model= models.shufflenet_v2_x1_0(pretrained=False, num_classes=9)
    # path = r"checkpoint/data_11_16/shufflenetv2/pre/111/shufflenetv2_1_my_acc=97.8541.pth"
    # to_path = r"checkpoint/data_11_16/shufflenetv2/pre/111/shufflenetv2_1.onnx"

    model = models.shufflenet_v2_x0_5(pretrained=False, num_classes=9)
    path = r"checkpoint/data_11_16/shufflenetv2/pre/333/shufflenetv2_05_my_acc=99.5708.pth"
    to_path = r"checkpoint/data_11_16/shufflenetv2/pre/333/shufflenetv2_05_my.onnx"

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["net"])
    print("loaded model with acc:{}".format(checkpoint["acc"]))
    model.cuda()
    # dummy_input = torch.randn(10, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
    torch.onnx.export(model, dummy_input, to_path, verbose=True)

def convert_mobilenetv2_kaggle():

    model = models.mobilenet_v2(pretrained=False, num_classes=10)
    # 加载模型参数
    path = r"checkpoint/mobilenetv2/111/mobilenetv2_kg_acc=99.5986.pth"
    to_path = r"checkpoint/mobilenetv2/111/mobilenetv2_kg_acc=99.5986.onnx"

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["net"])
    print("loaded model with acc:{}".format(checkpoint["acc"]))
    model.cuda()
    # dummy_input = torch.randn(10, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
    torch.onnx.export(model, dummy_input, to_path, verbose=True)

def convert_mobilenetv2_05_kaggle():

    model = models.mobilenet_v2(pretrained=False,num_classes=10,width_mult=0.5)
    # 加载模型参数
    path = r"checkpoint/mobilenetv2/222/moblienetv2_05_kg_acc=83.0508.pth"
    to_path = r"checkpoint/mobilenetv2/222/moblienetv2_05_kg_acc=83.0508.onnx"

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["net"])
    print("loaded model with acc:{}".format(checkpoint["acc"]))
    model.cuda()
    # dummy_input = torch.randn(10, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
    torch.onnx.export(model, dummy_input, to_path, verbose=True)

def convert_mobilenetv3():
    num_classes = 9
    model = MobileNetV3(n_class=num_classes, mode="small", dropout=0.2, width_mult=1.0)
    # model = MobileNetV3(n_class=num_classes, mode="large", dropout=0.2, width_mult=1.0)
    # 加载模型参数
    path = r"checkpoint/mobilenetv3/000/moblienetv3_s_my_acc=65.4676.pth"
    to_path = r"checkpoint/mobilenetv3/000/moblienetv3_s_my_acc=65.4676.onnx"

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["net"])
    print("loaded model with acc:{}".format(checkpoint["acc"]))
    model.cuda()
    # dummy_input = torch.randn(10, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
    torch.onnx.export(model, dummy_input, to_path, verbose=True)

def convert_resnet50():

    # model = models.resnet50(pretrained=True)
    # model = models.resnet18(pretrained=True)
    model = models.resnext50_32x4d(pretrained=True)
    # model = models.mobilenet_v2(pretrained=True)
    # 加载模型参数
    # path = r"checkpoint/B0/000/B0_acc=99.8528.pth"
    # to_path = r"weights/resnet50.onnx"
    # to_path = r"weights/resnet18.onnx"
    # to_path = r"weights/resnet18_2.onnx"
    to_path = r"weights/resnext50.onnx"
    # to_path = r"weights/mobilenetv2_2.onnx"
    # checkpoint = torch.load(path)
    # model.load_state_dict(checkpoint["net"])
    # print("loaded model with acc:{}".format(checkpoint["acc"]))
    model.cuda()
    # dummy_input = torch.randn(10, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
    torch.onnx.export(model, dummy_input, to_path, verbose=True)


# def test():
#
#     # Load the ONNX model
#     to_path = r"checkpoint\B0\B0_acc=99.8528.onnx"
#     model = onnx.load(to_path)
#
#     # Check that the IR is well formed
#     onnx.checker.check_model(model)
#
#     # Print a human readable representation of the graph
#     onnx.helper.printable_graph(model.graph)

def test_resnet50():

    model = models.resnet50(pretrained=True)

    # Preprocess image
    tfms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    img = tfms(Image.open('./data/imgs/elephant.jpg')).unsqueeze(0)
    print(img.shape) # torch.Size([1, 3, 224, 224])

    # Load ImageNet class names
    labels_map = json.load(open('.\data\labels_map.txt'))
    labels_map = [labels_map[str(i)] for i in range(1000)]

    # Classify
    model.eval()
    with torch.no_grad():
        outputs = model(img)

    # Print predictions
    print('-----')
    for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
        prob = torch.softmax(outputs, dim=1)[0, idx].item()
        print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))
        # print('{label:<75} ({p:.2f}%)'.format(label=idx, p=prob * 100))


def test_resnet50_cv2():

    # model = models.resnet50(pretrained=True).cuda()
    model = models.resnet50(pretrained=True)
    src = cv2.imread("./data/imgs/elephant.jpg") # aeroplane.jpg
    image = cv2.resize(src, (224, 224))
    image = np.float32(image) / 255.0
    image[:,:,] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
    image[:,:,] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
    image = image.transpose((2, 0, 1))
    input_x = torch.from_numpy(image).unsqueeze(0)
    print(input_x.size())

    # pred = model(input_x.cuda())
    pred = model(input_x)
    pred_index = torch.argmax(pred, 1).cpu().detach().numpy()
    print(pred_index)

    # Load ImageNet class names
    labels_map = json.load(open('.\data\labels_map.txt'))
    labels_map = [labels_map[str(i)] for i in range(1000)]
    print("current predict class name : %s"%labels_map[pred_index[0]])
    cv2.putText(src, labels_map[pred_index[0]], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imshow("input", src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def testONNX():

    # Load ImageNet class names
    labels_map = json.load(open('.\data\labels_map.txt'))
    labels_map = [labels_map[str(i)] for i in range(1000)]

    # path = r"weights/resnet18_2.onnx" # inferrence time:0.026459999999999928
    # path = r"checkpoint/mobilenetv2/000/mv2_acc=82.7338.onnx" #inferrence time:0.008720400000000073
    path = r"checkpoint/mobilenetv3/000/moblienetv3_s_my_acc=65.4676.onnx"  # inferrence time:0.013804399999999939

    # path = r"weights/resnet50.onnx" #inferrence time:0.07290449999999993
    # path = r"weights/resnext50.onnx"  # inferrence time:0.11427840000000011

    onnx_session = onnxruntime.InferenceSession(path,None)

    src = cv2.imread("./data/imgs/elephant.jpg")  # aeroplane.jpg
    src2 = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    image = cv2.resize(src2, (224, 224))
    image = np.float32(image) / 255.0
    image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
    image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))

    image = image.transpose(2, 0, 1)  # 转换轴，pytorch为channel first
    image = image.reshape(1, 3, 224, 224)  # barch,channel,height,weight
    pre_t = timer()
    inputs = {onnx_session.get_inputs()[0].name:image}
    probs =  onnx_session.run(None, inputs)
    index = np.argmax(probs)
    curr_t = timer()
    infer_t = curr_t - pre_t
    print("inferrence time:{}".format(infer_t))
    cv2.putText(src, labels_map[index], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imshow("input", src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def testONNX_dataset():

    # Load dataset
    dataset_path = r"data/drive_data.txt"
    with open(dataset_path) as f:
        datasets= [c.strip() for c in f.readlines()]
    path = r"ghostnet_my_nv_05_acc=97.4856_eval.onnx"
    # path = r"checkpoint/ghost_net/555/ghostnet_my_nv_05_acc=97.4856.onnx" # acc:0.2557471264367816,  356/1392
    # path = r"checkpoint/ghost_net/555/ghostnet_my_nv_05_acc=97.4856_eval.onnx" # acc:0.2557471264367816,  356/1392
    onnx_session = onnxruntime.InferenceSession(path, None)
    dnn_net = cv2.dnn.readNetFromONNX(path)

    model = ghost_net(num_classes=9, width_mult=0.5)
    # 加载模型参数
    path = r"checkpoint/ghost_net/555/ghostnet_my_nv_05_acc=97.4856.pth"
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
        dnn_probs = dnn_net.forward()
        dnn_index = np.argmax(dnn_probs)

        # print(image.shape)
        image = image.transpose(2, 0, 1) # 转换轴，pytorch为channel first
        image = image.reshape(1, 3, 224, 224) # barch,channel,height,weight

        # image = []
        # image.append(image)
        # image = np.asarray(image)

        inputs = {onnx_session.get_inputs()[0].name: image}
        probs =  onnx_session.run(None, inputs)
        index = np.argmax(probs)

        model_image =torch.from_numpy(image)
        output = model(model_image)
        model_index = np.argmax(output.detach().numpy())
        print("dnn_probs:{},probs:{},output:{}".format(dnn_probs, probs,output))
        print("dnn_index:{},index:{},model_index:{},label:{}".format(dnn_index,index,model_index,label))
        total +=1
        if index == label:
            right += 1

        if dnn_index == label:
            dnn_right += 1

        if model_index == label:
            model_right += 1
    print("acc:{},dnn_acc:{},model_acc :{}".format(right/total,dnn_right/total,model_right/total))


def testDNN():

    # Load ImageNet class names
    labels_map = json.load(open('.\data\labels_map.txt'))
    labels_map = [labels_map[str(i)] for i in range(1000)]

    # path = r"weights/resnet50.onnx" # inferrence time:0.25801849999999993
    # path = r"weights/resnet18.onnx" # inferrence time:0.09869299999999992
    path = r"weights/resnet18_2.onnx"  # inferrence time:0.09869299999999992
    # path = r"weights/resnext50.onnx" # inferrence time:0.30201659999999997
    # path = r"weights/mobilenetv2.onnx"
    net = cv2.dnn.readNetFromONNX(path)
    src = cv2.imread("./data/imgs/elephant.jpg")  # aeroplane.jpg
    src2 = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    image = cv2.resize(src2, (224, 224))
    image = np.float32(image) / 255.0
    image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
    image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
    pre_t = timer()
    blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224), (0, 0, 0), False)
    net.setInput(blob)
    probs = net.forward()
    index = np.argmax(probs)
    curr_t = timer()
    infer_t = curr_t - pre_t
    print("inferrence time:{}".format(infer_t))
    cv2.putText(src, labels_map[index], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imshow("input", src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 初始化openvino，必须在指定目录运行

# cd D:\Program Files (x86)\Intel\openvino_2020.3.341\bin
# setupvars.bat
# d:
def testDNN_openvino():
    print("testDNN_openvino")
    # Load ImageNet class names
    labels_map = json.load(open('.\data\labels_map.txt'))
    labels_map = [labels_map[str(i)] for i in range(1000)]

    xml_path = r"weights\resnet18.xml"
    bin_path = r"weights\resnet18.bin"

    net = cv2.dnn.readNetFromModelOptimizer(xml_path, bin_path)
    # 下面设置不需要
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    src = cv2.imread("./data/imgs/elephant.jpg")  # aeroplane.jpg
    image = cv2.resize(src, (224, 224))
    image = np.float32(image) / 255.0
    image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
    image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
    blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224), (0, 0, 0), False)
    net.setInput(blob)
    probs = net.forward()
    index = np.argmax(probs)
    cv2.putText(src, labels_map[index], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imshow("input", src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # convert()
    # convert_resnet18()
    # convert_resnet18_kg()
    # convert_ghostnet_1_kg()
    # convert_ghostnet_1_my()
    # convert_ghostnet_0_5_kg()
    # convert_ghostnet_0_3_kg()
    # convert_mobilenetv2()
    convert_mnext()
    # convert_shufflenetv2()
    # convert_mobilenetv2_kaggle()
    # convert_mobilenetv2_05_kaggle()
    # convert_mobilenetv3()
    # convert_resnet50()
    # test()
    # test_resnet50()
    # test_resnet50_cv2()
    # testONNX()
    # testDNN()
    # testDNN_openvino()
    # testONNX_dataset()