# -------------------------------------#
#       对单张图片进行预测
# -------------------------------------#
import onnxruntime
from PIL import Image,ImageFont, ImageDraw
import numpy as np
import time
import cv2
# 8class
crop_path = r"checkpoint/data_12_23/mobilenetv2/888/mobilenetv2_1_12_23_acc=91.6275.onnx"
# 9class
path = r"checkpoint/data_3_25_all/mobilenetv2/111/mobilenetv2_224_acc=85.6154.onnx"
# num_classes = 9
num_classes = 8
onnx_session = onnxruntime.InferenceSession(path, None)
labels_8 = ["正常", "侧视", "喝水", "吸烟", "操作中控", "玩手机", "侧身拿东西", "接电话"]
labels_9 = ["正常", "侧视", "喝水", "吸烟", "操作中控", "玩手机", "侧身拿东西", "整理仪容", "接电话"]
video_path = r"data/video/xw.mp4"

def softmax_np(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax

def detect_img():
    font = ImageFont.truetype(font='data/video/simhei.ttf', size=int(30))  # 20
    while True:
        img = input('Input image filename:')
        try:
            frame = cv2.imread(img)
            frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            print('Open Error! Try again!')
            continue
        else:

            # crop crop_bbox = (0, 0, 1252, 1296) 2304*1296
            # ~ cw = int(1252/2304*640)
            # ~ ch = 360
            # ~ frame2 = frame2[0:ch,0:cw]
            start = time.time()
            image = cv2.resize(frame2, (224, 224))
            # image = cv2.resize(frame2, (160, 160))
            # print("resize time:", time.time()-start)
            image = np.float32(image) / 255.0
            image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
            image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))

            image = image.transpose(2, 0, 1)  # 转换轴，pytorch为channel first
            image = image.reshape(1, 3, 224, 224)  # barch,channel,height,weight
            # image = image.reshape(1, 3, 160, 160) # barch,channel,height,weight
            inputs = {onnx_session.get_inputs()[0].name: image}
            print("preprocess time:", time.time() - start)
            # 注意返回为三维数组（1,1,class_num)
            probs = onnx_session.run(None, inputs)
            index = np.argmax(probs)
            # print(probs)
            softmax_probs = softmax_np(np.array(probs))
            prob = softmax_probs.max()
            # print(index,prob)
            # index =0
            print(labels_9[index], prob)
            predict_time = time.time() - start
            print("preprocess time:", predict_time)
            img_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_PIL)

            text = "{}:{:.2f}  检测时间：{:.3f}s".format(labels_9[index], prob, predict_time)
            draw.text((100, 30), text, fill=(255, 0, 0), font=font)
            # img_PIL.show()
            frame = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
            cv2.imshow("img", frame)
            cv2.waitKey(0)
            # cv2.imshow("result", frame)
            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #     break

def detect_video(output_path=""):
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
        print("!!! TYPE:", output_path,
            video_FourCC, video_fps, video_size)
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)

    accum_time = 0
    detect_fps = 0
    d_fps = "detect_FPS: ??"
    prev_time = time.time()
    font = ImageFont.truetype(font='data/video/simhei.ttf', size=int(30))  # 20
    while True:
        return_value, frame = vid.read()
        if not return_value:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        start = time.time()
        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # crop crop_bbox = (0, 0, 1252, 1296) 2304*1296
        # ~ cw = int(1252/2304*640)
        # ~ ch = 360
        # ~ frame2 = frame2[0:ch,0:cw]
        image = cv2.resize(frame2, (224, 224))
        # image = cv2.resize(frame2, (160, 160))
        # print("resize time:", time.time()-start)
        image = np.float32(image) / 255.0
        image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
        image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))

        image = image.transpose(2, 0, 1)  # 转换轴，pytorch为channel first
        image = image.reshape(1, 3, 224, 224)  # barch,channel,height,weight
        # image = image.reshape(1, 3, 160, 160) # barch,channel,height,weight
        inputs = {onnx_session.get_inputs()[0].name: image}
        print("preprocess time:", time.time() -start)
        # 注意返回为三维数组（1,1,class_num)
        probs = onnx_session.run(None, inputs)
        print("predict time:", time.time() - start)
        index = np.argmax(probs)
        # print(probs)
        softmax_probs = softmax_np(np.array(probs))
        prob = softmax_probs.max()
        # print(index,prob)
        # index =0
        print(labels_9[index], prob)

        img_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_PIL)
        detect_fps = detect_fps + 1
        curr_time = time.time()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        if accum_time > 1:
            accum_time = accum_time - 1
            d_fps = "detect_FPS:" + str(detect_fps)
            detect_fps = 0
        text = "{}:{:.2f} {}".format(labels_9[index], prob, d_fps)
        draw.text((200, 30), text, fill=(255, 0, 0), font=font)
        frame = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", frame)
        if isOutput:
            out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    # detect_video(r"data/video/test.mp4")
    # detect_video()
    detect_img()
