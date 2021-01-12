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
import multiprocessing as mp

import threading
import queue


def frame_put(frame_q,cap_path,video=False):
    cap = cv2.VideoCapture(cap_path)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    while cap.isOpened():
        return_value, frame = cap.read()
        # frame = cv2.resize(frame, (800, 600))
        if not return_value:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(frame2, (224, 224))
        image = np.float32(image) / 255.0
        image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
        image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
        frame_q.put((frame,image))
        if video:
            # time.sleep(0.05)
            pass
        else:
            frame_q.get() if frame_q.qsize() > 3 else time.sleep(0.01)
                # print("get")

    cap.release()

def predict(frame_q,predict_q,video=False):

    # mydataset
    classes_path = r"data/drive_classes.txt"
    with open(classes_path) as f:
        label_name = [c.strip() for c in f.readlines()]
    num_classes = len(label_name)

    # onnx
    path = r"checkpoint/resnet18/000/B0_acc=84.8921.onnx"
    net = cv2.dnn.readNetFromONNX(path)

    while True:
        # 没有项目自动阻塞
        frame,image = frame_q.get()
        blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224), (0, 0, 0), False)
        net.setInput(blob)
        probs = net.forward()
        index = np.argmax(probs)
        # index =0
        predict_q.put((frame,index))
        if video:
            # time.sleep(0.05)
            pass
        else:
            if predict_q.qsize() > 3:
                predict_q.get()
                print("预测比绘制快！")

def draw(predict_q,title):
    # mydataset
    classes_path = r"data/drive_classes.txt"
    with open(classes_path) as f:
        label_name = [c.strip() for c in f.readlines()]
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        frame,index = predict_q.get()
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
                    fontScale=1, color=(0, 0, 255), thickness=3)
        # cv2.putText(result, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def predict_draw(frame_q,title):
    # mydataset
    classes_path = r"data/drive_classes.txt"
    with open(classes_path) as f:
        label_name = [c.strip() for c in f.readlines()]
    # num_classes = len(label_name)

    # onnx
    path = r"checkpoint/resnet18/000/B0_acc=84.8921.onnx"
    net = cv2.dnn.readNetFromONNX(path)

    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()

    while True:
        # 没有项目自动阻塞
        frame,image = frame_q.get()
        blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224), (0, 0, 0), False)
        net.setInput(blob)
        probs = net.forward()
        index = np.argmax(probs)
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
                    fontScale=2, color=(0, 0, 255), thickness=3)
        # cv2.putText(result, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        # cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def detect_video_dnn_multi(video_path,title,video=False):

    mp.set_start_method(method='spawn')  # init
    frame_q = mp.Queue(maxsize=2)
    predict_q = mp.Queue(maxsize=2)
    put_p = mp.Process(target=frame_put, args=(frame_q, video_path,video))
    predict_p = mp.Process(target=predict, args=(frame_q, predict_q))
    draw_p = mp.Process(target=draw, args=(predict_q,title))
    # 启动进程
    put_p.start()
    predict_p.start()
    draw_p.start()
    # 等待绘制结束
    draw_p.join()
    # 杀死其余进行
    put_p.terminate()
    predict_p.terminate()

def detect_video_dnn_multi_3(video_path,title,video=False):

    mp.set_start_method(method='spawn')  # init
    frame_q = mp.Queue(maxsize=4)
    predict_q = mp.Queue(maxsize=4)
    put_p = mp.Process(target=frame_put, args=(frame_q, video_path,video))
    predict_p = mp.Process(target=predict, args=(frame_q, predict_q))
    predict_p2 = mp.Process(target=predict, args=(frame_q, predict_q))
    predict_p3 = mp.Process(target=predict, args=(frame_q, predict_q))
    draw_p = mp.Process(target=draw, args=(predict_q,title))
    # 启动进程
    put_p.start()
    predict_p.start()
    predict_p2.start()
    predict_p3.start()
    draw_p.start()
    # 等待绘制结束
    draw_p.join()
    # 杀死其余进行
    put_p.terminate()
    predict_p.terminate()
    predict_p2.terminate()
    predict_p3.terminate()

def detect_video_dnn_multi_t(cap_path,title,video=False):

    mp.set_start_method(method='spawn')  # init
    frame_q =queue.Queue(maxsize=2)
    predict_q = queue.Queue(maxsize=2)
    put_t = threading.Thread(target=frame_put, args=(frame_q, cap_path,video))
    predict_t = threading.Thread(target=predict, args=(frame_q, predict_q))
    draw_t = threading.Thread(target=draw, args=(predict_q,title))
    if video:
        predict_t.setDaemon(True)
        draw_t.setDaemon(True)
    else :
        put_t.setDaemon(True)
        predict_t.setDaemon(True)
    # 启动进程
    put_t.start()
    predict_t.start()
    draw_t.start()
    if video:
        # 等待读取帧结束
        put_t.join()
    else:
        # 等待绘制结束
        draw_t.join()

def detect_video_dnn_multi_t_2(cap_path,title,video=False):

    mp.set_start_method(method='spawn')  # init
    frame_q =queue.Queue(maxsize=2)
    put_t = threading.Thread(target=frame_put, args=(frame_q, cap_path,video))
    predict_draw_t = threading.Thread(target=predict_draw, args=(frame_q, title))
    if video:
        predict_draw_t.setDaemon(True)
    else :
        put_t.setDaemon(True)
    # 启动进程
    put_t.start()
    predict_draw_t.start()

    if video:
        # 等待读取帧结束
        put_t.join()
    else:
        # 等待绘制结束
        predict_draw_t.join()

def detect_video_dnn_multi_t_pn(cap_path,title,video=False):

    mp.set_start_method(method='spawn')  # init
    frame_q =queue.Queue(maxsize=2)
    predict_q = queue.Queue(maxsize=2)
    put_t = threading.Thread(target=frame_put, args=(frame_q, cap_path,video))
    predict_t = threading.Thread(target=predict, args=(frame_q, predict_q))
    predict_t2 = threading.Thread(target=predict, args=(frame_q, predict_q))
    predict_t3 = threading.Thread(target=predict, args=(frame_q, predict_q))
    draw_t = threading.Thread(target=draw, args=(predict_q,title))
    if video:
        predict_t.setDaemon(True)
        predict_t2.setDaemon(True)
        predict_t3.setDaemon(True)
        draw_t.setDaemon(True)
    else :
        put_t.setDaemon(True)
        predict_t.setDaemon(True)
        predict_t2.setDaemon(True)
        predict_t3.setDaemon(True)
    # 启动进程
    put_t.start()
    predict_t.start()
    predict_t2.start()
    draw_t.start()
    if video:
        # 等待读取帧结束
        put_t.join()
    else:
        # 等待绘制结束
        draw_t.join()

def detect_video_dnn_multi_2(video_path,title,video=False):

    mp.set_start_method(method='spawn')  # init
    frame_q = mp.Queue(maxsize=2)

    put_p = mp.Process(target=frame_put, args=(frame_q, video_path,video))
    predict_draw_p = mp.Process(target=predict_draw, args=(frame_q, title))
    # 启动进程
    put_p.start()
    predict_draw_p.start()
    # 等待绘制结束
    predict_draw_p.join()
    # 杀死其余进程
    put_p.terminate()

print("process!")
if __name__ == '__main__':

    # video_path = r"E:\zdk\videos\dataset1\raw\26.1.mp4"
    video_path = r"rtmp://58.200.131.2:1935/livetv/hunantv"
    # 是视频还是视频流
    video = False
    start = time.time()
    # 多进程 比单进程还慢
    # detect_video_dnn_multi(video_path,"multiprecess",video)
    # detect_video_dnn_multi_3(video_path, "multiprecess", video)
    # detect_video_dnn_multi_2(video_path, "multiprecess",video)
    # 多线程 407.25535321235657
    # detect_video_dnn_multi_t(video_path, "multiprecess",video)
    # 一个线程读取，一个预测绘制
    detect_video_dnn_multi_t_2(video_path, "multiprecess",video)
    # 开启多个个线程负责预测，问题是绘制顺序不能保证
    # detect_video_dnn_multi_t_pn(video_path, "multiprecess", video)
    end = time.time()
    total = end -start
    print("total time:{}".format(total))