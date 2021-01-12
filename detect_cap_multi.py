
import cv2
import numpy as np
from timeit import default_timer as timer
import time
import multiprocessing as mp

import threading
import queue

def frame_put(frame_q,cap_path):

    cap = cv2.VideoCapture(cap_path)
    while cap.isOpened():
        return_value, frame = cap.read()
        if not return_value:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame_q.put(frame)
        frame_q.get() if frame_q.qsize() > 2 else time.sleep(0.01)
        # print("get")

    cap.release()

def predict(frame_q,predict_q,video=False):

    # onnx
    # path = r"checkpoint/resnet18/000/B0_acc=84.8921.onnx"
    path = r"checkpoint/resnet18/111/resnet18_kg_acc=99.3310.onnx"
    net = cv2.dnn.readNetFromONNX(path)

    while True:
        # 没有项目自动阻塞
        frame = frame_q.get()
        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(frame2, (224, 224))
        image = np.float32(image) / 255.0
        image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
        image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
        blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224), (0, 0, 0), False)
        net.setInput(blob)
        probs = net.forward()
        index = np.argmax(probs)
        # index =0
        predict_q.put(index)
        if predict_q.qsize() > 1:
            predict_q.get()

def draw(frame_q,predict_q,title):

    # # mydataset
    # classes_path = r"data/drive_classes.txt"
    # with open(classes_path) as f:
    #     label_name = [c.strip() for c in f.readlines()]

    # kaggle dataset
    # label_name = ["正常", "右持手机", "右接电话", "左持手机", "左接电话", "操作仪器", "喝水", "向后侧身", "整理仪容", "侧视"]
    label_name = ["normal", "right holding mobile phone", "right answering phone", "left holding mobile phone",
                  "left answering phone", "operating instrument", "drinking water", "leaning back", "grooming",
                  "side view"]
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    index = 0
    while True:

        frame = frame_q.get()

        if predict_q.qsize() > 0:
            index = predict_q.get()

        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        text = "label:{}   {}".format(label_name[index], fps)
        # print(text)
        cv2.putText(frame, text=text, org=(150, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0, 0, 255), thickness=3)

        # cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def detect_video_dnn_multi(video_path,title):

    mp.set_start_method(method='spawn')  # init
    frame_q = mp.Queue(maxsize=3)
    predict_q = mp.Queue(maxsize=2)
    put_p = mp.Process(target=frame_put, args=(frame_q, video_path))
    predict_p = mp.Process(target=predict, args=(frame_q, predict_q))
    draw_p = mp.Process(target=draw, args=(frame_q,predict_q,title))
    # 启动进程
    put_p.start()
    predict_p.start()
    draw_p.start()
    # 等待绘制结束
    draw_p.join()
    # 杀死其余进行
    put_p.terminate()
    predict_p.terminate()


def detect_video_dnn_multi_t(cap_path,title):

    frame_q =queue.Queue(maxsize=2)
    predict_q = queue.Queue(maxsize=2)
    put_t = threading.Thread(target=frame_put, args=(frame_q, cap_path))
    predict_t = threading.Thread(target=predict, args=(frame_q, predict_q))
    draw_t = threading.Thread(target=draw, args=(frame_q, predict_q, title))

    put_t.setDaemon(True)
    predict_t.setDaemon(True)
    # 启动进程
    put_t.start()
    predict_t.start()
    draw_t.start()
    # 等待绘制结束
    draw_t.join()


print("process!")
if __name__ == '__main__':

    # video_path = r"E:\zdk\videos\dataset1\raw\26.1.mp4"
    video_path = r"rtmp://58.200.131.2:1935/livetv/hunantv"
    # video_path = "http://112.50.243.8/PLTV/88888888/224/3221225900/1.m3u8"
    video_path = "rtmp://202.115.17.6:8002/live/test3"
    # 是视频还是视频流
    start = time.time()
    # 多进程
    # detect_video_dnn_multi(video_path,"multiprecess")
    # 多线程
    detect_video_dnn_multi_t(video_path, "multiprecess")

    end = time.time()
    total = end -start
    print("total time:{}".format(total))