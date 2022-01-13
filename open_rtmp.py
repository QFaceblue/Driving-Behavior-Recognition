import time
import cv2


def get_cap(cap_path, draw=True):
    cap = cv2.VideoCapture(cap_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("fps:{} width:{} height:{}".format(fps, width, height))
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = time.time()
    while (cap.isOpened()):
        ret, frame = cap.read()
        # crop crop_bbox = (0, 0, 1252, 1296) 2304*1296
        # cw = int(1252 / 2304 * 640)
        # ch = 360
        # frame = frame[0:ch, 0:cw]
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if draw:
            curr_time = time.time()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            text = "{}".format(fps)
            # print(text)
            cv2.putText(frame, text=text, org=(150, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(0, 0, 255), thickness=3)
        cv2.imshow("cap", frame)
        # time.sleep(1/fps)
        time.sleep(0.015)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def save_cap(cap_path, save_path):

    cap = cv2.VideoCapture(0)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 保存视频的编码
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(save_path, fourcc, 30.0, (640, 480))
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            # frame = cv2.flip(frame, 0)
            # write the flipped frame
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # cap_path = r"rtmp://202.115.17.6:8002/live/test2"
    # cap_path = "http://112.50.243.8/PLTV/88888888/224/3221225900/1.m3u8"
    # cap_path = r"rtmp://58.200.131.2:1935/livetv/hunantv"
    # cap_path = r"D:\datasets\11_16\videos\shite.mp4"
    cap_path = r"rtmp://live-push.bilivideo.com/live-bvc/?streamname=live_68783796_70316579&key=3d5eea1bd6edfd59c16b9bb9416f9a62&schedule=rtmp"
    # cap_path = r"rtmp://127.0.0.1/demo/stream-1"
    # cap_path = r"rtmp://202.115.17.6::51935/demo/stream-1"
    # cap_path = r"http://192.168.1.37/live?port=1935&app=demo&stream=stream-1"
    cap_path = r"rtmp://202.115.17.6:50010/live/test2"
    cap_path = r"rtmp://202.115.17.6:40000/http_flv/test"
    # cap_path = r"http://192.168.2.222/live?port=50000&app=http_flv&stream=test_raw"
    cap_path = r"rtmp://192.168.8.103:1935/live/test_raw"
    # cap_path = r"rtmp://202.115.17.6:40000/http_flv/test_detect"
    # cap_path = r"rtmp://202.115.17.6:40000/http_flv/test_raw"
    # cap_path = 0

    # cap_path = r"D:\datasets\11_16\videos\shite.mp4"
    # cap_path = r"rtsp://admin:cs237239@192.168.1.80:554/h265/ch1/main/av_stream"
    # cap_path = r"rtsp://admin:cs237239@192.168.191.1:554/h265/ch1/main/av_stream"
    # cap_path = r"rtsp://admin:cs237239@192.168.188.1:554/h265/ch1/main/av_stream"
    # 4g wifi tplink
    # cap_path = r"rtsp://admin:cs237239@192.168.8.104:554/h265/ch1/main/av_stream"
    # cap_path = r"rtsp://admin:cs237239@192.168.8.104:554/h265/ch1/sub/av_stream"
    #
    # cap_path = r"rtsp://192.168.188.1:51503/A1D18CA94E7ACDA25A402E7B447ACA36_1" #sub
    # cap_path = r"rtsp://192.168.188.1:51503/A1D18CA94E7ACDA25A402E7B447ACA36_0" # main
    # rtsp://192.168.188.1:51503/A1D18CA94E7ACDA25A402E7B447ACA36_1
    # user, pwd, ip, channel = "admin", "cs237239", "192.168.1.158", 1
    #
    # cap_path = "rtsp://%s:%s@%s/h265/ch%s/main/av_stream" % (user, pwd, ip, channel)  # HIKIVISION old version 2015
    # cap_path = "rtsp://%s:%s@%s//Streaming/Channels/%d" % (user, pwd, ip, channel)  # HIKIVISION new version 2017
    # cap_path = "rtsp://%s:%s@%s/cam/realmonitor?channel=%d&subtype=0" % (user, pwd, ip, channel)  # dahua

    save_path = "./videos/{}.avi".format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    # save_path = "./videos/{}.mp4".format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    start = time.time()
    get_cap(cap_path)
    # save_cap(cap_path, save_path)
    end = time.time()
    total = end - start
    print("total time:{}".format(total))
