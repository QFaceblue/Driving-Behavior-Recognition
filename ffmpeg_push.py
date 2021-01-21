import subprocess as sp
import cv2 as cv
import time

##rtmpUrl = "rtmp://txy.live-send.acg.tv/live-txy/?streamname=live_68783796_70316579&key=3d5eea1bd6edfd59c16b9bb9416f9a62"
# rtmpUrl = "rtmp://qn.live-send.acg.tv/live-qn/?streamname=live_68783796_70316579&key=3d5eea1bd6edfd59c16b9bb9416f9a62&schedule=rtmp"
# rtmpUrl = "rtmp://live-push.bilivideo.com/live-bvc/?streamname=live_68783796_70316579&key=3d5eea1bd6edfd59c16b9bb9416f9a62&schedule=rtmp"

# rtmpUrl = "rtmp://127.0.0.1:1935/live"
# rtmpUrl = "rtmp://202.115.17.6:8002/live/test3"
# rtmpUrl = "rtmp://202.115.17.6:50010/live/test2"
rtmpUrl = "rtmp://192.168.2.86:50010/live/test2"
rtmpUrl = "rtmp://202.115.17.6:50010/sign/test2"
rtmpUrl = "rtmp://202.115.17.6:50010/hls/test2"
rtmpUrl = "rtmp://202.115.17.6:40000/http_flv/test"
# rtmpUrl = "rtmp://127.0.0.1/demo/stream-1" # http://localhost/live?app=demo&stream=stream-1
# rtmpUrl = "rtmp://127.0.0.1:1935/demo/stream-1" # http://localhost/live?app=demo&stream=stream-1
# rtmpUrl = "rtmp://202.115.17.6:1935/demo/stream-1" # http://202.115.17.6:61935/live?app=demo&stream=stream-1
# cap_path = 0
cap_path = r"rtmp://58.200.131.2:1935/livetv/hunantv"
# cap_path = "http://112.50.243.8/PLTV/88888888/224/3221225900/1.m3u8"
# cap_path = r"rtsp://admin:admin@192.168.1.80:554/h265/ch1/main/av_stream"
# cap_path = r"rtsp://admin:cs237239@192.168.1.80:554/h265/ch1/main/av_stream"
# cap_path = r"rtsp://admin:cs237239@192.168.1.80:554/h265/stream1"
# cap_path = r"rtsp://admin:cs237239@192.168.1.80:554/h265/stream2"
cap = cv.VideoCapture(cap_path)

# Get video information
fps = int(cap.get(cv.CAP_PROP_FPS))
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
print("fps:{} width:{} height:{}".format(fps, width, height))
# ffmpeg command
command = ['ffmpeg',
           '-y',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-pix_fmt', 'bgr24',
           '-s', "{}x{}".format(width, height),
           # '-r', str(fps),
           '-i', '-',
           # '-r', str(fps),
           '-c:v', 'libx264',
           '-pix_fmt', 'yuv420p',
           '-preset', 'ultrafast',
           '-f', 'flv',
           # '-rtmp_buffer',str(0),
           rtmpUrl]
# ~ command = ['ffmpeg',
        # ~ '-y',
        # ~ '-v','quiet', # 不输出推流信息
        # ~ '-f', 'rawvideo',
        # ~ '-vcodec','rawvideo',
        # ~ '-pix_fmt', 'bgr24',
        # ~ '-s', "{}x{}".format(width, height),
        # ~ #'-r', str(fps),
        # ~ '-i', '-',
        # ~ '-c:v', 'libx264',
        # ~ '-pix_fmt', 'yuv420p',
        # ~ '-preset', 'ultrafast',
        # ~ '-f', 'flv', 
        # ~ #'-rtmp_buffer',str(0),
        # ~ rtmpUrl]
# 管道配置
p = sp.Popen(command, stdin=sp.PIPE)

# read webcamera
draw = True
if draw:
    accum_time = 0
    curr_fps = 0
    true_fps = "FPS: ??"
    prev_time = time.time()
while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        if cap_path != 0:
                print("replay the video")
                cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                continue
        else:
            print("Can't receive frame (stream end?). Exiting ...")
            break
    if cap_path ==0:    
            # 垂直翻转
            frame = cv.flip(frame, 0)
            # 水平翻转
            frame = cv.flip(frame, 1)
    if draw:
        curr_time = time.time()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            true_fps = "FPS:" + str(curr_fps)
            curr_fps = 0
        # t =time.asctime(time.localtime(time.time()))
        t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        text = "{} time:{}".format(true_fps, t)

        cv.putText(frame, text=text, org=(10, 50), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=3)
    cv.imshow("front camera", frame)
    # print(cap.get(cv.CAP_PROP_POS_FRAMES))
    # time.sleep(1/fps)
    time.sleep(0.015)
    # write to pipe
    p.stdin.write(frame.tobytes())
    # p.stdin.write(frame.tostring())
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    

# def push_rtmp(pipe,video_path,rtmp_url):
#     return
