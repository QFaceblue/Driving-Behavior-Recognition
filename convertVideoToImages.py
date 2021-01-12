import cv2
import os
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', dest="input",
                    type=str, help="输入视频文件，也可以是包含视频的文件夹")
parser.add_argument('--output', '-o', dest="output",
                    type=str, help="输出图片保存的文件夹路径")
parser.add_argument('--frame_interval', '-t', dest="time_interval",
                    type=int, default=4, help="截取视频图片的时间间隔，单位为秒")
parser.add_argument('--size', '-s', dest="imgsize",
                    type=str, required=False, help="图片分辨率，格式如 1280*720，默认保持原状")

#
def convert_video_to_image(video_path, output_dir, img_size=None, interval_frame=4):
    # print("output_dir={}".format(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.split(video_path)[1]
    v_name = video_name.split(".")[0]
    cap = cv2.VideoCapture(video_path)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    # print("FPS={}".format(FPS))
    if math.isinf(FPS):
        FPS = 24

    frame_cnt = 0
    img_cnt = 0
    while(True):
        ret, frame = cap.read()
        if ret == False:
            print("false")
            break
        if frame_cnt % interval_frame == 0:
            img_cnt += 1
            img_file_name = "{}_{:0>4}.jpg".format(v_name, img_cnt)
            file_path = os.path.join(output_dir, img_file_name)
            print(file_path)
            if img_size is not None:
                frame = cv2.resize(frame, img_size)
            cv2.imwrite(file_path, frame)
        frame_cnt += 1
    cap.release()

# python convertVideoToImages.py --input D:\drivedata\11-9\mp4 --output D:\drivedata\11-9\images
# python convertVideoToImages.py --input D:\datasets\11_16\videos --output D:\datasets\11_16\images
# python convertVideoToImages.py --input D:\datasets\12_23_2\videos --output D:\datasets\12_23_2\images
# python convertVideoToImages.py --input D:\datasets\12_23_1\raw\mp4 --output D:\datasets\12_23_1\raw_images
if __name__ == "__main__":
    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    frame_interval = args.time_interval

    allow_format = (".mp4", ".flv", ".mov", ".avi")

    if os.path.isfile(input_path):
        convert_video_to_image(input_path, output_path,
                               interval_frame=frame_interval)
        print("convert video file: {}".format(input_path))

    elif os.path.isdir(input_path):
        for video_name in os.listdir(input_path):
            file_path = os.path.join(input_path, video_name)
            out_path = os.path.join(output_path, os.path.splitext(video_name)[0])
            if os.path.isdir(file_path):
                continue
            if os.path.splitext(file_path)[1] not in allow_format:
                continue
            convert_video_to_image(
                file_path, out_path, interval_frame=frame_interval)
            print("convert video file: {}".format(file_path))
