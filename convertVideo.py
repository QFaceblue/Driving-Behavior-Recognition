import os

# source_dir = r"D:\drivedata\11-9\mkv"
# dest_dir = r"D:\drivedata\11-9\mp4"
source_dir = r"D:\datasets\12_23_1\raw\mkv"
dest_dir = r"D:\datasets\12_23_1\raw\mp4"
os.makedirs(dest_dir, exist_ok=True)
for video_name in os.listdir(source_dir):
    file_path = os.path.join(source_dir, video_name)
    v_name = video_name.split(".")[0]

    dest_name= v_name+".mp4"
    dest_path = os.path.join(dest_dir, dest_name)
    # print(dest_path)
    cmd = "ffmpeg -i  {} {}".format(file_path, dest_path)
    print(cmd)
    os.system(cmd)