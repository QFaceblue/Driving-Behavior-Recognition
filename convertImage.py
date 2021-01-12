import os
from PIL import Image

def resize():
    # source_dir = r"D:\drivedata\train"
    # dest_dir = r"D:\drivedata\train224"
    source_dir = r"D:\datasets\12_23_2\dataset\test"
    dest_dir = r"D:\datasets\12_23_2\dataset\test224"
    weight = 224
    height = 224
    for dir in os.listdir(source_dir):

        dir_path = os.path.join(source_dir, dir)
        dest_path = os.path.join(dest_dir, dir)
        if not os.path.exists(dest_path):
            # print(dest_path)
            os.makedirs(dest_path)
        print(dir_path)
        for file in os.listdir(dir_path):
            img_path = os.path.join(dir_path, file)
            save_path = os.path.join(dest_path, file)
            im = Image.open(img_path)
            out = im.resize((weight, height), Image.ANTIALIAS)
            out.save(save_path)
# 图片按人分别放置
def resize2():
    # source_dir = r"D:\drivedata\train"
    # dest_dir = r"D:\drivedata\train224"
    # source_dir = r"D:\datasets\12_23_2\dataset\train"
    # dest_dir = r"D:\datasets\12_23_2\dataset\train224"
    source_dir = r"D:\datasets\12_23_1\dataset\train"
    dest_dir = r"D:\datasets\12_23_1\dataset\train224"
    weight = 224
    height = 224
    for dir in os.listdir(source_dir):
        image_path = os.path.join(source_dir, dir)
        for name in os.listdir(image_path):
            dir_path = os.path.join(image_path, name)
            dest_path = os.path.join(dest_dir,dir, name)
            if not os.path.exists(dest_path):
                # print(dest_path)
                os.makedirs(dest_path)
            print(dir_path)
            for file in os.listdir(dir_path):
                img_path = os.path.join(dir_path, file)
                save_path = os.path.join(dest_path, file)
                im = Image.open(img_path)
                out = im.resize((weight, height), Image.ANTIALIAS)
                out.save(save_path)

def crop():

    # source_dir = r"D:\datasets\11_16\dataset\train"
    # dest_dir = r"D:\datasets\11_16\dataset\train_crop224"
    # source_dir = r"D:\datasets\11_16\dataset\test"
    # dest_dir = r"D:\datasets\11_16\dataset\test_crop224"
    # crop_bbox = (0, 0, 1252, 1296)
    source_dir = r"D:\drivedata\train"
    dest_dir = r"D:\drivedata\train_crop224"
    source_dir = r"D:\drivedata\test"
    dest_dir = r"D:\drivedata\test_crop224"
    crop_bbox = (0, 0, 1060, 1080)
    weight = 224
    height = 224

    for dir in os.listdir(source_dir):

        dir_path = os.path.join(source_dir, dir)
        dest_path = os.path.join(dest_dir, dir)
        if not os.path.exists(dest_path):
            # print(dest_path)
            os.makedirs(dest_path)
        print(dir_path)
        for file in os.listdir(dir_path):
            img_path = os.path.join(dir_path, file)
            save_path = os.path.join(dest_path, file)
            im = Image.open(img_path)
            cropped_im = im.crop(crop_bbox)
            out = cropped_im.resize((weight, height), Image.ANTIALIAS)
            out.save(save_path)

if __name__ == '__main__':
    # resize()
    resize2()
    # crop()
