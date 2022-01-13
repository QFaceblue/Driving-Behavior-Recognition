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
    # source_dir = r"D:\datasets\12_23_2\dataset\train"
    # dest_dir = r"D:\datasets\12_23_2\dataset\train224"
    # source_dir = r"D:\datasets\3_25\cam_he\dataset\test"
    # dest_dir = r"D:\datasets\3_25\cam_he\dataset\test224"
    source_dir = r"D:\datasets\3_23\cam_chen\dataset\test"
    dest_dir = r"D:\datasets\3_23\cam_chen\dataset\test224"
    weight = 224
    height = 224
    for dir in os.listdir(source_dir):
        image_path = os.path.join(source_dir, dir)
        for name in os.listdir(image_path):
            if name == "delete":
                continue
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
    # source_dir = r"D:\drivedata\train"
    # dest_dir = r"D:\drivedata\train_crop224"
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

def crop224():

    # source_dir = r"D:\datasets\12_23_1\dataset\test"
    # dest_dir = r"D:\datasets\12_23_1\dataset\test_crop224"
    # source_dir = r"D:\datasets\12_23_1\dataset\train"
    # dest_dir = r"D:\datasets\12_23_1\dataset\train_crop224"
    # crop_bbox = (0, 0, 1060, 1080)
    source_dir = r"D:\datasets\12_23_2\dataset\test"
    dest_dir = r"D:\datasets\12_23_2\dataset\test_crop224"
    source_dir = r"D:\datasets\12_23_2\dataset\train"
    dest_dir = r"D:\datasets\12_23_2\dataset\train_crop224"
    crop_bbox = (0, 0, 338, 360)
    weight = 224
    height = 224
    for dir in os.listdir(source_dir):

        dir_path = os.path.join(source_dir, dir)
        dest_path = os.path.join(dest_dir, dir)

        for d in os.listdir(dir_path):

            class_path = os.path.join(dir_path, d)
            dest_class_path = os.path.join(dest_path, d)
            if not os.path.exists(dest_class_path):
                print(dest_class_path)
                os.makedirs(dest_class_path)
            for file in os.listdir(class_path):
                img_path = os.path.join(class_path, file)
                save_path = os.path.join(dest_class_path, file)
                im = Image.open(img_path)
                cropped_im = im.crop(crop_bbox)
                out = cropped_im.resize((weight, height), Image.ANTIALIAS)
                out.save(save_path)

def crop_show():

    img_path = r"D:\datasets\12_23_1\dataset\test\p1_1\0\15-58-24_2a_2002.jpg"
    crop_bbox = (0, 0, 941, 1080)
    img_path = r"D:\datasets\12_23_2\dataset\test\p1\3\p1_0255.jpg"
    crop_bbox = (0, 0, 338, 360)
    im = Image.open(img_path)
    cropped_im = im.crop(crop_bbox)
    cropped_im.show()
    # weight = 224
    # height = 224
    # out = cropped_im.resize((weight, height), Image.ANTIALIAS)
if __name__ == '__main__':
    # resize()
    resize2()
    # crop()
    # crop224()
    # crop_show()
