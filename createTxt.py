import os
import numpy as np
import copy

def createTxt(root,filename):
    data_path = root
    dirs = os.listdir(data_path)
    dirs.sort()
    print(dirs)
    with open(filename, "w", encoding="utf-8") as f:
        label = 0
        for c in dirs:
            img_path = os.path.join(data_path, c)
            if os.path.isdir(img_path):
                imgs = os.listdir(img_path)
                #             print(imgs)
                for img in imgs:
                    fullpath = os.path.join(img_path, img)
                    f.write("{} {}\n".format(fullpath, label))
            label += 1

def createTxt2(root,filename):
    data_path = root
    dirs = os.listdir(data_path)
    dirs.sort()
    print(dirs)
    with open(filename, "w", encoding="utf-8") as f:
        for p in dirs:
            path = os.path.join(data_path, p)
            print(path)
            label = 0
            for c in os.listdir(path):
                img_path = os.path.join(path, c)
                if os.path.isdir(img_path):
                    imgs = os.listdir(img_path)
                    #             print(imgs)
                    for img in imgs:
                        fullpath = os.path.join(img_path, img)
                        f.write("{} {}\n".format(fullpath, label))
                label += 1

def convertName(root):
    data_path = root
    dirs = os.listdir(data_path)
    dirs.sort()
    print(dirs)
    for c in dirs:
        dir_path = os.path.join(data_path, c)
        if os.path.isdir(dir_path):
            imgs = os.listdir(dir_path)
            # print(imgs)
            for index, img in enumerate(imgs):
                src_path = os.path.join(dir_path, img)
                dest_path = os.path.join(dir_path, "{}.jpg".format(index))
                # print(src_path, dest_path)
                os.rename(src_path, dest_path)


def createTxt_100():
    data_path = data_path = r"D:\IDM\imagenet2012\ILSVRC2012_img"
    dirs = os.listdir(data_path)
    dirs.sort()
    print(dirs)
    filename = r"./data/imagenet/imagenet2012_100.txt"
    with open(filename, "w", encoding="utf-8") as f:
        label = 0
        for i, c in enumerate(dirs):
            if i > 99:
                break
            img_path = os.path.join(data_path, c)
            if os.path.isdir(img_path):
                imgs = os.listdir(img_path)
                #             print(imgs)
                for img in imgs:
                    fullpath = os.path.join(img_path, img)
                    f.write("{} {}\n".format(fullpath, label))
            label += 1

def convert_imagenetVal():

    train_path = r"D:\IDM\imagenet2012\ILSVRC2012_img"
    dirs = os.listdir(train_path)
    print(dirs)
    print(len(dirs))
    map_path = r"D:\IDM\imagenet2012\train_label _mapping.txt"
    true_label =[]
    with open(map_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            # print(dirs.index(line.split()[1]))
            true_label.append(dirs.index(line.split()[1]))
    print(true_label)
    label_path = r"D:\IDM\imagenet2012\ILSVRC2012_devkit_t12\data\ILSVRC2012_validation_ground_truth.txt"
    img_label_list = []
    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            index = int(line.split()[0])
            # print(index)
            img_label_list.append(true_label[index-1])
    # print(img_label_list)

    label_map_path = r"D:\IDM\imagenet2012\ILSVRC2012_devkit_t12\data\ILSVRC2012_validation_ground_truth_map.txt"
    with open(label_map_path, "w", encoding="utf-8") as f:
        for label in img_label_list:
            f.write(str(label)+"\n")

def create_imagenet_val():
    label_list=[]
    label_map_path = r"D:\IDM\imagenet2012\ILSVRC2012_devkit_t12\data\ILSVRC2012_validation_ground_truth_map.txt"
    with open(label_map_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            label_list.append(line.split()[0])
    print(label_list)
    val_path = r"D:\IDM\imagenet2012\ILSVRC2012_img_val"
    dirs = os.listdir(val_path)
    print(dirs)
    print(len(dirs))
    val_txt = r"./data/imagenet/imagenet2012_val.txt"
    base_path = r"D:\IDM\imagenet2012\ILSVRC2012_img_val"
    with open(val_txt, "w", encoding="utf-8") as f:
        for index, dir in enumerate(dirs):
            fullpath = os.path.join(base_path, dir)
            f.write("{} {}\n".format(fullpath, label_list[index]))

def shuffleTxt(source, dest):

    with open(source, encoding='utf-8') as f:
        lines = f.readlines()
    np.random.seed(10101)  # seed( ) 用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed( )值，则每次生成的随即数都相同，如果不设置这个值，则系统根据时间来自己选择这个值，此时每次生成的随机数因时间差异而不同。
    np.random.shuffle(lines)  # 对X进行重排序，如果X为多维数组，只沿第一条轴洗牌，输出为None，改变原来的X
    np.random.seed(None)

    with open(dest, "w", encoding="utf-8") as f:
        for i in range(len(lines)):
            f.write(lines[i])

def createByLabel():

    # source_path = r"D:\dataset\state-farm-distracted-driver-detection\imgs\train"
    source_path = r"D:\dataset\AUC\AUC\trainVal224"
    dirs = os.listdir(source_path)
    dirs.sort()
    # dest_path = r"./data/kg2my.txt"
    dest_path = r"./data/AUCv2_my.txt"
    labels = [0, 5, 8, 10, 10, 4, 2, 6, 7, 1]

    with open(dest_path, "w", encoding="utf-8") as f:
        index = 0
        for c in dirs:
            label = labels[index]
            # print(c, label)
            if label ==10:
                index += 1
                continue
            print(c, label)
            img_path = os.path.join(source_path, c)
            if os.path.isdir(img_path):
                imgs = os.listdir(img_path)
                #             print(imgs)
                for img in imgs:
                    fullpath = os.path.join(img_path, img)
                    f.write("{} {}\n".format(fullpath, label))
            index += 1
# c0: safe driving
# c1: texting - right
# c2: talking on the phone - right
# c3: texting - left
# c4: talking on the phone - left
# c5: operating the radio
# c6: drinking
# c7: reaching behind
# c8: hair and makeup
# c9: talking to passenger

def concatTxt():

    # source1 = r'./data/drive224.txt'
    # source2 = r"./data/kg2my.txt"
    # dest = r"./data/kgAddmy.txt"

    source1 = r'data/total_train.txt'
    source2 = r"./data/kg2my.txt"
    dest = r"./data/kg_total.txt"
    with open(source1, encoding='utf-8') as f:
        lines1 = f.readlines()
    with open(source2, encoding='utf-8') as f:
        lines2 = f.readlines()
    dest_lines = lines1+lines2
    print(len(dest_lines))
    np.random.seed(10101)
    np.random.shuffle(dest_lines)
    np.random.seed(None)

    with open(dest, "w", encoding="utf-8") as f:
        for i in range(len(dest_lines)):
            f.write(dest_lines[i])

def countTxt(path=r"./data/kgAddmy.txt", class_num=9):

    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
    total = len(lines)
    print("total:{}".format(total))
    c_num = [0]*class_num
    for line in lines:
        # print(line.split()[1])
        c_num[int(line.split()[1])] += 1
    print(c_num)

def addClass():

    # source = r"./data/kgAddmy.txt"
    # dest_path = r"./data/kgAddmy_add.txt"
    source = r"data/kg_total.txt"
    dest_path = r"data/kg_total_add.txt"
    add_num = 5
    with open(source, encoding='utf-8') as f:
        lines = f.readlines()
    print(len(lines))
    new_lines = copy.deepcopy(lines)

    for line in lines:
        if int(line.split()[1]) == 3:
            for i in range(add_num):
                new_lines.append(line)
    print(len(new_lines))
    np.random.seed(10101)
    np.random.shuffle(new_lines)
    np.random.seed(None)
    with open(dest_path, "w", encoding="utf-8") as f:
        for i in range(len(new_lines)):
            f.write(new_lines[i])

if __name__ == '__main__':

    # createTxt_100()
    # convert_imagenetVal()
    # create_imagenet_val()
    # data_path = r"D:\drivedata\test_crop224"
    # createTxt(data_path, "./data/txt/119_testcrop224.txt")
    data_path = r"D:\datasets\12_23_1\dataset\train224"
    createTxt2(data_path, "data/txt/12_23_1_train224.txt")
    # data_path = r"D:\datasets\12_23_2\dataset\train224"
    # createTxt2(data_path, "data/txt/12_23_2_train224.txt")
    # data_path = r"D:\dataset\VLR-40\train"
    # convertName(data_path)
    # source = r"./data/imagenet/imagenet2012_trains.txt"
    # dest = r"./data/imagenet/imagenet2012_trains.txt"
    # shuffleTxt(source, dest)
    # countTxt(path=r"data/kg_total_add.txt")
    # createByLabel()
    # concatTxt()
    # countTxt()
    # addClass()
