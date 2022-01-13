import os
import numpy as np
import copy

# 文件夹下直接为类别文件夹
def createTxt(root,filename):
    data_path = root
    dirs = os.listdir(data_path)
    # dirs.sort()
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

# 文件夹下按人分别建立子文件夹
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
                if c =="delete":
                    continue
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

# old labels = ["正常", "侧视", "喝水", "吸烟", "操作中控", "玩手机", "侧身拿东西", "整理仪容", "接电话"]
# new labels = ["正常", "喝水", "吸烟", "操作中控", "玩手机", "接电话"] 6 [0,-1,1,2,3,4,-1,-1,5]
# new labels = ["正常", "侧视", "喝水", "吸烟", "玩手机", "接电话"] 6 2

# new labels = ["正常", "喝水", "吸烟", "操作中控", "玩手机", "接电话","其他"] 7

# new labels = ["正常", "侧视", "喝水", "吸烟", "玩手机", "接电话", "其他] 7 2

# old labels = ["正常", "喝水", "吸烟", "玩手机", "侧身拿东西", "接电话",其他] 7 3

# new labels = ["正常", "侧视", "喝水", "吸烟", "操作中控", "玩手机", "侧身拿东西", "接电话"] 8
# 转换标签
def convert_Label():

    # 9类转6类
    # source_txt = r"data/txt/12_23_12_addpre_train224.txt"
    # dest_txt = r"data/txt6/12_23_12_addpre_train224_6.txt"
    # source_txt = r"data/txt/12_23_12_addpre_test224.txt"
    # dest_txt = r"data/txt6/12_23_12_addpre_test224_6.txt"
    # source_txt = r"data/txt/12_23_12_addpre_train224_addcrop.txt"
    # dest_txt = r"data/txt6/12_23_12_addpre_train224_addcrop_6.txt"
    # source_txt = r"data/txt/12_23_12_addpre_train224_kg2my_aucv2_my.txt"
    # dest_txt = r"data/txt6/12_23_12_addpre_train224_kg2my_aucv2_my_6.txt"
    # source_txt = r"data/txt/12_23_12_addpre_train224_kg2my_aucv2_my_addcrop.txt"
    # dest_txt = r"data/txt6/12_23_12_addpre_train224_kg2my_aucv2_my_addcrop_6.txt"

    # source_txt = r"data/txt_raw/total_train.txt"
    # dest_txt = r"data/txt_raw/total_train_c6.txt"
    # source_txt = r"data/txt_raw/total_test.txt"
    # dest_txt = r"data/txt_raw/total_test_c6.txt"
    # convert_index = [0, -1, 1, 2, 3, 4, -1, -1, 5]

    # # 9类转6类 2 new labels = ["正常", "侧视", "喝水", "吸烟", "玩手机", "接电话"]
    # source_txt = r"data/txt_raw/total_train.txt"
    # dest_txt = r"data/txt_raw/total_train_62.txt"
    # convert_index = [0, 1, 2, 3, -1, 4, -1, -1, 5]

    # # 9类转7类
    # source_txt = r"data/txt/12_23_12_addpre_train224.txt"
    # dest_txt = r"data/txt7/12_23_12_addpre_train224_7.txt"
    # source_txt = r"data/txt/12_23_12_addpre_test224.txt"
    # dest_txt = r"data/txt7/12_23_12_addpre_test224_7.txt"
    # source_txt = r"data/txt/12_23_12_addpre_train224_addcrop.txt"
    # dest_txt = r"data/txt7/12_23_12_addpre_train224_addcrop_7.txt"
    # source_txt = r"data/txt/12_23_12_addpre_test224_addcrop.txt"
    # dest_txt = r"data/txt7/12_23_12_addpre_test224_addcrop_7.txt"
    # source_txt = r"data/txt/12_23_12_addpre_train224_kg2my_aucv2_my.txt"
    # dest_txt = r"data/txt7/12_23_12_addpre_train224_kg2my_aucv2_my_7.txt"
    # source_txt = r"data/txt/12_23_12_addpre_train224_kg2my_aucv2_my_addcrop.txt"
    # dest_txt = r"data/txt7/12_23_12_addpre_train224_kg2my_aucv2_my_addcrop_7.txt"

    # source_txt = r"data/txt_raw/total_train.txt"
    # dest_txt = r"data/txt_raw/total_train_c7.txt"
    # # source_txt = r"data/txt_raw/total_test.txt"
    # # dest_txt = r"data/txt_raw/total_test_c7.txt"
    # convert_index = [0, 6, 1, 2, 3, 4, 6, 6, 5]

    # 9类转7类2
    # source_txt = r"data/txt_raw/total_train.txt"
    # dest_txt = r"data/txt_raw/total_train_c7_2.txt"
    # source_txt = r"data/txt_raw/total_test.txt"
    # dest_txt = r"data/txt_raw/total_test_c7_2.txt"
    # source_txt = r"data/txt_raw_crop/total_train_crop.txt"
    # dest_txt = r"data/txt_raw_crop/total_train_crop_72.txt"
    # source_txt = r"data/txt_raw_crop/total_test_crop.txt"
    # dest_txt = r"data/txt_raw_crop/total_test_crop_72.txt"
    # convert_index = [0, 1, 2, 3, 6, 4, 6, 6, 5]

    # 9类转7类3
    # old labels = ["正常", "喝水", "吸烟", "玩手机", "侧身拿东西", "接电话",其他] 7 3
    # source_txt = r"data/txt_raw_crop/total_train_crop.txt"
    # dest_txt = r"data/txt_raw_crop/total_train_crop_73.txt"
    # source_txt = r"data/txt_raw/total_test.txt"
    # dest_txt = r"data/txt_raw/total_test_73.txt"
    # convert_index = [0, 6, 1, 2, 6, 3, 4, 6, 5]

    # # 9类转8类
    # # new labels = ["正常", "侧视", "喝水", "吸烟", "操作中控", "玩手机", "侧身拿东西", "接电话"] 8
    # source_txt = r"data/bus/test224.txt"
    # dest_txt = r"data/bus/test224_8.txt"
    # # source_txt = r"data/bus/addcar_test224.txt"
    # # dest_txt = r"data/bus/addcar_test224_8.txt"
    # convert_index = [0, 1, 2, 3, 4, 5, 6, -1, 7]

    # 9类转6类  new labels = ["正常", "喝水", "吸烟", "操作中控", "玩手机", "接电话"]
    # source_txt = r"data/ours/224/train_crop224.txt"
    # dest_txt = r"data/ours/224/train_crop224_6.txt"
    source_txt = r"data/ours/224/test_crop224.txt"
    dest_txt = r"data/ours/224/test_crop224_6.txt"
    convert_index = [0, -1, 1, 2, 3, 4, -1, -1, 5]

    with open(source_txt, encoding='utf-8') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        old_index = int(line.split()[1])
        img_path = line.split()[0]
        new_index = convert_index[old_index]
        if new_index == -1:
            continue
        else:
            # 需要转义\n
            new_lines.append("{} {}\n".format(img_path, new_index))

    print("old num:{}; new num:{}".format(len(lines),len(new_lines)))

    with open(dest_txt, "w", encoding="utf-8") as f:
        for i in range(len(new_lines)):
            f.write(new_lines[i])

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
    # data_path = r"D:\dataset\state-farm-distracted-driver-detection\imgs\train224"
    # createTxt(data_path, "data/stateFarm/stateFarm.txt")
    # data_path = r"D:\datasets\11_16\dataset\test224"
    # createTxt(data_path, "data/ours/cut224/11_16_test224.txt")
    # data_path = r"D:\datasets\11_16\dataset\test_crop224"
    # createTxt(data_path, "data/ours/cut224/11_16_test_crop224.txt")
    # data_path = r"D:\datasets\11_16\dataset\train224"
    # createTxt(data_path, "data/ours/cut224/11_16_train224.txt")
    # data_path = r"D:\datasets\11_09\train_crop224"
    # createTxt(data_path, "data/ours/cut224/11_9_train_crop224.txt")
    # data_path = r"D:\datasets\3_23\cam_chen\dataset\test224"
    # createTxt2(data_path, "data/bus/test224.txt")
    # data_path = r"D:\datasets\3_23\cam_chen\dataset\train224"
    # createTxt2(data_path, "data/bus/train224.txt")
    # data_path = r"D:\datasets\3_25\cam_he\dataset\train224"
    # createTxt2(data_path, "data/ours/cut224/3_25_train224.txt")
    # data_path = r"D:\datasets\12_23_2\dataset\train_crop224"
    # createTxt2(data_path, "data/ours/cut224/12_232_train_crop224.txt")
    # data_path = r"D:\datasets\3_23\cam_chen\dataset\test"
    # createTxt2(data_path, "data/bus/test.txt")
    # data_path = r"D:\datasets\3_23\cam_chen\dataset\train"
    # createTxt2(data_path, "data/bus/train.txt")
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
    convert_Label()
