import csv
import os
csv_path = r"D:\dataset\state-farm-distracted-driver-detection\driver_imgs_list.csv"
# img_path = r"D:\dataset\state-farm-distracted-driver-detection\imgs\train"
# train_path = r"data/txt/kg_train.txt"
# val_path = r"data/txt/kg_val.txt"
img_path = r"D:\dataset\state-farm-distracted-driver-detection\imgs\train224"
train_path = r"data/txt/kg_train224.txt"
val_path = r"data/txt/kg_val224.txt"
with open(csv_path, "r") as f:
    reader = csv.reader(f)
    lists = list(reader)
    print(len(lists))
    p_list = []
    c_list = []
    for l in lists[1:]:
        if l[0] not in p_list:
            p_list.append(l[0])
        if l[1] not in c_list:
            c_list.append(l[1])
    print(len(p_list), p_list)
    print(len(c_list), c_list)
    # print(c_list.index("c2"))
    p_train = p_list[:23]
    p_val = p_list[23:]
    print(p_train, p_val)

p_test = ['p014', 'p035', 'p051']
train_path2 = r"data/txt/kg_train2.txt"
val_path2 = r"data/txt/kg_val2.txt"
train_path2 = r"data/txt/kg_train2_224.txt"
val_path2 = r"data/txt/kg_val2_224.txt"
# with open(train_path2, "w", encoding="utf-8") as f:
#     for l in lists[1:]:
#         if l[0] not in p_test:
#             fullpath = os.path.join(img_path, l[1], l[2])
#             f.write("{} {}\n".format(fullpath, c_list.index(l[1])))
#
# with open(val_path2, "w", encoding="utf-8") as f:
#     for l in lists[1:]:
#         if l[0] in p_test:
#             fullpath = os.path.join(img_path, l[1], l[2])
#             f.write("{} {}\n".format(fullpath, c_list.index(l[1])))

with open(train_path, "w", encoding="utf-8") as f:
    for l in lists[1:]:
        if l[0] in p_train:
            fullpath = os.path.join(img_path, l[1], l[2])
            f.write("{} {}\n".format(fullpath, c_list.index(l[1])))

with open(val_path, "w", encoding="utf-8") as f:
    for l in lists[1:]:
        if l[0] in p_val:
            fullpath = os.path.join(img_path, l[1], l[2])
            f.write("{} {}\n".format(fullpath, c_list.index(l[1])))
# 22424 张图片
# 26 ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024', 'p026', 'p035', 'p039', 'p041', 'p042', 'p045', 'p047', 'p049', 'p050', 'p051', 'p052', 'p056', 'p061', 'p064', 'p066', 'p072', 'p075', 'p081']
# 10 ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
# 最后三人作为验证集
# ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024', 'p026', 'p035', 'p039', 'p041', 'p042', 'p045', 'p047', 'p049', 'p050', 'p051', 'p052', 'p056', 'p061', 'p064', 'p066'] ['p072', 'p075', 'p081']