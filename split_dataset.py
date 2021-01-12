import os
import numpy as np

# # mydataset
# annotation_path = './data/drive_data.txt'
# train_path = './data/drive_train.txt'
# val_path = './data/drive_val.txt'
# # kaggle dataset
# annotation_path = './data/train.txt'
# train_path = './data/dtrain.txt'
# val_path = './data/dval.txt'
# val_split = 0.1
# tt100k dataset
# annotation_path = './data/tt100k.txt'
# train_path = './data/tt100k_train.txt'
# val_path = './data/tt100k_val.txt'

# annotation_path = './data/daction.txt'
# train_path = './data/daction_train.txt'
# val_path = './data/daction_val.txt'
#
# annotation_path = './data/drive11_9.txt'
# train_path = './data/train11_9.txt'
# val_path = './data/val11_9.txt'

# annotation_path = './data/drive224.txt'
# train_path = './data/train224.txt'
# val_path = './data/val224.txt'

# annotation_path = r'./data/train_11_16s.txt'
# train_path = r'./data/train_11_16s_train.txt'
# val_path = r'./data/train_11_16s_val.txt'

# annotation_path = r'./data/train224_11_16.txt'
# train_path = r'./data/train224_11_16_train.txt'
# val_path = r'./data/train224_11_16_val.txt'

# annotation_path = r'data/kg_total_add.txt'
# train_path = r'data/kg_total_add_t.txt'
# val_path = r'data/kg_total_add_v.txt'

# imagenet2012_100
annotation_path = r'data/imagenet/imagenet2012_100.txt'
train_path = r'data/imagenet/imagenet2012_100_train.txt'
val_path = r'data/imagenet/imagenet2012_100_val.txt'
val_split = 0.1
with open(annotation_path,encoding = 'utf-8') as f:
    lines = f.readlines()
np.random.seed(10101) # seed( ) 用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed( )值，则每次生成的随即数都相同，如果不设置这个值，则系统根据时间来自己选择这个值，此时每次生成的随机数因时间差异而不同。
np.random.shuffle(lines) # 对X进行重排序，如果X为多维数组，只沿第一条轴洗牌，输出为None，改变原来的X
np.random.seed(None)
num_val = int(len(lines)*val_split)
num_train = len(lines) - num_val
train_set = lines[:num_train]
val_set = lines[num_train:]

with open(train_path, "w", encoding="utf-8") as f:
    for i in range(len(train_set)):
        f.write(train_set[i])

with open(val_path, "w", encoding="utf-8") as f:
    for i in range(len(val_set)):
        f.write(val_set[i])
