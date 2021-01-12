import xml.etree.ElementTree as ET
import os

if __name__ == "__main__":
    pic_dir = r"D:\code\keras-yolo3\dataset\train"
    save_file = r"data/drive_data.txt"
    classes_path = r"data/drive_classes.txt"
    with open(classes_path) as f:
        classes = [c.strip() for c in f.readlines()]
    with open(save_file, "w", encoding="utf-8") as output:
        for f in os.listdir(pic_dir):
            if f.split('.')[-1] != "xml":
                continue
            with open(os.path.join(pic_dir, f)) as xmlfile:
                tree = ET.parse(xmlfile)
                root = tree.getroot()
                pic_name = root.find('filename').text
                pic_name = pic_name.replace(" ", "")
                for obj in root.iter('object'):

                    classname = obj.find('name').text
                    if classname not in classes:
                        continue
                    label_id = classes.index(classname)
                    # xmlbox = obj.find('bndbox')
                    # xmin = int(xmlbox.find('xmin').text)
                    # ymin = int(xmlbox.find('ymin').text)
                    # xmax = int(xmlbox.find('xmax').text)
                    # ymax = int(xmlbox.find('ymax').text)
                    #
                    # path = "dataset/train/{}".format(pic_name)
                    # output.write("{} {},{},{},{},{}\n".format(
                    #     path, xmin, ymin, xmax, ymax, label_id))
                    path = os.path.join(pic_dir, pic_name)
                    output.write("{} {}\n".format(path, label_id))
