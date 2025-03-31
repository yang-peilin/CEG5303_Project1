import os
import xml.etree.ElementTree as ET
from PIL import Image
import random

VOC_CLASSES = [
    "bicycle", "bird", "bus", "car", "cat", "dog", "motorbike", "person"
]

# 2) 目录路径 - 请根据你的实际路径修改
VOC_ROOT = "/root/autodl-tmp/ypl/yolov7_adv/VOCdevkit/VOC2007_SMALL_ADV"
ANNOTATIONS_DIR = os.path.join(VOC_ROOT, "Annotations")
IMAGES_DIR = os.path.join(VOC_ROOT, "JPEGImages")
LABELS_DIR = os.path.join(VOC_ROOT, "labels")  # 转换后txt标注存这里

# 3) 如果你想用官方给出的split列表，则在 ImageSets/Main 下有train.txt/val.txt/test.txt
#    如果想自定义随机划分，可以设置以下参数：
RANDOM_SPLIT = False  # True = 根据下面比例随机生成train/val/test
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1


# test = 剩余

def voc_xml_to_yolo_txt(xml_file):
    """将一个VOC格式的xml标注文件转换为YOLO格式文本。返回[行, ...]列表。"""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 获取对应图片的宽高
    filename = root.find('filename').text
    # 部分标注filename可能与实际文件名有差异，这里用XML名也可以
    # 但最稳妥的是匹配JPEGImages下的同名jpg
    image_path = os.path.join(IMAGES_DIR, filename)

    # 处理一下：如果filename没有后缀，就加上.jpg
    if not os.path.exists(image_path):
        # 去掉后缀再拼 jpg
        base = os.path.splitext(filename)[0]
        image_path = os.path.join(IMAGES_DIR, base + ".jpg")

    # 打开图片获取宽高
    img = Image.open(image_path)
    w, h = img.size

    yolo_labels = []

    # 遍历所有object
    for obj in root.findall('object'):
        cls_name = obj.find('name').text

        # 若类别不在VOC_CLASSES里，就跳过
        if cls_name not in VOC_CLASSES:
            continue
        cls_id = VOC_CLASSES.index(cls_name)

        # 找到bndbox
        xmlbox = obj.find('bndbox')
        xmin = float(xmlbox.find('xmin').text)
        ymin = float(xmlbox.find('ymin').text)
        xmax = float(xmlbox.find('xmax').text)
        ymax = float(xmlbox.find('ymax').text)

        # 转换为 YOLO 中心点+宽高(归一化)
        x_center = (xmin + xmax) / 2.0 / w
        y_center = (ymin + ymax) / 2.0 / h
        box_w = (xmax - xmin) / w
        box_h = (ymax - ymin) / h

        # 拼成一行： "cls_id x_center y_center box_w box_h"
        yolo_line = f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
        yolo_labels.append(yolo_line)

    return yolo_labels


def main():
    os.makedirs(LABELS_DIR, exist_ok=True)

    # 获取Annotations目录下所有xml文件名
    xml_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.xml')]
    xml_files.sort()

    if RANDOM_SPLIT:
        # 如果我们想自己随机划分 train/val/test
        random.shuffle(xml_files)
        n = len(xml_files)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

        train_files = xml_files[:train_end]
        val_files = xml_files[train_end:val_end]
        test_files = xml_files[val_end:]
    else:
        # 使用 ImageSets/Main 下的 {train,val,test}.txt
        # 你可以自己决定用trainval.txt+test.txt，或者train.txt+val.txt+test.txt
        def read_split(file_name):
            path = os.path.join(VOC_ROOT, "ImageSets", "Main", file_name)
            lines = open(path, 'r').read().strip().split()
            return [x + ".xml" for x in lines]

        train_files = read_split("train.txt")
        val_files = read_split("val.txt")
        test_files = read_split("test.txt")

    # 下面循环三个list把对应文件转换+存储
    splits = [("train", train_files), ("val", val_files), ("test", test_files)]

    for split_name, file_list in splits:
        split_txt_path = os.path.join(VOC_ROOT, f"{split_name}.txt")
        # 用来记录该split对应的图片路径，用于之后YOLO训练

        with open(split_txt_path, 'w') as f_split:
            for xml_name in file_list:
                xml_path = os.path.join(ANNOTATIONS_DIR, xml_name)
                base = os.path.splitext(xml_name)[0]
                # 同名图像
                # 也可能是 base.jpg
                img_path = os.path.join(IMAGES_DIR, base + ".jpg")

                # 转换并写入 label txt
                labels = voc_xml_to_yolo_txt(xml_path)
                if labels:  # 若有目标
                    out_label_path = os.path.join(LABELS_DIR, base + ".txt")
                    with open(out_label_path, 'w') as f_label:
                        f_label.write("\n".join(labels))

                # 往 train.txt/val.txt/test.txt 里写绝对路径或者相对路径
                # YOLOv7 通常需要的是 image 的路径
                f_split.write(img_path + "\n")

    print("转换完成！请查看 labels 文件夹、以及 train.txt/val.txt/test.txt。")


if __name__ == "__main__":
    main()