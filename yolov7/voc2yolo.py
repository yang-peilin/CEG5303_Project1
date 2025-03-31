import os
import xml.etree.ElementTree as ET
from PIL import Image
import random

# VOC官方20类
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# 目录路径
VOC_ROOT = "/root/autodl-tmp/ypl/yolov7/VOCdevkit/VOC2007"
ANNOTATIONS_DIR = os.path.join(VOC_ROOT, "Annotations")
IMAGES_DIR = os.path.join(VOC_ROOT, "JPEGImages")
LABELS_DIR = os.path.join(VOC_ROOT, "labels")

# 如果想自定义随机划分
RANDOM_SPLIT = False
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1



def voc_xml_to_yolo_txt(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find('filename').text
    image_path = os.path.join(IMAGES_DIR, filename)

    if not os.path.exists(image_path):
        base = os.path.splitext(filename)[0]
        image_path = os.path.join(IMAGES_DIR, base + ".jpg")

    img = Image.open(image_path)
    w, h = img.size
    yolo_labels = []

    for obj in root.findall('object'):
        cls_name = obj.find('name').text
        if cls_name not in VOC_CLASSES:
            continue
        cls_id = VOC_CLASSES.index(cls_name)
        xmlbox = obj.find('bndbox')
        xmin = float(xmlbox.find('xmin').text)
        ymin = float(xmlbox.find('ymin').text)
        xmax = float(xmlbox.find('xmax').text)
        ymax = float(xmlbox.find('ymax').text)

        x_center = (xmin + xmax) / 2.0 / w
        y_center = (ymin + ymax) / 2.0 / h
        box_w = (xmax - xmin) / w
        box_h = (ymax - ymin) / h

        yolo_line = f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
        yolo_labels.append(yolo_line)

    return yolo_labels


def main():
    os.makedirs(LABELS_DIR, exist_ok=True)

    xml_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.xml')]
    xml_files.sort()

    if RANDOM_SPLIT:
        random.shuffle(xml_files)
        n = len(xml_files)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

        train_files = xml_files[:train_end]
        val_files = xml_files[train_end:val_end]
        test_files = xml_files[val_end:]
    else:
        def read_split(file_name):
            path = os.path.join(VOC_ROOT, "ImageSets", "Main", file_name)
            lines = open(path, 'r').read().strip().split()
            return [x + ".xml" for x in lines]

        train_files = read_split("train.txt")
        val_files = read_split("val.txt")
        test_files = read_split("test.txt")

    splits = [("train", train_files), ("val", val_files), ("test", test_files)]

    for split_name, file_list in splits:
        split_txt_path = os.path.join(VOC_ROOT, f"{split_name}.txt")

        with open(split_txt_path, 'w') as f_split:
            for xml_name in file_list:
                xml_path = os.path.join(ANNOTATIONS_DIR, xml_name)
                base = os.path.splitext(xml_name)[0]
                img_path = os.path.join(IMAGES_DIR, base + ".jpg")

                labels = voc_xml_to_yolo_txt(xml_path)
                if labels:
                    out_label_path = os.path.join(LABELS_DIR, base + ".txt")
                    with open(out_label_path, 'w') as f_label:
                        f_label.write("\n".join(labels))

                f_split.write(img_path + "\n")

    print("转换完成！请查看 labels 文件夹、以及 train.txt/val.txt/test.txt。")


if __name__ == "__main__":
    main()