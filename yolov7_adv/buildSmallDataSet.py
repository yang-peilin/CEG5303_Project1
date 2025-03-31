import os
import xml.etree.ElementTree as ET
import shutil

# ----------------------
# 配置参数
# ----------------------
# VOC2007 数据集原始路径
voc_path = r'/home/ypl/yolov7/VOCdevkit/VOC2007'
ann_dir = os.path.join(voc_path, 'Annotations')
img_dir = os.path.join(voc_path, 'JPEGImages')
imagesets_main_dir = os.path.join(voc_path, 'ImageSets', 'Main')

# 目标类别（全部转换为小写）
target_classes = {'bicycle', 'bird', 'bus', 'car', 'cat', 'dog', 'motorbike', 'person'}

# 输出小型数据集路径：VOC2007_SMALL
output_path = r'/home/ypl/yolov7/VOCdevkit/VOC2007_SMALL'
new_ann_dir = os.path.join(output_path, 'Annotations')
new_img_dir = os.path.join(output_path, 'JPEGImages')
new_imagesets_main_dir = os.path.join(output_path, 'ImageSets', 'Main')

# 创建输出目录
os.makedirs(new_ann_dir, exist_ok=True)
os.makedirs(new_img_dir, exist_ok=True)
os.makedirs(new_imagesets_main_dir, exist_ok=True)

# ----------------------
# 筛选并复制图片和标注
# ----------------------
filtered_ids = set()  # 用来存储符合条件的图片ID

print("正在筛选符合条件的图片...")
for xml_file in os.listdir(ann_dir):
    if not xml_file.endswith('.xml'):
        continue
    xml_path = os.path.join(ann_dir, xml_file)
    try:
        tree = ET.parse(xml_path)
    except Exception as e:
        print(f"解析 {xml_file} 时出错: {e}")
        continue
    root = tree.getroot()

    # 检查该标注中是否含有目标类别
    contains_target = False
    for obj in root.findall('object'):
        cls = obj.find('name').text.lower()
        if cls in target_classes:
            contains_target = True
            break

    if contains_target:
        # 得到图片ID（XML 文件名去除 .xml）
        img_id = os.path.splitext(xml_file)[0]
        filtered_ids.add(img_id)

        # 复制 XML 文件
        shutil.copy(xml_path, new_ann_dir)

        # 复制对应的图片文件（扩展名 .jpg）
        img_file = img_id + '.jpg'
        src_img_path = os.path.join(img_dir, img_file)
        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, new_img_dir)
        else:
            print(f"警告：找不到图片 {img_file} 对应于 {xml_file}")

print(f"筛选完成，共找到 {len(filtered_ids)} 张图片。")

# ----------------------
# 处理 ImageSets/Main
# ----------------------
# 如果原 ImageSets/Main 存在，则读取并重新过滤
if os.path.exists(imagesets_main_dir):
    print("正在处理 ImageSets/Main 划分文件...")
    for file in os.listdir(imagesets_main_dir):
        if not file.endswith('.txt'):
            continue
        src_file = os.path.join(imagesets_main_dir, file)
        dst_file = os.path.join(new_imagesets_main_dir, file)

        with open(src_file, 'r') as fr, open(dst_file, 'w') as fw:
            for line in fr:
                # 每行通常是图片ID（可能后面还有其它信息，用空格分隔）
                parts = line.strip().split()
                if parts:
                    img_id = parts[0]
                    if img_id in filtered_ids:
                        fw.write(line)
    print("ImageSets/Main 文件处理完成。")
else:
    print("原始 ImageSets/Main 目录不存在，跳过此步。")

print("VOC2007_SMALL 数据集构建完成！")
