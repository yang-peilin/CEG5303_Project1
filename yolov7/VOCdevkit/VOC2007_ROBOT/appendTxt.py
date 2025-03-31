import os

# VOC_ROOT 指定 VOC2007_ROBOT 的根目录（根据实际路径修改）
VOC_ROOT = "/root/autodl-tmp/ypl/yolov7/VOCdevkit/VOC2007_ROBOT"
imagesets_main_dir = os.path.join(VOC_ROOT, "ImageSets", "Main")

# 如果目录不存在则创建
os.makedirs(imagesets_main_dir, exist_ok=True)

# 根据要求生成编号（6位格式，前面补0）
train_ids = [f"{i:06d}" for i in range(10000, 10000 + 140)]   # 010000 ~ 010049
val_ids   = [f"{i:06d}" for i in range(10140, 10140 + 40)]     # 010050 ~ 010064
test_ids  = [f"{i:06d}" for i in range(10180, 10180 + 20)]       # 010065 ~ 010084

# 文件路径
train_file = os.path.join(imagesets_main_dir, "train.txt")
val_file = os.path.join(imagesets_main_dir, "val.txt")
test_file = os.path.join(imagesets_main_dir, "test.txt")

def append_ids(filepath, ids):
    with open(filepath, "a") as f:
        for id in ids:
            f.write(id + "\n")
    print(f"已在 {filepath} 追加 {len(ids)} 行内容。")

append_ids(train_file, train_ids)
append_ids(val_file, val_ids)
append_ids(test_file, test_ids)

print("追加操作完成！")

