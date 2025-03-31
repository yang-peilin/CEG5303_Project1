import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms

# YOLOv7 相关模块（确保在 YOLOv7 根目录或已设置 PYTHONPATH）
from models.experimental import attempt_load
from utils.general import non_max_suppression

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1) 加载模型并禁用 in-place 优化
weights_path = "/root/autodl-tmp/ypl/yolov7/runs/train/yolov7_output_small/weights/best.pt"
print(f"Loading YOLOv7 model from {weights_path} ...")
model = attempt_load(weights_path, map_location=device)  # 不用 fuse=False
model.to(device).eval()

# 禁用所有 inplace
for module in model.modules():
    if isinstance(module, nn.SiLU):
        module.inplace = False
    if hasattr(module, 'inplace'):
        module.inplace = False
    if module.__class__.__name__ in ['Detect', 'IDetect']:
        module.export = True


# 2) 定义最简单的 FGSM 攻击函数
def fgsm_attack(model, image, epsilon=0.005):
    """
    对单张图像执行最简单的 FGSM 攻击：
       x_adv = x + epsilon * sign(grad_x)
    """
    image = image.unsqueeze(0).to(device)
    image.requires_grad = True

    raw_pred = model(image)[0]  # [1, num_boxes, 85]
    pred = non_max_suppression(raw_pred, conf_thres=0.25, iou_thres=0.45)
    if len(pred[0]) == 0:
        # 没有检测框，就返回原图像
        image.requires_grad = False
        return image.detach()

    # 找最高置信度框
    best_box = pred[0][0]  # shape [x1, y1, x2, y2, conf, cls]
    raw_pred = raw_pred[0]  # shape [num_boxes, 85]
    scores = raw_pred[:, 4]
    best_idx = torch.argmax(scores)
    cls_idx = int(best_box[5].item())
    class_logit = raw_pred[best_idx, 5 + cls_idx]

    # 以 -class_logit 作为损失，让该类别置信度下降
    loss = -class_logit
    model.zero_grad()
    loss.backward()

    # 计算符号梯度并加扰动
    grad_sign = image.grad.data.sign()
    adv_image = image + epsilon * grad_sign
    adv_image = torch.clamp(adv_image, 0, 1).detach()
    return adv_image


# 3) 遍历所有原图像
orig_dir = "/root/autodl-tmp/ypl/yolov7/VOCdevkit/VOC2007_SMALL/JPEGImages"
img_list = [f for f in os.listdir(orig_dir) if f.lower().endswith(".jpg")]
print(f"Found {len(img_list)} total images in {orig_dir}")

# 4) 创建对抗图像输出文件夹
adv_dir = "/root/autodl-tmp/ypl/yolov7/VOCdevkit/VOC2007_SMALL_ADV/JPEGImages"
os.makedirs(adv_dir, exist_ok=True)

# 5) 定义图像预处理（640×640, ToTensor）
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

# 6) 批量生成对抗样本
for idx, img_file in enumerate(img_list):
    img_path = os.path.join(orig_dir, img_file)

    # 打开图像
    pil_img = Image.open(img_path).convert('RGB')
    img_tensor = transform(pil_img)

    # FGSM 攻击
    adv_tensor = fgsm_attack(model, img_tensor, epsilon=0.005)
    adv_tensor = adv_tensor.squeeze(0)  # [C,H,W]

    # 转回 numpy
    adv_img = adv_tensor.permute(1, 2, 0).cpu().numpy() * 255
    adv_img = adv_img.astype(np.uint8)

    out_path = os.path.join(adv_dir, img_file)
    cv2.imwrite(out_path, cv2.cvtColor(adv_img, cv2.COLOR_RGB2BGR))

    # 打印进度
    if idx % 50 == 0:
        print(f"Processed {idx} images...")

print(f"离线对抗样本生成完成！输出到: {adv_dir}")
