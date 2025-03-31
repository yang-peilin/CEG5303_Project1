import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.datasets import letterbox

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载YOLOv7模型
weights_path = "/root/autodl-tmp/ypl/yolov7_adv/runs/train/yolov7_output_small/weights/best.pt"
model = attempt_load(weights_path, map_location=device).eval().to(device)

# 禁用 in-place 操作
for module in model.modules():
    if isinstance(module, (nn.SiLU, nn.ReLU, nn.LeakyReLU)):
        module.inplace = False
    if hasattr(module, 'inplace'):
        module.inplace = False
    if module.__class__.__name__ in ['Detect', 'IDetect']:
        module.export = True

def preprocess_image(image_path, target_size=640):
    """返回预处理后的tensor及填充参数"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 使用letterbox对图像进行缩放和填充
    img_padded, _, (dw, dh) = letterbox(img, new_shape=target_size, auto=False)
    img_tensor = torch.from_numpy(img_padded.transpose(2, 0, 1)).to(device).float() / 255.0
    return img_tensor, (dw, dh), img.shape[:2]  # 返回填充参数和原图尺寸

# # FGSM攻击函数
# def fgsm_attack(model, orig_image, epsilon=5):
#     """
#     orig_image: 经过letterbox预处理的tensor [C,H,W]
#     返回: 对抗样本tensor [C,H,W]
#     """
#     image = orig_image.clone().unsqueeze(0)  # 添加batch维度 [1,C,H,W]
#     image.requires_grad = True
#
#     # 前向传播
#     with torch.enable_grad():
#         raw_pred = model(image)[0]  # 获取模型的预测结果
#         pred = non_max_suppression(raw_pred, conf_thres=0.25, iou_thres=0.45)
#
#         if len(pred[0]) == 0:
#             return orig_image  # 无检测框时返回原图
#
#         # 计算梯度（针对最高置信度类别）
#         best_box = pred[0][0]
#         cls_idx = int(best_box[5].item())
#         scores = raw_pred[:, 4]
#         best_idx = torch.argmax(scores)
#         class_logit = raw_pred[best_idx, 5 + cls_idx]
#
#         loss = -class_logit  # 目标降低该类别置信度
#         model.zero_grad()
#         loss.backward()
#
#     # 生成对抗样本
#     grad_sign = image.grad.data.sign()
#     adv_image = image + epsilon * grad_sign
#     adv_image = torch.clamp(adv_image, 0, 1).detach().squeeze(0)  # 移除batch维度
#
#     return adv_image  # [C,H,W]

def ifgsm_attack(model, orig_image, epsilon=0.01, iterations=10):
    """
    I-FGSM: Iterative Fast Gradient Sign Method
    orig_image: 经过letterbox预处理的tensor [C,H,W]
    epsilon: 每次迭代的扰动大小
    iterations: 迭代次数
    返回: 对抗样本tensor [C,H,W]
    """
    image = orig_image.clone().unsqueeze(0)  # 添加batch维度 [1,C,H,W]
    image.requires_grad = True

    for _ in range(iterations):
        with torch.enable_grad():
            raw_pred = model(image)[0]
            pred = non_max_suppression(raw_pred, conf_thres=0.25, iou_thres=0.45)

            if len(pred[0]) == 0:
                return orig_image  # 无检测框时返回原图

            best_box = pred[0][0]
            cls_idx = int(best_box[5].item())
            scores = raw_pred[:, 4]
            best_idx = torch.argmax(scores)
            class_logit = raw_pred[best_idx, 5 + cls_idx]

            loss = -class_logit
            model.zero_grad()
            loss.backward()

        grad_sign = image.grad.data.sign()
        image = image + epsilon * grad_sign  # 添加扰动
        image = torch.clamp(image, 0, 1)  # 确保在 [0, 1] 范围内

    return image.squeeze(0)  # 返回对抗样本


# 创建输出目录
orig_dir = "/root/autodl-tmp/ypl/yolov7_adv/VOCdevkit/VOC2007_SMALL/JPEGImages"
adv_dir = "/root/autodl-tmp/ypl/yolov7_adv/VOCdevkit/VOC2007_SMALL_ADV/JPEGImages"
os.makedirs(adv_dir, exist_ok=True)

# 获取原始数据集中的所有图像文件
img_list = [f for f in os.listdir(orig_dir) if f.lower().endswith(".jpg")]

# 批量生成对抗样本
for idx, img_file in enumerate(img_list):
    img_path = os.path.join(orig_dir, img_file)
    try:
        # 预处理（获取填充参数）
        orig_tensor, (dw, dh), (h_orig, w_orig) = preprocess_image(img_path)

        # 生成对抗样本
        # adv_tensor = fgsm_attack(model, orig_tensor, epsilon=0.2)
        adv_tensor = ifgsm_attack(model, orig_tensor)

        # 转换并裁剪填充区域
        adv_np = adv_tensor.permute(1, 2, 0).cpu().numpy() * 255
        adv_np = adv_np.astype(np.uint8)

        # 强制将 dh 和 dw 转为整数
        dh = int(dh)
        dw = int(dw)

        # 去除填充：有效区域为 [dh : height-dh, dw : width-dw]
        unpad_h = 640 - 2 * dh  # letterbox后的总高度为640
        unpad_w = 640 - 2 * dw  # letterbox后的总宽度为640

        adv_unpadded = adv_np[dh:dh + unpad_h, dw:dw + unpad_w]

        # 恢复原图尺寸
        adv_final = cv2.resize(adv_unpadded, (w_orig, h_orig))

        # 保存对抗样本
        cv2.imwrite(os.path.join(adv_dir, img_file),
                    cv2.cvtColor(adv_final, cv2.COLOR_RGB2BGR))

    except Exception as e:
        print(f"Error processing {img_file}: {str(e)}")
        continue

    # 进度打印
    if (idx + 1) % 500 == 0:
        print(f"Processed {idx + 1}/{len(img_list)} images")

print("对抗样本生成完成！")
