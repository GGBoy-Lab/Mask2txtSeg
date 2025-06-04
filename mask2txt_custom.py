import os
import cv2
import numpy as np
from pathlib import Path


# 定义颜色到类别ID的映射
COLOR_TO_CLASS = {
    (255, 112, 132): 1,     # 类别 1 - KAO_YAN
    (130, 134, 139): 2,     # 类别 2 - SHUI_DAO
    (233, 122, 238): 3,     # 类别 3 - YU_MI
}


def rgb_to_class_id(rgb_array, color_map):
    """ 将 RGB 像素值映射为类别 ID """
    h, w, _ = rgb_array.shape
    class_mask = np.zeros((h, w), dtype=np.uint8)

    for color, class_id in color_map.items():
        match = np.all(rgb_array == color, axis=-1)  #
        class_mask[match] = class_id

    return class_mask


def convert_mask_to_yolo(mask_path, img_width, img_height):
    # 以彩色模式读取 RGB 掩码
    mask_rgb = cv2.imread(mask_path)
    print(mask_rgb.shape)

    # 转换为类别 ID 掩码
    mask_class = rgb_to_class_id(mask_rgb, COLOR_TO_CLASS)

    contours_dict = {}

    unique_classes = np.unique(mask_class)
    print(f"Unique classes in {mask_path}: {unique_classes}")

    for class_value in unique_classes:
        if class_value == 0:  # 跳过背景
            continue

        binary_mask = np.uint8(mask_class == class_value) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if len(contour) < 3:  # 忽略小于3个点的轮廓（如线段、点）
                continue

            # 归一化坐标
            normalized = contour.astype(np.float32).reshape(-1, 2) / np.array([img_width, img_height])

            if class_value not in contours_dict:
                contours_dict[class_value] = []
            contours_dict[class_value].append(normalized)

    print(f"contours_dict: {contours_dict}")
    return contours_dict


# 配置路径
images_dir = "E:\\Desktop\\code\\Seg\\data\\VOC\\JPEGImages"
masks_dir = "E:\\Desktop\\code\\Seg\\data\\VOC\\SegmentationClass"
output_dir = "./labels"
os.makedirs(output_dir, exist_ok=True)


# 遍历所有图像和掩码
for img_file in Path(images_dir).glob("*.*"):
    mask_path = Path(masks_dir) / (img_file.stem + ".png")
    if not mask_path.exists():
        continue

    # 获取图像尺寸
    img = cv2.imread(str(img_file))
    img_height, img_width = img.shape[:2]

    # 转换掩码
    contours_dict = convert_mask_to_yolo(str(mask_path), img_width, img_height)

    # 生成YOLO格式的txt文件，每个类别一个标签文件
    txt_path = Path(output_dir) / (img_file.stem + ".txt")
    with open(txt_path, 'w') as f:
        for class_value, contours in contours_dict.items():
            for points in contours:
                if not isinstance(points, np.ndarray) or len(points.shape) != 2 or points.shape[1] != 2:
                    continue
                coords = " ".join([f"{x:.6f} {y:.6f}" for x, y in points])
                line = f"{class_value - 1} {coords}\n"  # 类别从 0 开始
                f.write(line)
