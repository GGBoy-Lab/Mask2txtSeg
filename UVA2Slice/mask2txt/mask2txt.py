import os
import cv2
import numpy as np
from pathlib import Path


# 定义颜色到类别ID的映射
COLOR_TO_CLASS = {
    (255, 255, 255):1,     # 类别 1
    # (255, 0, 0): 2,     # 类别 2
    #  (0, 255, 0): 3,     # 类别 3
}


def rgb_to_class_id(rgb_array, color_map):
    """ 将 RGB 像素值映射为类别 ID """
    h, w, _ = rgb_array.shape
    class_mask = np.zeros((h, w), dtype=np.uint8)

    for color, class_id in color_map.items():
        match = np.all(rgb_array == color, axis=-1)  #
        class_mask[match] = class_id

    return class_mask


def convert_mask_to_yolo_from_array(mask_rgb, img_width, img_height):
    """
    输入 RGB 掩码数组，输出 YOLO 格式的轮廓字典
    """
    # 转换为类别 ID 掩码
    mask_class = rgb_to_class_id(mask_rgb, COLOR_TO_CLASS)

    contours_dict = {}

    unique_classes = np.unique(mask_class)
    print(f"Unique classes in mask: {unique_classes}")

    for class_value in unique_classes:
        if class_value == 0:
            continue

        binary_mask = np.uint8(mask_class == class_value) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if len(contour) < 3:
                continue

            normalized = contour.astype(np.float32).reshape(-1, 2) / np.array([img_width, img_height])

            if class_value not in contours_dict:
                contours_dict[class_value] = []
            contours_dict[class_value].append(normalized)

    return contours_dict



# 配置路径
images_dir = r"E:\Desktop\SmokeDataSet\image"
masks_dir = r"E:\Desktop\SmokeDataSet\label"
output_dir = r"E:\Desktop\SmokeDataSet\txt"
os.makedirs(output_dir, exist_ok=True)


# 遍历所有图像和掩码
for img_file in Path(images_dir).glob("*.*"):
    mask_path = Path(masks_dir) / (img_file.stem + ".png")
    if not mask_path.exists():
        continue

    # 获取图像并统一尺寸
    img = cv2.imread(str(img_file))
    img = cv2.resize(img, (1024, 1024))  # 新增：统一图像尺寸
    img_height, img_width = img.shape[:2]

    # 读取掩码并统一尺寸
    mask_rgb = cv2.imread(str(mask_path))
    if mask_rgb is None:
        print(f"⚠️ Failed to read mask: {mask_path}")
        continue
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
    mask_rgb = cv2.resize(mask_rgb, (1024, 1024), interpolation=cv2.INTER_NEAREST)

    # 转换掩码
    contours_dict = convert_mask_to_yolo_from_array(mask_rgb, img_width, img_height)

    # 生成 YOLO 标签文件
    txt_path = Path(output_dir) / (img_file.stem + ".txt")
    with open(txt_path, 'w') as f:
        for class_value, contours in contours_dict.items():
            for points in contours:
                if not isinstance(points, np.ndarray) or len(points.shape) != 2 or points.shape[1] != 2:
                    continue
                coords = " ".join([f"{x:.6f} {y:.6f}" for x, y in points])
                line = f"{class_value - 1} {coords}\n"
                f.write(line)
