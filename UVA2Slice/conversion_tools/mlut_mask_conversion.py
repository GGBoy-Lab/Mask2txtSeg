import os
import cv2
import numpy as np


def convert_mask_to_class_index_vectorized(mask, palette):
    """
    使用NumPy向量化操作将四通道彩色掩码图像转换为单通道类别索引图像。

    :param mask: 四通道彩色掩码图像 (H, W, 4)
    :param palette: 调色板，包含颜色列表和对应的类别索引
    :return: 单通道类别索引图像 (H, W)
    """
    # 创建颜色到类别的映射字典
    color_to_class = {tuple(color): idx for idx, color in enumerate(palette)}

    # 提取前三个通道的颜色，并转换为RGB格式（假设输入是BGR）
    colors = mask[:, :, :3].reshape(-1, 3)  # 展平成 (H*W, 3) 形状
    colors_rgb = colors[:, ::-1]  # 将BGR转换为RGB

    # 构建查找表
    lookup_table = np.array([color_to_class.get(tuple(color), 0) for color in colors_rgb])

    # 重塑回原始图像形状
    class_map = lookup_table.reshape(mask.shape[:2])

    return class_map


mask1 = cv2.imread(r'F:\projcct-py\remote-sensing-image-training\data\DFL\20250312\label\209263_152112.png')

METAINFO = {
    'classes': ('background', '烤烟', '马铃薯', '水稻', '玉米', '高粱'),
    'palette': [[0, 0, 0], [132, 112, 255], [0, 0, 255], [139, 134, 130], [238, 122, 233], [0, 255, 255]]
}

# 转换为单通道类别索引
class_index = convert_mask_to_class_index_vectorized(mask1, METAINFO['palette'])

print(class_index.shape)  # 输出应为 (1024, 1024)

# 可选：保存结果以验证
cv2.imwrite('class_index.png', class_index.astype(np.uint8))

