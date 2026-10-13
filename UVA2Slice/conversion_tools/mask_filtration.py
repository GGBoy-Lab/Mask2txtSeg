import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import shutil


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def task(mask_folder, mask_name, image_folder, threshold, output_image_folder, output_mask_folder):
    if mask_name.lower().endswith('.png'):
        mask_path = os.path.join(mask_folder, mask_name)
        image_name = os.path.splitext(mask_name)[0] + '.tif'
        image_path = os.path.join(image_folder, image_name)

        # 检查原图是否存在
        if not os.path.exists(image_path):
            print(f"未找到对应原图: {image_name}")
            return

        # 读取图像
        image = cv_imread(mask_path)
        if image is None:
            print(f"无法读取图像: {image_name}")
            return

        # 计算背景像素数量
        b = np.array([0, 0, 0]).astype(np.uint8)
        background_pixels = np.sum(np.all(image == b, axis=-1))
        total_pixels = image.shape[0] * image.shape[1]
        background_ratio = background_pixels / total_pixels

        # 判断是否移动
        if background_ratio < threshold:
            shutil.move(image_path, os.path.join(output_image_folder, image_name))
            shutil.move(mask_path, os.path.join(output_mask_folder, mask_name))
            print(f"{image_name} 及其 mask 已移动，背景占比: {background_ratio:.4f}")
        else:
            print(f"{image_name} 未移动，背景占比: {background_ratio:.4f}")
    return


def filter_samples_by_background(image_folder, mask_folder, output_image_folder, output_mask_folder, threshold):
    """
    过滤样本：根据背景占比阈值筛选图像，并移动 .tif 原图 和对应的 .png mask。

    :param image_folder: 原图文件夹路径 (.tif)
    :param mask_folder: mask 文件夹路径 (.png)
    :param output_image_folder: 输出原图路径
    :param output_mask_folder: 输出 mask 路径
    :param threshold: 背景占比阈值 (0~1)
    """
    # 确保输出目录存在
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)

    ll = os.listdir(mask_folder)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = [executor.submit(task, mask_folder, mask_name, image_folder, threshold, output_image_folder,
                                   output_mask_folder) for
                   mask_name in
                   os.listdir(mask_folder)]
        for result in results:
            _ = result.result()


if __name__ == '__main__':
    # 示例调用
    image_folder = r'E:\训练样本\L0四分类-优化后\sample_1024\images'
    mask_folder = r'E:\训练样本\L0四分类-优化后\sample_1024\labels'
    output_image_folder = r'E:\训练样本\L0四分类-优化后\sample_1024\images1'
    output_mask_folder = r'E:\训练样本\L0四分类-优化后\sample_1024\labels1'
    threshold = 0.95

    filter_samples_by_background(image_folder, mask_folder, output_image_folder, output_mask_folder, threshold)
