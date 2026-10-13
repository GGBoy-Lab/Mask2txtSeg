import os

import numpy as np
from PIL import Image


def convert_mask_to_one(folder_path):
    label_dir = os.path.join(folder_path, "label")
    label_dir_channel = os.path.join(folder_path, "label_channel")
    os.makedirs(label_dir_channel, exist_ok=True)

    # 遍历指定文件夹中的所有文件
    for filename in os.listdir(label_dir):
        # 构建文件的完整路径
        file_path = os.path.join(label_dir, filename)

        # 检查文件是否是图像（这里简单检查文件扩展名，可以根据需要扩展）
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # 打开图像文件
            with Image.open(file_path) as img:
                out_dir = os.path.join(label_dir_channel, filename)

                # 确保图像是灰度图（如果不是，则转换为灰度图）
                if img.mode != 'L':
                    img = img.convert('L')

                # 将图像数据转换为可修改的numpy数组
                img_data = np.array(img)

                # 将像素值为255的地方改为1
                img_data[img_data != 0] = 1

                # 将修改后的numpy数组转换回图像对象
                modified_img = Image.fromarray(img_data)

                modified_img.save(out_dir)
                print(f"Converted {filename}")

def convert_mask(folder_path):
    label_dir = os.path.join(folder_path, "label")

    # 遍历指定文件夹中的所有文件
    for filename in os.listdir(label_dir):
        # 构建文件的完整路径
        file_path = os.path.join(label_dir, filename)

        # 检查文件是否是图像（这里简单检查文件扩展名，可以根据需要扩展）
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # 打开图像文件
            with Image.open(file_path) as img:
                # 确保图像是灰度图（如果不是，则转换为灰度图）
                if img.mode != 'L':
                    img = img.convert('L')

                # 将图像数据转换为可修改的numpy数组
                img_data = np.array(img)

                # 将像素值为255的地方改为1
                img_data[img_data != 255] = 0

                # 将修改后的numpy数组转换回图像对象
                modified_img = Image.fromarray(img_data)

                # 保存修改后的图像（覆盖原文件）
                modified_img.save(file_path)
                print(f"Converted {filename}")
