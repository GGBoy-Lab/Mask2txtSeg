import cv2
import os
import numpy as np
from pathlib import Path


def convert_png2seg(input_dir, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有文件
    files = os.listdir(input_dir)

    for file in files:
        # 只处理图像文件
        if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue

        # 使用 Path 正确处理路径
        input_path = Path(input_dir) / file
        output_path = Path(output_dir) / file

        print(f"正在处理：{input_path}")

        # 读取图像（支持中文路径）
        img = cv2.imdecode(np.fromfile(str(input_path), dtype=np.uint8), cv2.IMREAD_COLOR)

        # 检查图像是否成功加载
        if img is None:
            print(f"警告：无法读取图像 {file}，跳过...")
            continue

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"  像素范围: {img_gray.min()} ~ {img_gray.max()}, 非零像素数: {np.count_nonzero(img_gray)}")

        # 前景像素值变为 255，背景像素值变为 0
        retval, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV)  # 阈值处理

        # 保存图像（支持中文路径）
        ext = Path(file).suffix
        success, buf = cv2.imencode(ext, binary)
        if success:
            buf.tofile(str(output_path))
            print(f"✓ 已保存：{output_path}")
        else:
            print(f"警告：保存失败 {file}，跳过...")



if __name__ == "__main__":
    # 输入 mask 数据路径
    input_dir = r"E:\Desktop\SmokeDataSet\label"

    # binary后的图像输出路径
    output_dir = r"E:\Desktop\SmokeDataSet\masks_binary"
    convert_png2seg(input_dir, output_dir)

