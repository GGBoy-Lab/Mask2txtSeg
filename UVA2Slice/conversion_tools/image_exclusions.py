from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

def is_white_or_transparent_less_than_50(image_path):
    # 打开图像并转换为 RGBA 模式
    img = Image.open(image_path).convert("RGBA")
    width, height = img.size
    total_pixels = width * height

    # 使用 NumPy 加速统计
    data = np.array(img)
    alpha_channel = data[:, :, 3]

    # 统计纯白色像素数量（RGB 为 255, 255, 255 并且 Alpha 为 255）
    white_mask = np.all(data[:, :, :3] == [255, 255, 255], axis=-1) & (alpha_channel == 255)
    white_pixel_count = np.sum(white_mask)

    # 统计透明像素数量（Alpha 通道为 0）
    transparent_pixel_count = np.sum(alpha_channel == 0)

    # 计算占比
    total_percentage = (white_pixel_count + transparent_pixel_count) / total_pixels

    return total_percentage < 0.5

def process_image(filename, input_dir, output_dir):
    image_path = os.path.join(input_dir, filename)
    if is_white_or_transparent_less_than_50(image_path):
        # 将符合条件的图像复制到输出目录
        output_path = os.path.join(output_dir, filename)
        try:
            img = Image.open(image_path)
            img.save(output_path)
            print(f"Copied: {output_path}")
        except Exception as e:
            print(f"Failed to copy {image_path}: {e}")
    else:
        print(f"Skipped: {image_path}")

def filter_images(input_dir, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有需要处理的文件列表
    filenames = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # 使用线程池并发处理每个文件
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(process_image, filename, input_dir, output_dir): filename
            for filename in filenames
        }

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"An error occurred: {exc}")

if __name__ == '__main__':


    input_directory = r'F:\工作文件\目标识别与分割\label\18\emptySamplesDir'
    output_directory = r'F:\工作文件\目标识别与分割\label\18\emptySamplesDir1'

    filter_images(input_directory, output_directory)