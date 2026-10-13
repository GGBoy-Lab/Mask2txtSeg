import shutil
from asyncio import as_completed
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
import os


def is_white_or_transparent_less_than_50(image_path):
    # 打开图像
    img = Image.open(image_path).convert("RGBA")
    width, height = img.size
    total_pixels = width * height

    # 统计纯白色像素数量
    white_pixel_count = 0
    for x in range(width):
        for y in range(height):
            pixel = img.getpixel((x, y))
            if pixel == (255, 255, 255, 255):  # 纯白色
                white_pixel_count += 1

    # 统计透明像素数量
    transparent_pixel_count = 0
    for x in range(width):
        for y in range(height):
            pixel = img.getpixel((x, y))
            if pixel[3] == 0:  # 透明像素（Alpha 通道为0）
                transparent_pixel_count += 1

    # 计算占比
    white_percentage = white_pixel_count / total_pixels
    transparent_percentage = transparent_pixel_count / total_pixels
    total_percentage = (white_pixel_count + transparent_pixel_count) / total_pixels

    # 检查条件
    if white_percentage < 0.5 and transparent_percentage < 0.5 and total_percentage < 0.5:
        return True
    else:
        return False


def process_image(filename, img_dir, ann_dir, sample_dir, label_dir):
    image_path = os.path.join(img_dir, filename)
    if  is_white_or_transparent_less_than_50(image_path):
        print(f"Copied: {filename}")
        # img_path = os.path.join(sample_dir, filename)
        # new_filename = filename.replace(".tif", ".png")
        #
        # ann_path = os.path.join(label_dir, new_filename)
        # shutil.copyfile(os.path.join(img_dir, filename), img_path)
        # shutil.copyfile(os.path.join(ann_dir, new_filename), ann_path)
        # print(f"Copied: {img_path}")
    else:
        new_filename = filename.replace(".tif", ".png")
        img = os.path.join(img_dir, filename)
        ann = os.path.join(ann_dir, new_filename)
        os.remove(img)
        os.remove(ann)
        print(f"Remove: {image_path}")


def filter_images(input_dir, output_dir):
    img_dir = os.path.join(input_dir, "images")
    ann_dir = os.path.join(input_dir, "labels")

    label_dir = os.path.join(output_dir, "label")
    sample_dir = os.path.join(output_dir, "sampleDir")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # 获取所有图片文件
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]

    # 使用 ThreadPoolExecutor 进行多线程处理
    with ThreadPoolExecutor(max_workers=238) as executor:
        future_to_filename = {
            executor.submit(process_image, filename, img_dir, ann_dir, sample_dir, label_dir): filename for filename in
            image_files}
        for future in as_completed(future_to_filename):
            filename = future_to_filename[future]
            try:
                future.result()  # 这将重新抛出异常，如果有的话
            except Exception as exc:
                print(f"{filename} generated an exception: {exc}")


if __name__ == '__main__':
    # 路径
    input_directory = r'E:\label_data\out'
    output_directory = r'E:\20250324\remote-sensing-image-training\data\KAO_YAN\20250325'

    filter_images(input_directory, output_directory)
