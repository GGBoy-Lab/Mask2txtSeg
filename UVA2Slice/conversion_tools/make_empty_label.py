import glob
import os

from PIL import Image


def create_black_image(size):
    """创建单通道图片"""
    return Image.new("L", size, 0)


def get_all_images(directory, extensions=('*.jpg', '*.jpeg', '*.png')):
    """获取指定目录下所有指定格式的图片"""
    image_paths = []
    for ext in extensions:
        pattern = os.path.join(directory, ext)
        image_paths.extend(glob.glob(pattern))
    return image_paths


if __name__ == '__main__':
    input_dir = r"F:\projcct-py\remote-sensing-image-training\data\1.23\SINGLECLASSDataset\negative_img_dir\train"
    out_dir = r"F:\projcct-py\remote-sensing-image-training\data\1.23\SINGLECLASSDataset\negative_ann_dir\train"

    image_paths = get_all_images(input_dir)

    for image_path in image_paths:
        with Image.open(image_path) as img:
            size = img.size
            label_img = create_black_image(size)
            basename = os.path.basename(image_path)
            out = os.path.join(out_dir, basename)
            label_img.save(out)
