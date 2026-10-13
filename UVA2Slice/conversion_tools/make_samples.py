import os
import shutil
import random


def mkdir_path(out_data_dir):
    # 数据集路径
    dataset_dir = os.path.join(out_data_dir, 'SINGLECLASSDataset')
    # 标注集路径
    ann_dir = os.path.join(dataset_dir, 'labels')
    ann_train_dir = os.path.join(ann_dir, 'train')
    ann_val_dir = os.path.join(ann_dir, 'val')
    # 图片集路径
    img_dir = os.path.join(dataset_dir, 'images')
    img_train_dir = os.path.join(img_dir, 'train')
    img_val_dir = os.path.join(img_dir, 'val')

    os.makedirs(ann_train_dir, exist_ok=True)
    os.makedirs(ann_val_dir, exist_ok=True)
    os.makedirs(img_train_dir, exist_ok=True)
    os.makedirs(img_val_dir, exist_ok=True)

    return ann_train_dir, ann_val_dir, img_train_dir, img_val_dir


def split_dataset(img_dir, mask_dir, output_img_train_dir, output_img_test_dir, output_mask_train_dir,
                  output_mask_test_dir, test_ratio=0.2):
    # 确保输出目录存在
    for dir_path in [output_img_train_dir, output_img_test_dir, output_mask_train_dir, output_mask_test_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 获取所有图像文件名（假设图像和mask文件名一致）
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]

    # 随机打乱文件列表
    random.shuffle(image_files)

    # 计算分割点
    split_index = int(len(image_files) * (1 - test_ratio))

    # 分割成训练集和测试集
    train_images = image_files[:split_index]
    test_images = image_files[split_index:]

    def copy_files(file_list, src_img_dir, src_mask_dir, dst_img_dir, dst_mask_dir):
        for filename in file_list:
            img_src = os.path.join(src_img_dir, filename)
            new_filename = filename.replace(".tif", ".txt")

            mask_src = os.path.join(src_mask_dir, new_filename)

            img_dst = os.path.join(dst_img_dir, filename)
            mask_dst = os.path.join(dst_mask_dir, new_filename)
            if not os.path.exists(img_src):
                return
            if not os.path.exists(mask_src):
                return
            try:
                shutil.move(img_src, img_dst)
                shutil.move(mask_src, mask_dst)
                print(f"Copied: {filename}")
            except FileNotFoundError as e:
                print(f"Warning: File not found for {filename}: {e}")

    # 复制训练集文件到各自的目录
    copy_files(train_images, img_dir, mask_dir, output_img_train_dir, output_mask_train_dir)

    # 复制测试集文件到各自的目录
    copy_files(test_images, img_dir, mask_dir, output_img_test_dir, output_mask_test_dir)


def mask_dataset(out_data_dir, test_ratio=0.2):
    # 标注集路径
    mask_dir = os.path.join(out_data_dir, 'txt')
    # 图片路径
    img_dir = os.path.join(out_data_dir, 'images1')

    ann_train_dir, ann_val_dir, img_train_dir, img_val_dir = mkdir_path(out_data_dir)

    split_dataset(
        img_dir,
        mask_dir,
        img_train_dir,
        img_val_dir,
        ann_train_dir,
        ann_val_dir,
        test_ratio=test_ratio
    )


def create_black_image(image_path, output_dir):
    # 打开原始图像以获取其尺寸
    from PIL import Image
    with Image.open(image_path) as img:
        size = img.size  # 获取宽度和高度

    # 创建一个纯黑色的单通道 (L 模式) 图像
    black_img = Image.new('L', size, 0)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 构建输出路径并保存图像
    output_path = os.path.join(output_dir, os.path.basename(image_path).replace(".tif",".png"))
    black_img.save(output_path)


def split_and_extract_images(src_img_dir, dst_img_dir_2_per_group, dst_img_dir_3rd, dst_black_img_dir_2_per_group,
                             dst_black_img_dir_3rd, num_groups=10, extract_per_group_initial=2):
    # 确保所有输出目录存在
    for dir_path in [dst_img_dir_2_per_group, dst_img_dir_3rd, dst_black_img_dir_2_per_group, dst_black_img_dir_3rd]:
        os.makedirs(dir_path, exist_ok=True)

    # 获取所有图像文件名（假设图像文件格式为常见的几种）
    image_files = [f for f in os.listdir(src_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]

    # 如果图片数量少于需要的数量，则抛出异常
    if len(image_files) < num_groups * (extract_per_group_initial + 1):
        raise ValueError("图片数量不足以满足分组和抽取的要求")

    # 随机打乱文件列表
    random.shuffle(image_files)

    # 分割成指定数量的组
    group_size = len(image_files) // num_groups
    groups = [image_files[i:i + group_size] for i in range(0, len(image_files), group_size)]

    # 如果有剩余的图片，分配给前面的组
    remainder = len(image_files) % num_groups
    for i in range(remainder):
        groups[i].append(image_files[-(i + 1)])

    extracted_images_2_per_group = []

    # 从每个组中随机抽取指定数量的图片，并生成对应的纯黑图片
    for idx, group in enumerate(groups):
        selected_images_2 = random.sample(group, min(extract_per_group_initial, len(group)))
        extracted_images_2_per_group.extend(selected_images_2)

        # 将选中的前两张图片复制到目标文件夹，并生成纯黑图片
        for img in selected_images_2:
            src_path = os.path.join(src_img_dir, img)
            dst_path = os.path.join(dst_img_dir_2_per_group, img)
            shutil.move(src_path, dst_path)
            print(f"Copied to {dst_img_dir_2_per_group}: {img}")
            create_black_image(src_path, dst_black_img_dir_2_per_group)

    # 从每组中抽取第三张图片，确保不重复，并生成对应的纯黑图片
    extracted_images_3rd = []
    for idx, group in enumerate(groups):
        # 剔除已经被抽取的图片
        available_images = list(set(group) - set(extracted_images_2_per_group))

        if available_images:  # 确保组内还有可用图片
            selected_image_3 = random.choice(available_images)
            extracted_images_3rd.append(selected_image_3)

            # 将选中的第三张图片复制到目标文件夹，并生成纯黑图片
            src_path = os.path.join(src_img_dir, selected_image_3)
            dst_path = os.path.join(dst_img_dir_3rd, selected_image_3)
            shutil.move(src_path, dst_path)
            print(f"Copied to {dst_img_dir_3rd}: {selected_image_3}")
            create_black_image(src_path, dst_black_img_dir_3rd)
        else:
            print(f"Warning: No available images left in group {idx + 1} for third extraction.")

    return extracted_images_2_per_group, extracted_images_3rd




if __name__ == '__main__':
    # 获取当前文件所在的目录
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    out_data_dir = r"E:\训练样本\L0四分类-优化后\sample_512"
    # 制作训练集和测试集
    mask_dataset(out_data_dir, 0.1)


