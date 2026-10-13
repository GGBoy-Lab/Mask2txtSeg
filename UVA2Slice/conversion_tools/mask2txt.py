# yolo 标签制作by mask
import concurrent
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import os

from tqdm import tqdm


# new_metainfo = dict(
#     classes=('YM','GL','SD','DP','SG','DL', 'JZ','Others'),
#     palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70],
#             [102, 102, 156], [190, 153, 153], [153, 153, 153],
#           [250, 170,30], [220, 220, 0]])


def task(mask_name, new_metainfo, mask_path,out):
    mask = cv_imread(os.path.join(mask_path, mask_name))

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    # 初始化mask为0，int8类型
    b_img = np.zeros(mask.shape[:2], dtype=np.uint8)

    height, width = mask.shape[:2]

    # 遍历调色板
    for idx, color in enumerate(new_metainfo["palette"]):
        # 生成颜色掩膜：找到所有匹配的像素
        color_mask = np.all(mask == color, axis=-1)
        # 在mask中设置对应的索引+1
        b_img[color_mask] = idx + 1

    output_txt_path = os.path.join(out, mask_name.replace("png", "txt"))
    with open(output_txt_path, 'w') as f:
        # 提取轮廓
        for label in range(len(new_metainfo["palette"])):
            mask_label = np.uint8(b_img == label + 1) * 255
            contours, _ = cv2.findContours(mask_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # print(f"标签 {label + 1} 找到 {len(contours)} 个轮廓")
            # 遍历轮廓
            for cnt in contours:
                coords = []
                ll = len(cnt)
                if ll < 3:
                    continue
                for point in cnt:
                    x, y = point[0]
                    coords.append(x / height)
                    coords.append(y / width)

                # 写入 txt 文件
                line = str(label) + " " + " ".join(f"{x:.6f}" for x in coords) + "\n"
                f.write(line)


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def mask2txt(mask_path, out_path, classes, palette):
    new_metainfo = dict(
        classes=classes,
        palette=palette
    )
    os.makedirs(out_path, exist_ok=True)
    mask_lists = os.listdir(mask_path)
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        results = [executor.submit(task, mask_name, new_metainfo, mask_path,out_path) for mask_name in tqdm(mask_lists)]
        for result in concurrent.futures.as_completed(results):
            result.result()

# if __name__ == '__main__':
#     new_metainfo = dict(
#         classes=("水稻", "玉米", "烤烟", "高粱"),
#         palette=[[255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]])
#
#     mask_path = r'E:\训练样本\4分类农作物-水稻、玉米、烤烟、高粱\新增-高粱\2048\labels1'
#     out = r"E:\训练样本\4分类农作物-水稻、玉米、烤烟、高粱\新增-高粱\2048\txt"
#     mask_lists = os.listdir(mask_path)
#
#     os.makedirs(out, exist_ok=True)
