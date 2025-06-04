import cv2
import numpy as np


def read_txt_labels(txt_file, image_file=None):
    """
    从 txt 标注文件中读取标签
    :param txt_file: txt 标注文件路径
    :param image_file: 对应图像文件路径（可选）
    :return: 若提供 image_file 返回 (labels, image)，否则返回 labels
    """
    with open(txt_file, "r") as f:
        labels = []
        for line in f.readlines():
            label_data = line.strip().split(" ")
            class_id = int(label_data[0])
            coordinates = [float(x) for x in label_data[1:]]
            labels.append([class_id, coordinates])

    if image_file is not None:
        image = cv2.imread(image_file)
        return labels, image

    return labels

def draw_segmentation_labels(image, labels, class_names=None, colors=None):
    """
    将多点分割标签绘制在图像上
    :param image: 输入图像 (numpy array)
    :param labels: 从 read_txt_labels 返回的标签列表
    :param class_names: 类别名称列表（可选）
    :param colors: 每个类别的颜色列表（可选）
    :return: 绘制后的图像
    """
    h, w = image.shape[:2]

    for label in labels:
        class_id, coordinates = label
        # 将归一化坐标转换为像素坐标
        points = []
        for i in range(0, len(coordinates), 2):
            x = int(coordinates[i] * w)
            y = int(coordinates[i + 1] * h)
            points.append([x, y])
        pts = np.array(points, np.int32).reshape((-1, 1, 2))

        # 设置颜色
        if colors is not None:
            color = colors[class_id]
        else:
            color = (0, 255, 0)  # 默认绿色

        # 绘制多边形
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)

        # 添加类别标签文本
        label_text = class_names[class_id] if class_names else f"Class {class_id}"
        cv2.putText(image, label_text, tuple(pts[0][0]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return image


# 示例
class_names = ['1', '2', '3']
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR 格式

labels, image = read_txt_labels("./labels/TEST.txt",
                                "E:\\Desktop\\code\\Seg\\data\\VOC\\JPEGImages\\TEST.jpg")
image_with_labels = draw_segmentation_labels(image, labels, class_names, colors)

cv2.imshow("Image with Segmentation Labels", image_with_labels)
cv2.waitKey(0)
cv2.destroyAllWindows()

