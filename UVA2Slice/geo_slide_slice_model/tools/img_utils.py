import numpy as np


def merge_images(images):
    """
    合并相同尺寸的图片数组到自适应画布

    参数：
    images : list[np.ndarray] 需满足：
             - 所有数组维度相同
             - 形状为(H, W, 3)
             - uint8数据类型

    返回：
    np.ndarray 合并后的图片数组，保持原始图片尺寸
    """
    # 验证输入一致性
    assert len(images) > 0, "输入图片列表不能为空"
    base_shape = images[0].shape
    for i, img in enumerate(images):
        if img.shape != base_shape:
            raise ValueError(f"图片{i}形状不匹配，期望{base_shape}，实际{img.shape}")
        if img.dtype != np.uint8:
            raise TypeError(f"图片{i}数据类型需为uint8")

    # 创建初始画布（纯黑背景）
    merged = np.zeros_like(images[0])

    # 多图叠加处理
    for img in images:
        # 生成有效像素掩膜（非纯黑区域）
        non_black = np.any(img != [0, 0, 0], axis=-1)
        # 覆盖写入策略
        merged[non_black] = img[non_black]

    return merged