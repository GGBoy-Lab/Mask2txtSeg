# 🧾 README - YOLO Segmentation Label Converter

该脚本用于将 **RGB 彩色分割掩码图像** 转换为 **YOLOv8 兼容的 `.txt` 标签文件**。它会从掩码中提取轮廓，并按照 YOLO 实例分割任务所需的多边形归一化格式进行写入。

---

## 📁 项目概览

- **用途**：将 RGB 分割掩码转换为 YOLO 标签格式。
- **输入**：
  - RGB 彩色分割掩码图像（`.png`）
  - 对应的原始图像文件
- **输出**：
  - 包含归一化多边形坐标的 `.txt` 标签文件
- **支持**：通过自定义颜色到类别的映射支持多类别目标

---

## ⚙️ 环境要求

安装以下依赖：

```bash
pip install opencv-python numpy pathlib
