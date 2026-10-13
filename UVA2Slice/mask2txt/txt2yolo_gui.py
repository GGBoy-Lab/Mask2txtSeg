import cv2
import numpy as np
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


class YOLOLabelViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO标签可视化工具")
        self.root.geometry("1600x900")

        self.image_folder = ""
        self.label_folder = ""
        self.current_image_idx = 0
        self.image_files = []

        # 顶部控制区
        control_frame = tk.Frame(root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Button(control_frame, text="选择图像文件夹", command=self.select_image_folder).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="选择标签文件夹", command=self.select_label_folder).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="绘制", command=self.draw_labels).pack(side=tk.LEFT, padx=5)

        self.nav_frame = tk.Frame(control_frame)
        self.nav_frame.pack(side=tk.LEFT, padx=20)
        tk.Button(self.nav_frame, text="< 上一张", command=self.prev_image).pack(side=tk.LEFT, padx=2)
        self.info_label = tk.Label(self.nav_frame, text="0/0")
        self.info_label.pack(side=tk.LEFT, padx=5)
        tk.Button(self.nav_frame, text="下一张 >", command=self.next_image).pack(side=tk.LEFT, padx=2)

        # 图像显示区
        display_frame = tk.Frame(root)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        left_frame = tk.Frame(display_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        tk.Label(left_frame, text="原图").pack(anchor=tk.W)
        self.original_canvas = tk.Canvas(left_frame, bg='gray')
        self.original_canvas.pack(fill=tk.BOTH, expand=True)

        right_frame = tk.Frame(display_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        tk.Label(right_frame, text="标注结果").pack(anchor=tk.W)
        self.result_canvas = tk.Canvas(right_frame, bg='gray')
        self.result_canvas.pack(fill=tk.BOTH, expand=True)

    def select_image_folder(self):
        folder = filedialog.askdirectory(title="选择图像文件夹")
        if folder:
            self.image_folder = folder
            self.image_files = sorted([f for f in os.listdir(folder)
                                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))])
            self.current_image_idx = 0
            messagebox.showinfo("提示", f"已选择图像文件夹，共 {len(self.image_files)} 张图片")

    def select_label_folder(self):
        folder = filedialog.askdirectory(title="选择标签文件夹")
        if folder:
            self.label_folder = folder
            messagebox.showinfo("提示", f"已选择标签文件夹: {folder}")

    def draw_labels(self):
        if not self.image_folder or not self.label_folder:
            messagebox.showwarning("警告", "请先选择图像和标签文件夹")
            return

        if not self.image_files:
            messagebox.showwarning("警告", "图像文件夹为空")
            return

        self.current_image_idx = 0
        self.show_current_image()

    def show_current_image(self):
        if not self.image_files:
            return

        image_name = self.image_files[self.current_image_idx]
        image_path = os.path.join(self.image_folder, image_name)

        label_name = Path(image_name).with_suffix('.txt').name
        label_path = os.path.join(self.label_folder, label_name)

        # 读取图像
        image_array = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            messagebox.showerror("错误", f"无法读取图像: {image_name}")
            return

        height, width = image.shape[:2]

        # 创建副本用于绘制
        result_image = image.copy()

        # 读取并绘制标签
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                data = list(map(float, line.strip().split()))
                if len(data) < 1:
                    continue

                class_id = int(data[0])
                points = np.array(data[1:], dtype=np.float32).reshape(-1, 2)

                # 转换归一化坐标 -> 实际像素坐标
                points[:, 0] *= width
                points[:, 1] *= height
                points = points.astype(np.int32)

                # 绘制多边形轮廓
                cv2.polylines(result_image, [points], isClosed=True,
                              color=(0, 255, 0), thickness=2)

                # 绘制类别标签
                label_text = f"Class {class_id}"
                cv2.putText(result_image, label_text, tuple(points[0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 转换为RGB并调整大小
        original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

        # 缩放以适应画布
        canvas_width = self.original_canvas.winfo_width() or 700
        canvas_height = self.original_canvas.winfo_height() or 800

        scale_w = canvas_width / width
        scale_h = canvas_height / height
        scale = min(scale_w, scale_h, 1.0)

        new_width = int(width * scale)
        new_height = int(height * scale)

        original_resized = cv2.resize(original_rgb, (new_width, new_height))
        result_resized = cv2.resize(result_rgb, (new_width, new_height))

        # 转换为PhotoImage
        original_pil = Image.fromarray(original_resized)
        result_pil = Image.fromarray(result_resized)

        original_photo = ImageTk.PhotoImage(original_pil)
        result_photo = ImageTk.PhotoImage(result_pil)

        # 显示在画布上
        self.original_canvas.delete("all")
        self.result_canvas.delete("all")

        x_offset_orig = (canvas_width - new_width) // 2
        y_offset_orig = (canvas_height - new_height) // 2
        self.original_canvas.create_image(x_offset_orig, y_offset_orig, anchor=tk.NW, image=original_photo)
        self.original_canvas.image = original_photo

        x_offset_result = (canvas_width - new_width) // 2
        y_offset_result = (canvas_height - new_height) // 2
        self.result_canvas.create_image(x_offset_result, y_offset_result, anchor=tk.NW, image=result_photo)
        self.result_canvas.image = result_photo

        # 更新信息标签
        self.info_label.config(text=f"{self.current_image_idx + 1}/{len(self.image_files)}")

    def prev_image(self):
        if self.current_image_idx > 0:
            self.current_image_idx -= 1
            self.show_current_image()

    def next_image(self):
        if self.current_image_idx < len(self.image_files) - 1:
            self.current_image_idx += 1
            self.show_current_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOLabelViewer(root)
    root.mainloop()
