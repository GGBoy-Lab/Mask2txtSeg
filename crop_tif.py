import os
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed



class BatchTifCrop:
    """
    批量TIF图像滑动窗口裁剪工具
    """

    def __init__(self, crop_size=1024, overlap=0.3, stride=None, keep_partial=False,
                 num_threads=1, compression='deflate', predictor=2):
        """
        初始化裁剪器

        Args:
            crop_size (int): 裁剪窗口大小，默认1024
            overlap (float): 窗口重叠比例，默认0.3
            stride (int): 窗口步长，如果指定则覆盖overlap参数
            keep_partial (bool): 是否保留不完整的图块，默认False表示只保留完整图块
            num_threads (int): 处理线程数，默认1
            compression (str): 输出文件压缩方式，默认'deflate'
            predictor (int): 压缩预测器，默认2
        """
        self.crop_size = crop_size
        self.overlap = overlap
        self.stride = stride
        self.keep_partial = keep_partial
        self.num_threads = num_threads
        self.compression = compression
        self.predictor = predictor

    def crop_single_tif(self, tif_path, output_dir):
        """
        对单个TIF文件进行滑动窗口裁剪

        Args:
            tif_path (str): 输入TIF文件路径
            output_dir (str): 输出目录路径
        """
        # 创建输出目录
        tif_name = Path(tif_path).stem
        os.makedirs(output_dir, exist_ok=True)

        # 使用rasterio打开影像
        with rasterio.open(tif_path) as src:
            height = src.height
            width = src.width
            count = src.count
            dtype = src.dtypes[0]
            crs = src.crs
            transform = src.transform

            # 裁剪参数计算
            crop_size = int(self.crop_size)
            # 如果指定了步长，则使用指定步长，否则根据重叠率计算
            if self.stride is not None:
                stride = int(self.stride)
                print(f"使用自定义步长: {stride}")
            else:
                stride = int(crop_size * (1 - self.overlap))
                print(f"根据重叠率计算步长: {stride} (重叠率: {self.overlap})")

            # 计算行数和列数
            if self.keep_partial:
                row_num = (height - crop_size + stride - 1) // stride
                col_num = (width - crop_size + stride - 1) // stride
            else:
                # 只计算完整的裁剪窗口
                row_num = (height - crop_size) // stride
                col_num = (width - crop_size) // stride

            # 进度条
            total_tiles = (row_num + 1) * (col_num + 1)
            pbar = tqdm(total=total_tiles, desc=f"Processing {tif_name}")

            tile_count = 0

            # 准备压缩选项
            compress_options = {
                'compress': self.compression,
                'predictor': self.predictor
            }

            # 如果使用多线程，收集所有任务
            if self.num_threads > 1:
                tasks = []
                for i in range(row_num + 1):
                    for j in range(col_num + 1):
                        yoff = i * stride
                        xoff = j * stride

                        if self.keep_partial:
                            ysize = min(crop_size, height - yoff)
                            xsize = min(crop_size, width - xoff)
                        else:
                            ysize = crop_size
                            xsize = crop_size

                            if yoff + ysize > height or xoff + xsize > width:
                                pbar.update(1)
                                continue

                        window = Window(xoff, yoff, xsize, ysize)
                        # 保存窗口参数而不是直接引用src
                        tasks.append((yoff, xoff, ysize, xsize, tile_count))
                        tile_count += 1

                # 使用线程池处理任务
                with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                    # 提交所有任务
                    future_to_task = {
                        executor.submit(self._process_tile, tif_path, task, output_dir, tif_name,
                                        crop_size, count, dtype, crs, compress_options): task
                        for task in tasks
                    }

                    # 处理完成的任务
                    completed = 0
                    for future in as_completed(future_to_task):
                        try:
                            result = future.result()
                            if result:
                                completed += 1
                        except Exception as e:
                            print(f"处理图块时出错: {e}")
                        pbar.update(1)

                tile_count = completed
            else:
                # 单线程处理
                for i in range(row_num + 1):
                    for j in range(col_num + 1):
                        # 计算裁剪窗口
                        yoff = i * stride
                        xoff = j * stride

                        if self.keep_partial:
                            ysize = min(crop_size, height - yoff)
                            xsize = min(crop_size, width - xoff)
                        else:
                            # 确保总是裁剪完整尺寸
                            ysize = crop_size
                            xsize = crop_size

                            # 如果窗口超出边界，则跳过
                            if yoff + ysize > height or xoff + xsize > width:
                                pbar.update(1)
                                continue

                        # 创建窗口
                        window = Window(xoff, yoff, xsize, ysize)

                        # 读取窗口数据
                        tile_data = src.read(window=window)

                        # 检查是否为全黑图像
                        if np.all(tile_data == 0):
                            pbar.update(1)
                            continue

                        # 如果需要保持部分图像但尺寸小于crop_size，则进行填充
                        if self.keep_partial and (ysize < crop_size or xsize < crop_size):
                            # 创建填充后的数组
                            padded_data = np.zeros((count, crop_size, crop_size), dtype=tile_data.dtype)

                            # 将原始数据复制到填充数组中
                            padded_data[:, :ysize, :xsize] = tile_data

                            # 更新tile_data为填充后的数据
                            tile_data = padded_data

                        # 获取窗口的地理变换信息
                        window_transform = src.window_transform(window)

                        # 构造输出文件名 (使用tif名加数字后缀)
                        output_path = os.path.join(
                            output_dir,
                            f"{tif_name}_{tile_count}.tif"
                        )

                        # 使用rasterio保存裁剪图像
                        with rasterio.open(
                                output_path, 'w',
                                driver='GTiff',
                                height=crop_size,
                                width=crop_size,
                                count=count,
                                dtype=dtype,
                                crs=crs,
                                transform=window_transform,
                                **compress_options
                        ) as dst:
                            dst.write(tile_data)

                        tile_count += 1
                        pbar.update(1)

            pbar.close()
            print(f"完成 {tif_name} 的裁剪，共生成 {tile_count} 个图块")

    def _process_tile(self, tif_path, task, output_dir, tif_name, crop_size, count, dtype, crs, compress_options):
        """
        处理单个图块的内部方法（用于多线程处理）
        """
        yoff, xoff, ysize, xsize, tile_count = task

        try:
            # 每个线程独立打开文件
            with rasterio.open(tif_path) as src:
                # 创建窗口
                window = Window(xoff, yoff, xsize, ysize)

                # 读取窗口数据
                tile_data = src.read(window=window)

                # 检查是否为全黑图像
                if np.all(tile_data == 0):
                    return False

                # 如果需要保持部分图像但尺寸小于crop_size，则进行填充
                if self.keep_partial and (ysize < crop_size or xsize < crop_size):
                    # 创建填充后的数组
                    padded_data = np.zeros((count, crop_size, crop_size), dtype=tile_data.dtype)

                    # 将原始数据复制到填充数组中
                    padded_data[:, :ysize, :xsize] = tile_data

                    # 更新tile_data为填充后的数据
                    tile_data = padded_data

                # 获取窗口的地理变换信息
                window_transform = src.window_transform(window)

                # 构造输出文件名 (使用tif名加数字后缀)
                output_path = os.path.join(
                    output_dir,
                    f"{tif_name}_{tile_count}.tif"
                )

                # 使用rasterio保存裁剪图像
                with rasterio.open(
                        output_path, 'w',
                        driver='GTiff',
                        height=crop_size,
                        width=crop_size,
                        count=count,
                        dtype=dtype,
                        crs=crs,
                        transform=window_transform,
                        **compress_options
                ) as dst:
                    dst.write(tile_data)

            return True
        except Exception as e:
            print(f"处理图块 {tile_count} 时出错: {e}")
            return False

    def batch_crop(self, input_dir, output_dir):
        """
        批量裁剪TIF文件

        Args:
            input_dir (str): 输入TIF文件目录
            output_dir (str): 输出目录路径
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 获取所有TIF文件
        tif_files = set()  # 使用集合自动去重
        input_path = Path(input_dir)

        # 查找所有可能的TIF文件扩展名
        for ext in ['.tif', '.tiff']:
            # 查找小写扩展名
            tif_files.update(input_path.glob(f'*{ext}'))
            # 查找大写扩展名
            tif_files.update(input_path.glob(f'*{ext.upper()}'))

        # 转换为列表
        tif_files = list(tif_files)

        # 显示找到的所有文件（调试用）
        print(f"在目录 {input_dir} 中找到以下文件:")
        for i, file in enumerate(tif_files, 1):
            print(f"  {i}. {file.name}")

        if not tif_files:
            print(f"在目录 {input_dir} 中未找到TIF文件")
            return

        print(f"总共找到 {len(tif_files)} 个TIF文件")

        # 逐个处理TIF文件
        for tif_file in tif_files:
            try:
                print(f"\n正在处理: {tif_file.name}")
                self.crop_single_tif(str(tif_file), output_dir)
            except Exception as e:
                print(f"处理文件 {tif_file.name} 时出错: {e}")


def main():
    parser = argparse.ArgumentParser(description='批量TIF图像滑动窗口裁剪工具')
    parser.add_argument('--input_dir', default="E:\\Desktop\\TIF", type=str,
                        help='输入TIF文件目录')
    parser.add_argument('--output_dir', '-o', type=str, default="E:\\Desktop\\TIF\\output",
                        help='输出裁剪图像目录')
    parser.add_argument('--crop_size', '-s', type=int, default=1024,
                        help='裁剪窗口大小 (默认: 1024)')
    parser.add_argument('--overlap', type=float, default=0.2,
                        help='窗口重叠比例 (默认: 0.3)')
    parser.add_argument('--stride', type=int, default=None,
                        help='窗口步长，如果指定则覆盖重叠率参数')
    parser.add_argument('--keep_partial', action='store_true', default=True,
                        help='是否保留不完整的边缘图块并填充到完整尺寸 (默认: True)')
    parser.add_argument('--num_threads', type=int, default=4,
                        help='处理线程数 (默认: 1)')
    parser.add_argument('--compression', type=str, default='deflate',
                        help='输出文件压缩方式 (默认: deflate)')

    args = parser.parse_args()

    # 创建裁剪器实例
    cropper = BatchTifCrop(
        crop_size=args.crop_size,
        overlap=args.overlap,
        stride=args.stride,
        keep_partial=args.keep_partial,
        num_threads=args.num_threads,
        compression=args.compression
    )

    # 执行批量裁剪
    cropper.batch_crop(args.input_dir, args.output_dir)

    print("批量裁剪完成!")


if __name__ == "__main__":
    main()
