import concurrent.futures
import os
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

def rgba_to_rgb(image_path, out_path):
    print(image_path)

    image = Image.open(image_path)

    if image.mode == "RGBA":
        image = image.convert("RGB")
        image.save(out_path)





input_url = r"E:\农作物识别测试数据（2025年）\00样本库\毕节\20260105切片\毕节-地膜-其他农作物-蔬菜-油菜\SINGLECLASSDataset\images\val"

fileList = os.listdir(input_url)

with ThreadPoolExecutor(max_workers=60) as executor:
    results = [executor.submit(rgba_to_rgb, os.path.join(input_url, file),os.path.join(input_url, file))  for file in fileList]
    for result in concurrent.futures.as_completed(results):
        result.result()
