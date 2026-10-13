import os
from glob import glob

import numpy as np
from osgeo import gdal, ogr, osr
from PIL import Image
import json

from tqdm import tqdm


class GeoSpatialProcessor:
    def __init__(self, tif_path, shp_path, output_dir):
        self.tif_path = tif_path
        self.shp_path = shp_path
        self.output_dir = output_dir
        self._init_dirs()

        # 加载数据
        self.img_ds = gdal.Open(tif_path)
        self.shp_ds = ogr.Open(shp_path)
        self.proj = self.img_ds.GetProjection()
        self.width = self.img_ds.RasterXSize
        self.height = self.img_ds.RasterYSize
        self.img = self.img_ds.ReadAsArray(0, 0, self.width, self.height)
        self.bands = self.img_ds.RasterCount
        self.geotrans = self.img_ds.GetGeoTransform()

        # 坐标系处理
        self.shp_ds = self._reproject_vector()
        self.shp_layer = self.shp_ds.GetLayer()

    def _init_dirs(self):
        """创建输出目录结构"""
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "labels"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "meta"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "empty"), exist_ok=True)

    def _reproject_vector(self):
        """
        检查给定的TIFF和Shapefile是否具有相同的空间参考系统。
        """
        # 获取TIFF文件的空间参考
        img_srs = osr.SpatialReference()

        if hasattr(osr,'OAMS_TRADITIONAL_GIS_ORDER'):
            img_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        img_srs.ImportFromWkt(self.img_ds.GetProjection())

        # 获取Shapefile图层的空间参考
        shp_layer = self.shp_ds.GetLayer()
        shp_srs = shp_layer.GetSpatialRef()

        if hasattr(osr,'OAMS_TRADITIONAL_GIS_ORDER') and shp_srs:
            shp_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        # 检查是否需要坐标转换
        if not img_srs.IsSame(shp_srs):
            print("坐标系不同，进行坐标转换...")

            # 创建坐标转换对象
            coord_transform = osr.CoordinateTransformation(shp_srs, img_srs)

            # 创建内存中的Shapefile数据源
            mem_driver = ogr.GetDriverByName('Memory')
            out_ds = mem_driver.CreateDataSource('mem_data')
            if out_ds is None:
                raise IOError("创建内存数据源失败")

            # 复制输入图层的定义到输出图层
            out_layer = out_ds.CopyLayer(shp_layer, 'converted_layer')

            # 遍历图层中的所有要素并进行坐标转换
            for feature in out_layer:
                geom = feature.GetGeometryRef()
                if geom is not None:
                    geom.Transform(coord_transform)
                    feature.SetGeometry(geom)
                    out_layer.SetFeature(feature)

            print("坐标转换完成")
            return out_ds
        else:
            print("坐标系相同，无需转换。")
            return self.shp_ds

    def CoordTransf(self, Xpixel, Ypixel):
        XGeo = self.geotrans[0] + self.geotrans[1] * Xpixel + Ypixel * self.geotrans[2]
        YGeo = self.geotrans[3] + self.geotrans[4] * Xpixel + Ypixel * self.geotrans[5]
        return XGeo, YGeo

    def get_tif_bbox(self, im_data, geotransform):
        im_data = np.array([im_data])
        _, im_bands, height, width = im_data.shape

        # 计算四个角点的地理坐标
        def transform_point(x, y):
            """根据地理变换参数转换像素坐标到地理坐标"""
            lon = geotransform[0] + x * geotransform[1] + y * geotransform[2]
            lat = geotransform[3] + x * geotransform[4] + y * geotransform[5]
            return lon, lat

        # 四个角点的像素坐标
        corners = [(0, 0), (width, 0), (width, height), (0, height), (0, 0)]

        # 转换为地理坐标
        bbox_coords = [transform_point(x, y) for x, y in corners]

        # 获取空间参考系统（SRS）
        srs = osr.SpatialReference()
        srs.ImportFromWkt(self.proj)

        # 如果需要将坐标转换为其他坐标系，可以在这里进行转换
        # 例如，转换为EPSG:4326 (WGS84)
        # if srs.GetAttrValue("AUTHORITY", 0) != "EPSG" or srs.GetAttrValue("AUTHORITY", 1) != "4326":
        #     target_srs = osr.SpatialReference()
        #     target_srs.ImportFromEPSG(4326)
        #     coord_transform = osr.CoordinateTransformation(srs, target_srs)
        #     bbox_coords = [coord_transform.TransformPoint(x, y)[:2] for x, y in bbox_coords]

        return bbox_coords

    def create_bbox_geometry(self, bbox):
        """根据边界框坐标创建一个多边形几何对象"""
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for x1, y1 in bbox:
            ring.AddPoint(x1, y1)
        bbox_geom = ogr.Geometry(ogr.wkbPolygon)
        bbox_geom.AddGeometry(ring)

        return bbox_geom

    def check_intersection(self, tif_bbox):
        self.shp_layer.SetSpatialFilter(None)
        self.shp_layer.ResetReading()

        # 获取图斑的外包矩形
        tif_envelope = tif_bbox.GetEnvelope()  # (min_x, max_x, min_y, max_y)

        for feature in self.shp_layer:
            shp_geom = feature.GetGeometryRef()
            if shp_geom:
                # 获取shp要素的外包矩形
                shp_envelope = shp_geom.GetEnvelope()  # (min_x, max_x, min_y, max_y)

                # 判断外包矩形是否相交（这就是SetSpatialFilter内部做的事情）
                bbox_intersects = not (tif_envelope[1] < shp_envelope[0] or  # tif右边界 < shp左边界
                                       tif_envelope[0] > shp_envelope[1] or  # tif左边界 > shp右边界
                                       tif_envelope[3] < shp_envelope[2] or  # tif上边界 < shp下边界
                                       tif_envelope[2] > shp_envelope[3])  # tif下边界 > shp上边界

                if bbox_intersects:
                    return True

        return False

    def bbox_intersects(self, geom1, geom2):
        """
        判断两个几何的外包矩形是否相交

        Args:
            geom1: 第一个几何对象
            geom2: 第二个几何对象

        Returns:
            bool: 外包矩形是否相交
        """
        if not geom1 or not geom2:
            return False

        # 获取两个几何的外包矩形
        envelope1 = geom1.GetEnvelope()  # (min_x, max_x, min_y, max_y)
        envelope2 = geom2.GetEnvelope()  # (min_x, max_x, min_y, max_y)

        # 判断外包矩形是否相交
        # 原理：两个矩形不相交的条件是：一个在另一个的左边、右边、上边或下边
        if (envelope1[1] < envelope2[0] or  # geom1右边界 < geom2左边界
                envelope1[0] > envelope2[1] or  # geom1左边界 > geom2右边界
                envelope1[3] < envelope2[2] or  # geom1上边界 < geom2下边界
                envelope1[2] > envelope2[3]):  # geom1下边界 > geom2上边界
            return False
        else:
            return True

    def writeTiff(self, im_data, im_geotrans, path):
        datatype = gdal.GDT_Float32
        im_data = np.array([im_data])
        _, im_bands, im_height, im_width = im_data.shape
        # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
        if (dataset != None):
            dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
            dataset.SetProjection(self.proj)  # 写入投影
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[0, i])
        del dataset

    def process(self, crop_size, palette_config, overlap=0.1, suffix=""):
        """主处理流程"""
        # 参数验证
        self.crop_size = crop_size
        self.suffix = suffix
        self.palette_config = palette_config
        self._validate_palette(palette_config)

        # 保存调色板配置
        with open(os.path.join(self.output_dir, "meta/palette.json"), 'w') as f:
            json.dump(palette_config, f)

        # 获取影像参数
        width = self.img_ds.RasterXSize
        height = self.img_ds.RasterYSize
        geotrans = self.img_ds.GetGeoTransform()

        # 校验shp有没有在整个tif中
        # ring = ogr.Geometry(ogr.wkbLinearRing)
        # ring.AddPoint(geotrans[0], geotrans[3])
        # ring.AddPoint(geotrans[0] + width * geotrans[1], geotrans[3])
        # ring.AddPoint(geotrans[0] + width * geotrans[1],
        #               geotrans[3] + height * geotrans[5])
        # ring.AddPoint(geotrans[0], geotrans[3] + height * geotrans[5])
        # ring.AddPoint(geotrans[0], geotrans[3])
        # tif_poly = ogr.Geometry(ogr.wkbLinearRing)
        # tif_poly.AddGeometry(ring)
        # if not self.check_intersection(tif_poly):
        #     return

        # 裁剪参数计算
        crop_size = int(crop_size)
        RowNum = int((height - crop_size * overlap) / (crop_size * (1 - overlap)))
        ColumnNum = int((width - crop_size * overlap) / (crop_size * (1 - overlap)))

        # 裁剪主循环
        for i in range(RowNum + 1):
            for j in range(ColumnNum + 1):
                # 计算裁剪窗口
                yoff = int(i * crop_size * (1 - overlap)) if i < RowNum else height - crop_size
                xoff = int(j * crop_size * (1 - overlap)) if j < ColumnNum else width - crop_size
                ysize = min(crop_size, height - yoff)
                xsize = min(crop_size, width - xoff)

                # 获取裁剪块地理范围
                xgeo, ygeo = self.CoordTransf(xoff, yoff)
                crop_geotrans = (xgeo, self.geotrans[1], self.geotrans[2],
                                 ygeo, self.geotrans[4], self.geotrans[5])

                cropped = self.img[:, yoff:yoff + ysize, xoff:xoff + xsize] if self.bands > 1 else self.img[
                    yoff:yoff + ysize,
                    xoff:xoff + xsize]
                bbox = self.get_tif_bbox(cropped, crop_geotrans)

                crop_geom = self.create_bbox_geometry(bbox)

                # 判断相交性
                if self.check_intersection(crop_geom):
                    save_dir = 'images'
                    self._generate_label(xoff, yoff, xsize, ysize, crop_geotrans, palette_config, crop_geom)
                    self._save_crop(xoff, yoff, xsize, ysize, crop_geotrans, save_dir)
                else:
                    save_dir = 'empty'
                    # self._save_crop(xoff, yoff, xsize, ysize, crop_geotrans, save_dir)

    def _validate_palette(self, palette):
        """调色板配置验证"""
        required_fields = ['classes', 'colors', 'field']
        if not all(k in palette for k in required_fields):
            raise ValueError("调色板配置缺少必要字段")

        if len(palette['classes']) != len(palette['colors']):
            raise ValueError("类别与颜色数量不匹配")

    def _save_crop(self, xoff, yoff, xsize, ysize, geotrans, save_dir):
        """保存裁剪影像"""
        output_path = os.path.join(
            self.output_dir,
            f"{save_dir}/{geotrans[0]:.8f}_{geotrans[3]:.8f}_{self.crop_size}_{self.suffix}.tif"
        )

        # 读取裁剪区域是否全黑
        is_all_black = True
        is_all_white = True
        is_b_and_w = True
        for b in range(self.img_ds.RasterCount):
            band_data = self.img_ds.GetRasterBand(b + 1).ReadAsArray(xoff, yoff, xsize, ysize)
            if not (band_data == 0).all():
                is_all_black = False
            if not (band_data >= 254).all():
                is_all_white = False
            if not ((band_data == 0) | ((band_data) >= 254)).all():
                is_b_and_w = False
            if not is_all_black and not is_all_white and not is_b_and_w:
                break

        if is_all_black:
            return
        if is_all_white:
            return
        if is_b_and_w:
            return

        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(output_path, xsize, ysize,
                           3)
        ds.SetGeoTransform(geotrans)
        ds.SetProjection(self.proj)

        for b in range(3):
            band_data = self.img_ds.GetRasterBand(b + 1).ReadAsArray(xoff, yoff, xsize, ysize)
            ds.GetRasterBand(b + 1).WriteArray(band_data)
        ds = None

    def _generate_label(self, xoff, yoff, xsize, ysize, geotrans, palette, crop_geom):
        """生成分类标注图"""
        # 保存结果
        output_path = os.path.join(
            self.output_dir,
            f"labels/{geotrans[0]:.8f}_{geotrans[3]:.8f}_{self.crop_size}_{self.suffix}.png"
        )
        # 创建内存数据集
        driver = gdal.GetDriverByName('MEM')
        mem_driver = ogr.GetDriverByName('Memory')
        ds = driver.Create('', xsize, ysize, 3, gdal.GDT_Byte)
        ds.SetGeoTransform(geotrans)
        ds.SetProjection(self.proj)

        # 遍历图层中的所有要素，并根据类型进行栅格化
        for feature in self.shp_layer:
            geom = feature.GetGeometryRef()
            if geom is None:
                continue
            if not self.bbox_intersects(crop_geom, geom):
                continue

            # 获取要素的类型属性
            feature_type = feature.GetField(f"{self.palette_config['field']}")
            if feature_type not in self.palette_config["classes"]:
                print(f"未知类型 {feature_type}，跳过...")
                continue

            color = self.palette_config["colors"][self.palette_config["classes"].index(feature_type)]
            temp_ds = mem_driver.CreateDataSource('')
            temp_layer = temp_ds.CreateLayer('', srs=self.shp_layer.GetSpatialRef(), geom_type=geom.GetGeometryType())
            temp_feat = ogr.Feature(temp_layer.GetLayerDefn())
            temp_feat.SetGeometry(geom.Clone())
            temp_layer.CreateFeature(temp_feat)
            # 创建临时内存栅格数据集用于单个要素的栅格化
            temp_raster_ds = driver.Create('', xsize, ysize, 1, gdal.GDT_Byte)
            temp_raster_ds.SetGeoTransform(geotrans)
            temp_band = temp_raster_ds.GetRasterBand(1)
            temp_band.Fill(0)

            # 栅格化单个要素
            gdal.RasterizeLayer(temp_raster_ds, [1], temp_layer, burn_values=[1],
                                options=["ALL_TOUCHED=TRUE"])

            # 将栅格化的结果应用到目标栅格数据集的RGB波段上
            temp_array = temp_raster_ds.ReadAsArray()
            for i, c in enumerate(color):
                band = ds.GetRasterBand(i + 1)
                band_array = band.ReadAsArray()
                band_array[temp_array == 1] = c
                band.WriteArray(band_array)
            # 清理临时数据集
            temp_ds.Destroy()
            temp_raster_ds = None

        # 读取栅格数据并保存为PNG
        rgb_array = np.dstack([ds.GetRasterBand(i + 1).ReadAsArray() for i in range(3)])
        Image.fromarray(rgb_array, mode='RGB').save(output_path)


def batch_slice(tif_url, out_url, shp_url, crop_size, classes, colors, suffix=""):
    image_paths = glob(os.path.join(tif_url, "**", '*.tif'), recursive=True)
    with tqdm(total=len(image_paths), desc="Processing Images") as pbar:

        for e in image_paths:
            try:
                # url = os.path.join(image_per, os.path.dirname(e)[len(image_tif):][len('\\'):],"sample")
                url = os.path.join(out_url)
                config = {
                    "input_tif": e,
                    "input_shp": shp_url,
                    "output_dir": url,
                    "crop_size": crop_size,
                    "suffix": suffix,
                    "overlap": 0.1,
                    "palette": {
                        "field": "type",  # shp中的类别字段名
                        "classes": classes,
                        "colors": colors
                    }
                }
                processor = GeoSpatialProcessor(
                    config["input_tif"],
                    config["input_shp"],
                    config["output_dir"]
                )
                processor.process(
                    crop_size=config["crop_size"],
                    palette_config=config["palette"],
                    overlap=config["overlap"],
                    suffix=config["suffix"]
                )
            except Exception as e:
                print("错误：" + str(e))
            finally:
                pbar.update(1)
