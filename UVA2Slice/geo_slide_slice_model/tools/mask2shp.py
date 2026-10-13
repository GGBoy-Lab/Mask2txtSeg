import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS
from rasterio.features import shapes
from shapely import Polygon, make_valid
from shapely.ops import orient

from .shp_utils import merge_intersecting_polygons_with_field1, smooth_gaussian, smooth_chaikin, \
    merge_intersecting_polygons
from .romve_acute_angle import adjust_sharp_angles
from .shp_utils import base_collinear, merge_adjacent_patches, process_shapefiles

sys.setrecursionlimit(10000)


class MaskSpatialProcessor:
    def __init__(self, output_dir, crs):
        self.output_dir = output_dir
        self.crs = crs
        columns = ["type", "geometry"]
        self.gdf = gpd.GeoDataFrame(columns=columns, crs=crs, geometry="geometry")

    def add_polygon(self, polygon):
        self.gdf = self.gdf.append(polygon, ignore_index=True)

    def make_valid(self):
        self.gdf['geometry'] = self.gdf.geometry.buffer(0)
        self.gdf['geometry'] = self.gdf.geometry.apply(
            lambda geom: make_valid(geom) if not geom.is_valid else geom
        )

    def merge_intersecting_polygons(self):
        self.gdf.geometry = self.gdf.geometry.buffer(0)
        self.gdf = merge_intersecting_polygons(self.gdf)

    def merge_intersecting_polygons_with_field(self):
        # 预修复几何
        # self.make_valid()
        self.gdf = merge_intersecting_polygons_with_field1(self.gdf, field_name="type")

    def add_mask(self, mask, transform, pixel=1, type=None):
        for geo, value in shapes(
                mask,
                transform=transform,  # 使用TIFF的地理坐标
                connectivity=8,
                mask=(mask == pixel)
        ):
            feature = {
                "geometry": geo,
                "properties": {"type": type}  # 必须包含"properties"字段
            }
            new_gdf = gpd.GeoDataFrame.from_features([feature], crs=self.crs)
            self.gdf = pd.concat([self.gdf, new_gdf], ignore_index=True)

    def _process_contour(self, contour, transform):
        """处理单个轮廓返回坐标点，并确保闭合"""
        contour = cv2.approxPolyDP(contour, epsilon=1, closed=True)
        if len(contour) < 3:
            return None

        points = []
        for point in contour.squeeze():
            x, y = point
            geo_coord = transform * (x, y)
            points.append(geo_coord)

        # 确保多边形闭合
        if points[0] != points[-1]:
            points.append(points[0])
        return points

    def add_contours(self, contours, hierarchy, transform, type=None):
        if hierarchy.shape[1] != 1:
            # 转换hierarchy为二维数组
            hierarchy = hierarchy.squeeze()

            parent_indices = [i for i, h in enumerate(hierarchy) if h[3] == -1]

            for parent_idx in parent_indices:
                parent_contour = contours[parent_idx]
                # 简化外轮廓
                simplified_parent = cv2.approxPolyDP(parent_contour, epsilon=1, closed=True)
                if simplified_parent.shape[0] < 3:  # 过滤无效多边形
                    return contours
                exterior = self._process_contour(simplified_parent, transform)
                if exterior is None:
                    continue

                # 查找所有子孔洞
                holes = []
                child_idx = hierarchy[parent_idx][2]
                while child_idx != -1:
                    child_contour = contours[child_idx]
                    # 简化内轮廓
                    # simplified_child = cv2.approxPolyDP(child_contour, epsilon=1, closed=True)
                    simplified_child = child_contour
                    hole = self._process_contour(simplified_child, transform)
                    if hole:
                        # 确保孔洞方向与外环相反
                        hole_poly = Polygon(hole)
                        if hole_poly.exterior.is_ccw != Polygon(exterior).exterior.is_ccw:
                            hole = list(hole_poly.exterior.coords)[::-1]
                        holes.append(hole)
                    child_idx = hierarchy[child_idx][0]  # 下一个同级孔洞

                # 创建带孔洞的多边形

                try:
                    if holes:
                        polygon = Polygon(exterior, holes=holes)
                    else:
                        polygon = Polygon(exterior)
                    # 标准化多边形方向
                    polygon = orient(polygon, sign=1.0)
                except Exception as e:
                    print(f"创建多边形失败: {e}")
                    continue

                feature = {
                    "geometry": polygon,
                    "type": type
                }
                new_gdf = gpd.GeoDataFrame([feature], crs=self.crs)
                self.gdf = pd.concat([self.gdf, new_gdf], ignore_index=True)
        else:
            for contour in contours:
                self.add_contour(contour, transform=transform, type=type)

    def add_contour(self, contour, transform, type=None):
        # 简化轮廓点
        contour = cv2.approxPolyDP(contour, 1, closed=True)
        # 修正点坐标转换
        if contour.shape[0] < 3:  # 过滤无效多边形
            return False
        # 转换坐标点格式
        points = [transform * (point[0][0], point[0][1]) for point in contour]

        # 构建闭合多边形（确保首尾点相同）
        if not np.array_equal(points[0], points[-1]):
            points.append(points[0])

        feature = {
            "geometry": Polygon(points),
            "type": type
        }
        new_gdf = gpd.GeoDataFrame([feature], crs=self.crs)
        self.gdf = pd.concat([self.gdf, new_gdf], ignore_index=True)

    def simplify(self, tolerance):
        self.gdf["geometry"] = self.gdf.geometry.simplify(tolerance)

    def save_shp(self, shp_name=None):
        self.gdf = self.gdf[self.gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]

        if shp_name is not None:
            self.gdf.to_file(os.path.join(self.output_dir, str(shp_name)))
        else:
            timestamp = time.time()
            output_shp = os.path.join(self.output_dir, f"output_mask-{timestamp}.shp")
            self.gdf.to_file(output_shp, encoding="utf-8")
            print(f"SHP文件已保存至: {output_shp}")

    def remove_small_polygons(self, area_threshold):
        # self.gdf["geometry"] = self.gdf.to_crs("4524")
        self.gdf['area'] = self.gdf.geometry.area
        self.gdf = self.gdf[self.gdf['area'] >= area_threshold].drop(columns=['area'])
        # self.gdf = self.gdf.to_crs(self.crs)

    def smooth(self):
        self.gdf["geometry"] = self.gdf.geometry.apply(lambda g: smooth_gaussian(g, sigma=1))
        self.gdf["geometry"] = self.gdf.geometry.apply(lambda g: smooth_chaikin(g, iterations=4))

    def get_polygons(self):
        return self.gdf.geometry.tolist()

    def get_gdf(self):
        return self.gdf

    def get_crs(self):
        return self.crs

    def set_crs(self, crs):
        self.crs = crs
        columns = ["type", "geometry"]
        self.gdf = gpd.GeoDataFrame(columns=columns, crs=crs, geometry="geometry")

    def to_crs(self, out_epsg):
        self.gdf = self.gdf.to_crs(out_epsg)

    def buffer(self, distance):
        self.gdf.geometry = self.gdf.geometry.buffer(distance)

    def merge_nested_polygons(self):
        # 添加面积列
        self.gdf['area'] = self.gdf.geometry.area

        # 按面积从大到小排序
        gdf_sorted = self.gdf.sort_values('area', ascending=False).reset_index(drop=True)

        result = []

        for idx, row in gdf_sorted.iterrows():
            geom = row.geometry

            # 判断当前几何是否被之前保留的任意一个几何包含
            is_contained = False
            for kept_row in result:
                if isinstance(kept_row.geometry, Polygon):
                    if kept_row.geometry.contains(geom.centroid):  # 用质心判断是否在内部
                        is_contained = True
                        break
                else:
                    # MultiPolygon 处理
                    for poly in kept_row.geometry.geoms:
                        if poly.contains(geom.centroid):
                            is_contained = True
                            break
                    if is_contained:
                        break

            if not is_contained:
                result.append(row)

        # 构造新的 GeoDataFrame
        self.gdf = gpd.GeoDataFrame(result, columns=self.gdf.columns, crs=self.gdf.crs)

    def merge_contained_polygons(self):
        """
        合并完全被其他图斑包含的图斑，保留外层图斑的属性
        """

        # 按面积降序排序
        gdf = self.gdf.iloc[self.gdf.geometry.area.argsort()[::-1]]
        gdf.geometry = gdf.geometry.buffer(0)

        # 创建标记数组，记录需要删除的图斑
        to_remove = [False] * len(gdf)

        # 存储合并后的图斑列表
        merged_polygons = []

        for i, row in gdf.iterrows():
            if to_remove[i]:
                continue

            main_poly = row.geometry
            main_attr = row.drop('geometry').to_dict()

            # 收集被当前图斑完全包含的所有图斑
            union_polys = [main_poly]

            for j in range(i + 1, len(gdf)):
                if not to_remove[j]:
                    other_poly = gdf.iloc[j].geometry

                    # 检查完全包含关系
                    if main_poly.contains(other_poly):
                        union_polys.append(other_poly)
                        to_remove[j] = True

            # 如果有被包含图斑，合并几何
            if len(union_polys) > 1:
                # 注意：包含的图斑应该都是独立多边形，直接合并
                merged_geom = gpd.GeoSeries(union_polys).unary_union
            else:
                merged_geom = main_poly

            # 保留主图斑属性
            merged_polygons.append({
                **main_attr,
                'geometry': merged_geom
            })

        # 创建结果GeoDataFrame并保存
        self.gdf = gpd.GeoDataFrame(merged_polygons, crs=gdf.crs)

    def merge_shp_by_intersection(self, threshold=0.8):
        """
        处理Shapefile: 合并相交面积占自身80%的图斑，其他字段按最大图斑赋值
        """
        # 确保有有效几何
        gdf = self.gdf[~self.gdf.is_empty & self.gdf.is_valid]

        # 创建空间索引
        sindex = gdf.sindex

        # 添加面积列并转换为平面坐标
        gdf['_area'] = gdf.geometry.area

        # 标记已处理图斑
        processed = np.zeros(len(gdf), dtype=bool)
        results = []

        for idx in range(len(gdf)):
            if processed[idx]:
                continue

            # 创建合并组
            merge_group = set([idx])
            queue = [idx]
            processed[idx] = True

            # 查找相交图斑
            while queue:
                current_idx = queue.pop(0)
                current_geom = gdf.iloc[current_idx].geometry

                # 空间查询
                possible_index = list(sindex.intersection(current_geom.bounds))
                for other_idx in possible_index:
                    if processed[other_idx] or other_idx == current_idx:
                        continue

                    other_geom = gdf.iloc[other_idx].geometry
                    intersection = current_geom.intersection(other_geom)

                    if not intersection.is_empty:
                        # 计算相交面积占比
                        current_area = gdf.iloc[current_idx]['_area']
                        other_area = gdf.iloc[other_idx]['_area']

                        ratio_current = intersection.area / current_area
                        ratio_other = intersection.area / other_area

                        # 满足任一条件则加入
                        if ratio_current >= threshold or ratio_other >= threshold:
                            merge_group.add(other_idx)
                            queue.append(other_idx)
                            processed[other_idx] = True

            # 获取合并组中面积最大的图斑
            merge_group = list(merge_group)
            max_idx = gdf.iloc[merge_group]['_area'].idxmax()
            max_row = gdf.iloc[max_idx].drop(['_area', 'geometry']).to_dict()

            # 合并几何
            merged_geom = gdf.iloc[merge_group].geometry.unary_union
            max_row['geometry'] = merged_geom

            results.append(max_row)

        # 创建结果GeoDataFrame
        self.gdf = gpd.GeoDataFrame(results, crs=gdf.crs)

    def get_intersect_map(self, sorted_indexs, sindex, gdf=None):
        intersect_map = {}

        if gdf is None:
            gdf = self.gdf

        def process_row(idx_row):
            idx, row = idx_row
            geom = row.geometry
            if geom.is_empty:
                return idx, []
            possible_matches_index = list(sindex.intersection(geom.bounds))
            possible_matches = gdf.iloc[possible_matches_index]
            intersects = possible_matches[possible_matches.geometry.intersects(geom)]
            intersecting_indices = intersects.index[intersects.index != idx].tolist()
            return idx, intersecting_indices

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_row, (idx, gdf.loc[idx])) for idx in sorted_indexs]
            for future in futures:
                idx, result = future.result()
                intersect_map[idx] = result
        return intersect_map

    def union_peocessing(self):
        polygon_count = len(self.gdf)
        print(f"当前图斑数量: {polygon_count}")

        # 确保几何有效
        self.gdf.geometry = self.gdf.geometry.buffer(0)

        # 添加面积字段用于排序
        self.gdf['area'] = self.gdf.geometry.area
        # 按面积从大到小排序
        sorted_indices = self.gdf.sort_values('area', ascending=False).index

        sindex = self.gdf.sindex

        intersect_map = self.get_intersect_map(sorted_indices, sindex)

        # 待删除列表
        to_remove = set()
        recording_key = set()

        # 将全局变量改为局部变量传递
        current_data = {'geom': None, 'type': None}

        def union(main_idx, other_idx):
            recording_key.add(other_idx)
            other = self.gdf.iloc[other_idx].geometry
            current_data['geom'] = current_data['geom'].union(other)
            self.gdf.at[main_idx, 'geometry'] = current_data['geom']
            to_remove.add(other_idx)
            main_inter = intersect_map[main_idx]
            if intersect_map[other_idx]:
                child_idxs = intersect_map[other_idx]
                child_idxs = [i for i in child_idxs if i != main_idx and i not in main_inter]
                for child_idx in child_idxs:
                    if child_idx in recording_key:
                        continue
                    process(main_idx, child_idx)

        def process(main_idx, child_idx):

            # 使用字典代替global变量
            current_data['geom'] = self.gdf.iloc[main_idx].geometry
            current_data['type'] = self.gdf.iloc[main_idx]['type']
            # 先处理main_idx, child_idx
            other_row = self.gdf.iloc[child_idx]
            other_geom = other_row.geometry
            other_type = other_row['type']

            # 1.类型相同，只要相交直接合并
            if current_data['type'] == other_type:
                if current_data['geom'].intersects(other_geom):
                    union(main_idx, child_idx)

        for key, intersecting_indices in intersect_map.items():
            if key in recording_key:
                continue
            recording_key.add(key)

            # 复制列表避免在迭代时修改
            for child_idx in list(intersecting_indices):
                try:
                    if child_idx in to_remove:
                        continue
                    process(key, child_idx)
                except Exception as e:
                    print(f"处理错误: {e}")

        # 删除被标记的图斑
        for idx in sorted(to_remove, reverse=True):
            if idx < len(self.gdf):  # 检查索引是否有效
                self.gdf = self.gdf.drop(index=idx)

        self.gdf["geometry"] = self.gdf.geometry.buffer(0)
        # 删除面积小于20的图斑
        self.gdf = self.gdf[self.gdf['area'] >= 20]
        # 重置索引
        self.gdf.reset_index(drop=True, inplace=True)

        # 最后重新计算一次面积
        self.gdf['area'] = self.gdf.geometry.area

        self.gdf["geometry"] = self.gdf.geometry.simplify(tolerance=0.01)

        self.gdf = self.gdf[self.gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]

    def post_processing(self):
        polygon_count = len(self.gdf)
        print(f"当前图斑数量: {polygon_count}")

        # 确保几何有效
        self.gdf.geometry = self.gdf.geometry.buffer(0)

        # 添加面积字段用于排序
        self.gdf['area'] = self.gdf.geometry.area
        # 按面积从大到小排序
        sorted_indices = self.gdf.sort_values('area', ascending=False).index
        sindex = self.gdf.sindex
        intersect_map = self.get_intersect_map(sorted_indices, sindex)

        # 待删除列表
        to_remove = set()
        recording_key = set()

        # 将全局变量改为局部变量传递
        current_data = {'geom': None, 'type': None}

        def union(main_idx, other_idx):
            recording_key.add(other_idx)
            other = self.gdf.iloc[other_idx].geometry
            current_data['geom'] = current_data['geom'].union(other)
            self.gdf.at[main_idx, 'geometry'] = current_data['geom']
            to_remove.add(other_idx)
            main_inter = intersect_map[main_idx]
            if intersect_map[other_idx]:
                child_idxs = intersect_map[other_idx]
                child_idxs = [i for i in child_idxs if i != main_idx and i not in main_inter]
                for child_idx in child_idxs:
                    if child_idx in recording_key:
                        continue
                    process(main_idx, child_idx)

        def difference(main_idx, other_idx):
            other = self.gdf.iloc[other_idx].geometry
            if current_data['geom'].area >= other.area:
                self.gdf.at[other_idx, 'geometry'] = other.difference(current_data['geom'])
            else:
                current_data['geom'] = current_data['geom'].difference(other)
                self.gdf.at[main_idx, 'geometry'] = current_data['geom']

        def process(main_idx, child_idx):

            # 使用字典代替global变量
            current_data['geom'] = self.gdf.iloc[main_idx].geometry
            current_data['type'] = self.gdf.iloc[main_idx]['type']
            # 先处理main_idx, child_idx
            other_row = self.gdf.iloc[child_idx]
            other_geom = other_row.geometry
            other_type = other_row['type']

            # 1.类型相同，只要相交直接合并
            if current_data['type'] == other_type:
                if current_data['geom'].intersects(other_geom):
                    union(main_idx, child_idx)

            # 2.类型不同，相交面积过小，则裁剪掉相交面积
            elif current_data['geom'].intersects(other_geom):
                intersection_area = current_data['geom'].intersection(other_geom).area
                # 如果相交面积小于当前图斑面积的10%，将这个小图斑裁剪
                if intersection_area / current_data['geom'].area < 0.2:
                    difference(main_idx, child_idx)

                # 3.类型不同，相交面积过大，对比面积，满足条件则合并
                elif intersection_area / max(current_data['geom'].area, other_geom.area) >= 0.2:
                    union(main_idx, child_idx)

        for key, intersecting_indices in intersect_map.items():
            if key in recording_key:
                continue
            recording_key.add(key)

            # 复制列表避免在迭代时修改
            for child_idx in list(intersecting_indices):
                try:
                    if child_idx in to_remove:
                        continue
                    process(key, child_idx)
                except Exception as e:
                    print(f"处理错误: {e}")

        # 删除被标记的图斑
        for idx in sorted(to_remove, reverse=True):
            if idx < len(self.gdf):  # 检查索引是否有效
                self.gdf = self.gdf.drop(index=idx)

        # 清理无效几何
        # self.gdf = self.gdf[~self.gdf.is_empty & self.gdf.is_valid]
        self.gdf["geometry"] = self.gdf.geometry.buffer(0)
        # 最后重新计算一次面积
        self.gdf['area'] = self.gdf.geometry.area
        # 删除面积小于20的图斑
        self.gdf = self.gdf[self.gdf['area'] >= 20]
        # 重置索引
        self.gdf = self.gdf.reset_index(drop=True)

        self.gdf["geometry"] = self.gdf.geometry.simplify(tolerance=0.01)

        self.gdf = self.gdf[self.gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]

    def add_gdf(self, gdf):
        self.gdf = pd.concat([self.gdf, gdf], ignore_index=True)

    def collinear(self):
        self.gdf = self.gdf.explode(index_parts=False).reset_index(drop=True)
        self.gdf["geometry"] = self.gdf.simplify(tolerance=0.8)
        self.gdf["geometry"] = self.gdf.buffer(1)
        self.gdf["geometry"] = self.gdf.buffer(-1)
        self.gdf["geometry"] = self.gdf.buffer(1)
        self.gdf["geometry"] = self.gdf.buffer(-1)
        self.gdf = base_collinear(self.gdf)

    def base_collinear(self):
        self.gdf = self.gdf.explode(index_parts=False).reset_index(drop=True)
        self.gdf = base_collinear(self.gdf)

    def merge_adjacent_patches(self):
        self.gdf = merge_adjacent_patches(self.gdf)

    def rationalization(self):
        self.gdf = self.gdf[self.gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        self.gdf = self.gdf.explode(index_parts=True).reset_index(drop=True)

    def process_shapefiles(self, secondary_gdf):
        self.gdf = process_shapefiles(self.gdf, secondary_gdf)

    def collinear2(self):
        self.gdf["geometry"] = self.gdf.geometry.buffer(0)
        self.make_valid()
        self.collinear_en1()
        self.gdf["geometry"] = self.gdf.geometry.buffer(0)
        self.collinear_en_ascending()
        self.gdf["geometry"] = self.gdf.geometry.buffer(1)
        self.gdf["geometry"] = self.gdf.geometry.buffer(-1)

        # # self.gdf["geometry"] = self.gdf.geometry.buffer(-0.2)
        # self.gdf["geometry"] = self.gdf.geometry.buffer(0.001)
        # self.gdf["geometry"] = self.gdf.geometry.buffer(-0.001)
        # # self.gdf["geometry"] = self.gdf.geometry.buffer(-1)
        # 防止相交情况
        self.gdf = base_collinear(self.gdf)

    def collinear_en1(self, iterate=1):
        self.gdf = self.gdf.explode(index_parts=False).reset_index(drop=True)
        buffer_num = 0.3
        self.remove_small_polygons(10)
        self.gdf['area'] = self.gdf.geometry.area
        # 按面积从小到大排序
        gdf_input = self.gdf.sort_values('area').reset_index(drop=True)

        gdf_buffer = gdf_input.copy()
        # 创建缓冲
        gdf_buffer["geometry"] = gdf_buffer.geometry.buffer(buffer_num)
        re_set = set()
        # 遍历图斑（从小到大）
        for idx, row in gdf_input.iterrows():
            re_set.add(idx)
            geom = row['geometry']
            are = geom.area
            geom_buffer = geom.buffer(buffer_num, cap_style="flat", join_style="mitre")
            possible_matches_index = list(gdf_buffer.sindex.intersection(geom_buffer.bounds))
            intersecting_indices = [
                i for i in possible_matches_index
                if gdf_buffer.iloc[i].geometry.intersects(geom_buffer) and not (i == idx)
            ]
            print(f"图斑 {idx} 相交的图斑索引: {intersecting_indices}")
            if len(intersecting_indices) > 0:
                for i in intersecting_indices:
                    if i in re_set:
                        continue
                    geo_in = gdf_input.iloc[i].geometry
                    geom_in_buffer = geo_in.buffer(buffer_num, cap_style="flat", join_style="mitre")
                    if not geom_buffer.is_valid:
                        geom_buffer = make_valid(geom_buffer)
                    if not geom_in_buffer.is_valid:
                        geom_in_buffer = make_valid(geom_in_buffer)
                    geo_interserc = geom_buffer.intersection(geom_in_buffer)
                    if geo_interserc.area > 0:
                        geom = geom.union(geo_interserc)
                        geo_in = geo_in.difference(geo_interserc)

                        gdf_input.at[idx, 'geometry'] = geom
                        gdf_input.at[i, 'geometry'] = geo_in

        self.gdf = gdf_input

    def collinear_en_ascending(self):
        self.gdf = self.gdf.explode(index_parts=False).reset_index(drop=True)
        buffer_num = 0.3
        self.remove_small_polygons(10)

        self.gdf['area'] = self.gdf.geometry.area
        # 按面积从大到小排序
        gdf_input = self.gdf.sort_values('area', ascending=False).reset_index(drop=True)
        gdf_buffer = gdf_input.copy()
        # 创建缓冲
        gdf_buffer["geometry"] = gdf_buffer.geometry.buffer(0.3)
        re_set = set()
        # 遍历图斑（从大到小）
        for idx, row in gdf_input.iterrows():
            re_set.add(idx)
            geom = row['geometry']
            geom_buffer = geom.buffer(buffer_num, cap_style="flat", join_style="mitre")
            possible_matches_index = list(gdf_buffer.sindex.intersection(geom_buffer.bounds))
            intersecting_indices = [
                i for i in possible_matches_index
                if gdf_buffer.iloc[i].geometry.intersects(geom_buffer) and not (i == idx)
            ]
            print(f"图斑 {idx} 相交的图斑索引: {intersecting_indices}")
            if len(intersecting_indices) > 0:
                for i in intersecting_indices:
                    if i in re_set:
                        continue
                    geo_in = gdf_input.iloc[i].geometry


                    geom_in_buffer = geo_in.buffer(buffer_num, cap_style="flat", join_style="mitre")

                    if not geom_buffer.is_valid:
                        geom_buffer = make_valid(geom_buffer)
                    if not geom_in_buffer.is_valid:
                        geom_in_buffer = make_valid(geom_in_buffer)


                    geo_interserc = geom_buffer.intersection(geom_in_buffer)
                    if geo_interserc.area > 0:
                        geom = geom.union(geo_interserc)
                        geo_in = geo_in.difference(geo_interserc)

                        gdf_input.at[idx, 'geometry'] = geom
                        gdf_input.at[i, 'geometry'] = geo_in

        self.gdf = gdf_input

    def collinear_enhance(self):
        """增强共线"""
        # 找到所有相交的图斑
        gdf_buffer = self.gdf.copy()
        gdf_buffer["geometry"] = gdf_buffer.geometry.buffer(0)
        # 添加面积字段用于排序
        gdf_buffer['area'] = gdf_buffer.geometry.area
        # 按面积从大到小排序
        sorted_indices = gdf_buffer.sort_values('area', ascending=False).index
        sindex = gdf_buffer.sindex
        intersect_map = self.get_intersect_map(sorted_indices, sindex, gdf_buffer)
        for key, intersecting_indices in intersect_map.items():
            geom_buffer = gdf_buffer.iloc[key].geometry
            geom = self.gdf.iloc[key].geometry
            for i in intersecting_indices:
                geo_intersect_buffer = gdf_buffer.iloc[i].geometry
                geo_intersect = self.gdf.iloc[i].geometry
                # 找到两个图斑相交部分
                intersection = geom_buffer.intersection(geo_intersect_buffer)
                # 面积小的图斑去裁剪面积大的图斑,面积小的合并
                if geo_intersect_buffer.area >= geom_buffer.area:
                    geom = geom.union(intersection)
                    geo_intersect = geo_intersect.difference(geom)
                    self.gdf.at[i, 'geometry'] = geo_intersect
                    self.gdf.at[key, 'geometry'] = geom
                else:
                    geo_intersect = geo_intersect.union(intersection)
                    geom = geom.difference(geo_intersect)
                    self.gdf.at[i, 'geometry'] = geo_intersect
                    self.gdf.at[key, 'geometry'] = geom

    def remove_acute_angle(self):
        self.gdf["geometry"] = self.gdf.simplify(tolerance=0.5)
        self.base_collinear()
        self.gdf = adjust_sharp_angles(self.gdf)
        # self.base_collinear()



if __name__ == '__main__':
    image = cv2.imread("416040_303466.png")
    image = np.all(image == (255, 255, 255), axis=-1).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    with rasterio.open(r"E:\备份\油菜_tif\ceshi\36440083.128219_2914595.672296.tif") as src_tif:
        transform = src_tif.transform

    mask_pre = MaskSpatialProcessor(r"F:\工作文件\目标识别与分割\test255", CRS.from_epsg("4524"))
    mask_pre.add_contours(contours, hierarchy, transform=transform, type=778)
    mask_pre.save_shp()
