import os
import tempfile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from osgeo import ogr
from rtree.index import Index
from scipy.ndimage import gaussian_filter1d
from shapely import Polygon, MultiPolygon, box, make_valid
from shapely.ops import unary_union
from shapely.ops import unary_union


def smooth_chaikin(geom, iterations=3):
    """修复坐标长度问题的 Chaikin 平滑"""
    if geom.is_empty:
        return geom

    def _smooth_ring(ring):
        coords = np.array(ring.coords[:-1])  # 移除闭合重复点（假设原环已闭合）
        for _ in range(iterations):
            n = len(coords)
            # 生成插值点
            even = 0.75 * coords + 0.25 * np.roll(coords, -1, axis=0)
            odd = 0.25 * coords + 0.75 * np.roll(coords, -1, axis=0)
            # 合并并保持闭合
            coords = np.empty((2 * n, 2))
            coords[0::2] = even
            coords[1::2] = odd
        # 重新闭合多边形
        return np.vstack([coords, coords[0]]).tolist()

    if isinstance(geom, Polygon):
        exterior = _smooth_ring(geom.exterior)
        interiors = [_smooth_ring(interior) for interior in geom.interiors]
        return Polygon(exterior, interiors)
    elif isinstance(geom, MultiPolygon):
        return MultiPolygon([smooth_chaikin(poly) for poly in geom.geoms])
    else:
        return geom


def smooth_gaussian(geom, sigma=2.0):
    """高斯平滑多边形坐标"""
    if geom.is_empty or not (isinstance(geom, (Polygon, MultiPolygon))):
        return geom

    def _smooth_ring(ring):
        coords = np.array(ring.coords)
        # 分别对 x/y 坐标应用高斯滤波
        x = gaussian_filter1d(coords[:, 0], sigma=sigma, mode="wrap")
        y = gaussian_filter1d(coords[:, 1], sigma=sigma, mode="wrap")
        return np.column_stack([x, y]).tolist()

    if isinstance(geom, Polygon):
        exterior = _smooth_ring(geom.exterior)
        interiors = [_smooth_ring(interior) for interior in geom.interiors]
        return Polygon(exterior, interiors)
    elif isinstance(geom, MultiPolygon):
        return MultiPolygon([smooth_gaussian(poly) for poly in geom.geoms])
    else:
        return geom


def merge_intersecting_polygons(gdf):
    """
    合并相交或相邻的图斑。

    参数:
        gdf (GeoDataFrame): 输入的 GeoDataFrame，包含多边形几何对象。

    返回:
        GeoDataFrame: 合并后的 GeoDataFrame。
    """
    # 创建一个列表存储已处理的索引
    processed = set()
    merged_geometries = []

    # 遍历每个图斑
    for idx, geom in gdf.geometry.items():
        if idx in processed:
            continue

        # 找到与当前图斑相交或相邻的所有图斑
        group = [geom]
        to_process = {idx}

        while to_process:
            current_idx = to_process.pop()
            processed.add(current_idx)

            # 找到与当前图斑相交的所有图斑
            neighbors = gdf[gdf.geometry.intersects(gdf.geometry[current_idx])]
            for neighbor_idx, neighbor_geom in neighbors.geometry.items():
                if neighbor_idx not in processed:
                    group.append(neighbor_geom)
                    to_process.add(neighbor_idx)

        # 合并该组图斑
        merged_geometry = unary_union(group)
        merged_geometries.append(merged_geometry)

    # 创建新的 GeoDataFrame
    merged_gdf = gpd.GeoDataFrame(geometry=merged_geometries, crs=gdf.crs)
    return merged_gdf


def merge_intersecting_polygons_with_field(gdf, field_name="type"):
    """
    合并相交、相邻或包含的图斑，并根据指定字段进行分组。

    参数:
        gdf (GeoDataFrame): 输入的 GeoDataFrame，包含多边形几何对象和指定字段。
        field_name (str): 用于分组的字段名称，默认为 "type"。

    返回:
        GeoDataFrame: 合并后的 GeoDataFrame，包含合并后的几何对象和指定字段。
    """
    # 创建一个列表存储已处理的索引
    processed = set()
    merged_geometries = []
    merged_fields = []

    # 遍历每个图斑
    for idx, row in gdf.iterrows():
        geom = row.geometry
        field_value = row[field_name]  # 获取当前图斑的字段值

        if idx in processed:
            continue

        # 找到与当前图斑相交、相邻或包含的所有图斑
        group = [geom]
        to_process = {idx}

        while to_process:
            current_idx = to_process.pop()
            processed.add(current_idx)

            # 找到与当前图斑相交、相邻或包含的所有图斑，并且字段值相同
            neighbors = gdf[
                (
                        (gdf.geometry.intersects(gdf.geometry[current_idx])) |  # 相交
                        (gdf.geometry.contains(gdf.geometry[current_idx])) |  # 包含
                        (gdf.geometry.within(gdf.geometry[current_idx]))  # 被包含
                ) &
                (gdf[field_name] == field_value)
                ]
            for neighbor_idx, neighbor_row in neighbors.iterrows():
                if neighbor_idx not in processed:
                    group.append(neighbor_row.geometry)
                    to_process.add(neighbor_idx)

        # 合并该组图斑
        merged_geometry = unary_union(group)
        merged_geometries.append(merged_geometry)
        merged_fields.append(field_value)  # 继承字段值

    # 创建新的 GeoDataFrame
    merged_gdf = gpd.GeoDataFrame(
        {field_name: merged_fields, "geometry": merged_geometries},
        crs=gdf.crs
    )
    return merged_gdf


def merge_intersecting_polygons_with_field_optimized(gdf, field_name="type"):
    """
    使用空间索引和并查集优化合并相交、相邻或包含的图斑，并根据字段分组。

    参数:
        gdf (GeoDataFrame): 输入的 GeoDataFrame，包含多边形几何对象和指定字段。
        field_name (str): 用于分组的字段名称，默认为 "type"。

    返回:
        GeoDataFrame: 合并后的 GeoDataFrame，包含合并后的几何对象和指定字段。
    """
    # 创建空间索引
    sindex = gdf.sindex
    n = len(gdf)
    geometries = gdf.geometry.tolist()
    field_values = gdf[field_name].values

    # 并查集初始化
    parent = np.arange(n)

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # 路径压缩
            u = parent[u]
        return u

    def union(u, v):
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            parent[root_v] = root_u

    # 构建邻接关系
    for i in range(n):
        if parent[i] != i:  # 已合并过的跳过
            continue
        geom = geometries[i]
        field_value = field_values[i]

        # 查询空间索引（快速筛选）
        possible_matches = list(sindex.intersection(geom.bounds))
        # 仅处理i之后的元素避免重复检查
        possible_matches = [j for j in possible_matches if j > i]

        for j in possible_matches:
            # 字段值不同则跳过
            if field_values[j] != field_value:
                continue
            # 精确几何关系检查
            if geom.intersects(geometries[j]) or \
                    geom.contains(geometries[j]) or \
                    geometries[j].contains(geom):
                union(i, j)

    # 分组聚合
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    # 合并几何并保留字段
    merged_geometries = []
    merged_fields = []
    for group in groups.values():
        merged_geom = unary_union([geometries[i] for i in group])
        merged_geometries.append(merged_geom)
        merged_fields.append(field_values[group[0]])

    # 生成结果
    return gpd.GeoDataFrame(
        {field_name: merged_fields, "geometry": merged_geometries},
        crs=gdf.crs
    )


def merge_intersecting_polygons_with_field1(gdf, field_name="type"):
    """
    合并相交、相邻或包含的图斑，并根据指定字段进行分组。

    参数:
        gdf (GeoDataFrame): 输入的 GeoDataFrame，包含多边形几何对象和指定字段。
        field_name (str): 用于分组的字段名称，默认为 "type"。

    返回:
        GeoDataFrame: 合并后的 GeoDataFrame，包含合并后的几何对象和指定字段。
    """
    sindex = gdf.sindex  # 预加载空间索引
    field_values = gdf[field_name].values  # 提前提取字段值
    processed = set()
    merged_geometries = []
    merged_fields = []

    for idx, geom in enumerate(gdf.geometry):
        if idx in processed:
            continue
        field_value = field_values[idx]

        group = [geom]
        to_process = set([idx])
        while to_process:
            current_idx = to_process.pop()
            if current_idx in processed:
                continue
            processed.add(current_idx)
            current_geom = gdf.geometry[current_idx]
            # 通过 sindex 缩小范围
            possible_matches_index = list(sindex.query(current_geom, predicate='intersects'))
            possible_matches = gdf.iloc[possible_matches_index]
            for neighbor_idx in possible_matches.index:
                if neighbor_idx not in processed and field_values[neighbor_idx] == field_value:
                    # 精确判断空间关系
                    if (possible_matches.geometry[neighbor_idx].intersects(current_geom) or
                            possible_matches.geometry[neighbor_idx].contains(current_geom) or
                            possible_matches.geometry[neighbor_idx].within(current_geom)):
                        group.append(possible_matches.geometry[neighbor_idx])
                        to_process.add(neighbor_idx)
        # 批量合并
        merged_geometry = unary_union(group)
        merged_geometries.append(merged_geometry)
        merged_fields.append(field_value)

    # 创建新的 GeoDataFrame
    merged_gdf = gpd.GeoDataFrame(
        {field_name: merged_fields, "geometry": merged_geometries},
        crs=gdf.crs
    )
    return merged_gdf


# def remove_small_polygons(gdf, min_area):
#     """
#     删除面积小于指定阈值的图斑。
#     :param gdf: 输入的GeoDataFrame
#     :param min_area: 最小面积阈值（单位与CRS一致，例如平方米）
#     :return: 过滤后的GeoDataFrame
#     """
#     # 计算每个图斑的面积并过滤
#     gdf['area'] = gdf.geometry.area
#     filtered_gdf = gdf[gdf['area'] >= min_area].drop(columns=['area'])  # 删除临时面积列
#     return filtered_gdf

def remove_small_polygons(gdf, min_area):
    """
    删除面积小于指定阈值的图斑。
    :param gdf: 输入的GeoDataFrame
    :param min_area: 最小面积阈值（单位：平方米）
    :param target_crs: 用于面积计算的目标CRS（例如UTM）
    :return: 过滤后的GeoDataFrame
    """
    # 投影到目标CRS以计算面积

    gdf['area'] = gdf.geometry.area  # 计算面积
    filtered_gdf = gdf[gdf['area'] >= min_area].drop(columns=['area'])  # 删除临时面积列

    return filtered_gdf


def merge_shp_from_dir(input_dir):
    # 获取所有SHP文件路径（含子目录）
    shp_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".shp"):
                shp_files.append(os.path.join(root, file))

    if not shp_files:
        raise ValueError("输入目录中未找到任何Shapefile文件")

    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    output_shp = os.path.join(temp_dir, 'merged.shp')

    # 读取首个文件的元数据
    first_ds = ogr.Open(shp_files[0])
    first_layer = first_ds.GetLayer()
    srs = first_layer.GetSpatialRef()
    geom_type = first_layer.GetGeomType()
    defn = first_layer.GetLayerDefn()

    # 创建输出文件
    driver = ogr.GetDriverByName('ESRI Shapefile')
    out_ds = driver.CreateDataSource(output_shp)
    out_layer = out_ds.CreateLayer('merged', srs=srs, geom_type=geom_type)

    # 复制字段定义
    for i in range(defn.GetFieldCount()):
        out_layer.CreateField(defn.GetFieldDefn(i))

    # 合并要素
    for shp in shp_files:
        ds = ogr.Open(shp)
        layer = ds.GetLayer()
        for feature in layer:
            out_feature = ogr.Feature(out_layer.GetLayerDefn())
            out_feature.SetGeometry(feature.GetGeometryRef().Clone())
            for j in range(feature.GetFieldCount()):
                out_feature.SetField(j, feature.GetField(j))
            out_layer.CreateFeature(out_feature)
            out_feature = None
        ds = None

    out_ds = None
    return temp_dir


def resolve_overlaps(gdf, field_name):
    """
    处理GeoDataFrame中的相交图斑，将相交部分分配给面积最大的图斑，其他图斑去除相交部分。

    参数:
        gdf (gpd.GeoDataFrame): 包含多边形图斑的GeoDataFrame
        field_name (str): 图斑的标识字段名（本实现中未使用，但保留为参数）

    返回:
        gpd.GeoDataFrame: 处理后的无相交图斑的GeoDataFrame
    """
    # 复制原始GeoDataFrame，避免修改原始数据
    gdf = gdf.copy()

    # 添加面积字段
    gdf['area'] = gdf.geometry.area

    # 按面积降序排序
    gdf_sorted = gdf.sort_values(by='area', ascending=False).reset_index(drop=True)

    # 初始化结果GeoDataFrame
    result_gdf = gpd.GeoDataFrame(columns=gdf.columns, crs=gdf.crs)

    for idx in range(len(gdf_sorted)):
        row = gdf_sorted.iloc[idx].copy()
        current_geom = row.geometry

        # 如果已有结果，计算其几何并集
        if not result_gdf.empty:
            existing_union = result_gdf.unary_union
            current_geom = current_geom.difference(existing_union)

        # 如果处理后几何非空，则添加到结果中
        if not current_geom.is_empty:
            row['geometry'] = current_geom
            new_row_gdf = gpd.GeoDataFrame([row], columns=gdf.columns, crs=gdf.crs)
            result_gdf = pd.concat([result_gdf, new_row_gdf], ignore_index=True)

    # 删除临时的面积字段
    result_gdf = result_gdf.drop(columns=['area'])

    return result_gdf


def resolve_overlaps1(gdf):
    """
    处理GeoDataFrame中的相交图斑，将相交部分分配给面积最大的图斑，其他图斑去除相交部分。

    参数:
        gdf (gpd.GeoDataFrame): 包含多边形图斑的GeoDataFrame

    返回:
        gpd.GeoDataFrame: 处理后的无相交图斑的GeoDataFrame
    """
    # 添加面积字段并按面积降序排序
    gdf = gdf.copy()
    gdf['area'] = gdf.geometry.area
    gdf_sorted = gdf.sort_values(by='area', ascending=False).reset_index(drop=True)

    # 初始化结果列表和空间索引
    result_rows = []
    existing_geoms = []  # 存储已处理的几何体
    idx = Index()  # 空间索引（R树）

    for _, row in gdf_sorted.iterrows():
        current_geom = row.geometry

        # 通过空间索引快速查询可能相交的几何体
        intersecting_indices = list(idx.intersection(current_geom.bounds, objects=False))
        if intersecting_indices:
            intersecting_geoms = [existing_geoms[i] for i in intersecting_indices]
            union_intersecting = unary_union(intersecting_geoms)
            current_geom = current_geom.difference(union_intersecting)

        # 如果处理后几何非空，则添加到结果中
        if not current_geom.is_empty:
            row['geometry'] = current_geom
            result_rows.append(row)
            # 更新空间索引和几何体列表
            existing_geoms.append(current_geom)
            idx.insert(len(existing_geoms) - 1, current_geom.bounds)

    # 删除临时的面积字段并生成结果GeoDataFrame
    result_gdf = gpd.GeoDataFrame(result_rows, crs=gdf.crs)
    result_gdf = result_gdf.drop(columns=['area'])

    return result_gdf


def process_shapefiles(main_gdf, secondary_gdf):
    # 确保坐标系一致
    # if main_gdf.crs != secondary_gdf.crs:
    #     secondary_gdf = secondary_gdf.to_crs(main_gdf.crs)

    result_features = []

    for _, sec_row in secondary_gdf.iterrows():
        sec_geom = sec_row.geometry

        # 判断是否与主图斑有交集
        intersecting_main = main_gdf[main_gdf.intersects(sec_geom)]

        if intersecting_main.empty:
            # 没有交集，直接加入结果
            result_features.append(sec_row)
        else:
            # 有交集，进行差集操作
            diff_geom = sec_geom.difference(intersecting_main.unary_union)
            if not diff_geom.is_empty:
                new_row = sec_row.copy()
                new_row['geometry'] = diff_geom
                result_features.append(new_row)

    # 将符合条件的新要素组成 GeoDataFrame
    if result_features:
        new_features_gdf = gpd.GeoDataFrame(result_features, crs=main_gdf.crs)
        # 合并主图斑和新增部分
        final_gdf = pd.concat([main_gdf, new_features_gdf], ignore_index=True)
    else:
        final_gdf = main_gdf.copy()

    return final_gdf


def base_collinear(gdf_input):
    """
    基础共线
    :param gdf_input:
    :return:
    """
    # gdf_input = gpd.read_file(gdf_input)
    # gdf_input = gdf_input.to_crs("4524")

    # gdf_input = remove_small_polygons(gdf_input, 30)
    # 计算面积并添加为新列
    gdf_input['area'] = gdf_input.geometry.area
    # 按面积从小到大排序
    gdf_input = gdf_input.sort_values('area').reset_index(drop=True)

    # 遍历图斑（从小到大）
    for idx, row in gdf_input.iterrows():
        geom = row['geometry']
        geom = geom.buffer(0)
        if geom.is_valid:
            geom = geom
        else:
            geom = make_valid(geom)

        area = row['area']
        possible_matches_index = list(gdf_input.sindex.intersection(geom.bounds))
        intersecting_indices = [
            i for i in possible_matches_index
            if gdf_input.iloc[i].geometry.intersects(geom) and not (i == idx)
        ]
        print(f"图斑 {idx} 相交的图斑索引: {intersecting_indices}")
        if len(intersecting_indices) > 0:
            for i in intersecting_indices:
                geom = geom.difference(gdf_input.iloc[i].geometry)

                if geom.is_valid:
                    geom = geom
                else:
                    geom = make_valid(geom)

                gdf_input.at[idx, 'geometry'] = geom

        print(f"正在处理第 {idx} 个图斑，面积为 {area:.2f}")
    return gdf_input


def clip_and_merge(gdf_range, gdf_input):
    """
    参数说明:
    范围线路径: 范围线 shapefile 路径 (单多边形)
    输入shp路径: 输入图斑的 shapefile 路径
    输出路径: 结果 shapefile 的输出路径
    """

    # 确保坐标系一致
    if gdf_range.crs != gdf_input.crs:
        gdf_range = gdf_range.to_crs(gdf_input.crs)

    gdf_input['geometry'] = gdf_input.geometry.apply(
        lambda geom: make_valid(geom) if not geom.is_valid else geom
    )
    gdf_input = gdf_input[gdf_input.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]
    # 合并所有输入图斑（用于差异计算）
    merged_input = unary_union(gdf_input.geometry)

    # 用范围线减去输入图斑
    result_geom = gdf_range.geometry.unary_union.difference(merged_input)

    # 拆分剩余区域为单个图斑
    diff_gdf = gpd.GeoDataFrame(geometry=list(result_geom.geoms), crs=gdf_range.crs)
    diff_gdf["type"] = "WCL"

    # 合并差异图斑和原始输入
    final_gdf = gpd.GeoDataFrame(
        pd.concat([gdf_input, diff_gdf], ignore_index=True),
        crs=gdf_range.crs
    )

    return final_gdf


def tif_bounds_to_range(tif_path):
    """
    从 GeoTIFF 文件中提取范围边界并导出为 Shapefile。

    参数:
        tif_path (str): GeoTIFF 文件路径
        output_shp (str): 输出 Shapefile 路径（不带 .shp 后缀）
        dst_crs (str): 目标坐标参考系统，默认 EPSG:4326（WGS84）
    """
    with rasterio.open(tif_path) as src:
        # 获取原始 CRS 和图像范围
        bounds = src.bounds
        src_crs = src.crs

        # 创建矩形多边形（范围线）
        polygon = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

    # 构建 GeoDataFrame
    gdf = gpd.GeoDataFrame(
        index=[0],
        crs=src_crs,
        geometry=[polygon]
    )
    return gdf


def get_intersect_area(gdf_input, geom, i):
    neighbor_geom = gdf_input.iloc[i].geometry
    infer_geom = geom.intersection(neighbor_geom)
    return infer_geom.area


def merge_adjacent_patches(gdf_input):
    """
    合并相邻细小图斑_按照面积
    :param gdf_input:
    :return:
    """
    # 破碎
    gdf_input = gdf_input.explode(index_parts=False).reset_index(drop=True)

    # 获取所有面积小于20的图斑
    gdf_input['area_1'] = gdf_input.geometry.area
    # 计算面积并添加为新列
    gdf_input['area_1'] = gdf_input.geometry.area

    small_areas = gdf_input[gdf_input['area_1'] < 40]
    small_areas = small_areas.sort_values('area_1', ascending=True)

    read_ids = set()

    indices_to_drop = []
    # 遍历所有小面积图斑，尝试将其合并到相邻的更大幅员图斑中
    for idx, small_row in small_areas.iterrows():
        read_ids.add(idx)
        small_row = gdf_input.iloc[idx].copy()
        geom = small_row.geometry
        geom_buffer = geom.buffer(0.001)
        possible_matches_index = list(gdf_input.sindex.intersection(geom_buffer.bounds))
        intersecting_indices = [
            i for i in possible_matches_index
            if gdf_input.iloc[i].geometry.intersects(geom_buffer) and not (i == idx) and not i in read_ids
        ]
        if intersecting_indices == []:
            continue
        neighbor_idx = max(intersecting_indices, key=lambda i: get_intersect_area(gdf_input, geom_buffer, i))

        # 寻找最近邻或相交的较大图斑
        neighbor_geom = gdf_input.iloc[neighbor_idx].geometry

        # 再次寻找相交图斑
        intersecting_indices_1 = [
            i for i in possible_matches_index
            if gdf_input.iloc[i].geometry.intersects(geom_buffer) and not (i == neighbor_idx)
               and not (i == idx)
               and not i in read_ids
        ]

        if len(intersecting_indices_1) > 0:
            for i in intersecting_indices_1:
                ll = gdf_input.iloc[i].geometry
                geom_buffer = geom_buffer.difference(ll)

        new_geom = unary_union([neighbor_geom, geom_buffer])

        if new_geom.is_valid:
            gdf_input.at[neighbor_idx, 'geometry'] = new_geom
        else:
            gdf_input.at[neighbor_idx, 'geometry'] = make_valid(new_geom)
        indices_to_drop.append(idx)

    # 删除已经合并的小图斑
    # gdf_input = gdf_input.drop(columns=['area_1'])
    gdf_input = gdf_input.drop(indices_to_drop).reset_index(drop=True)
    gdf_input = gdf_input.explode(index_parts=False).reset_index(drop=True)

    return gdf_input


def fix_overlapping_polygons(gdf):
    """
    确保GeoDataFrame中的多边形没有相交（可共边）
    :param gdf: 输入的GeoDataFrame（必须包含Polygon/MultiPolygon）
    :return: 修复后的GeoDataFrame
    """
    # 添加面积字段并按面积降序排序
    fixed_gdf = gdf.copy()
    fixed_gdf['area'] = fixed_gdf.geometry.area
    sorted_gdf = fixed_gdf.sort_values(by='area', ascending=False).reset_index(drop=True)

    for idx in range(len(sorted_gdf)):
        current_geom = sorted_gdf.iloc[idx].geometry

        # 跳过空几何对象
        if current_geom.is_empty:
            continue

        # 使用空间索引加速查询
        possible_matches = list(sorted_gdf.sindex.query(current_geom, predicate='intersects'))

        for intersecting_idx in possible_matches:
            # 跳过自身或较小的图斑（因为已按面积排序）
            if intersecting_idx <= idx:
                continue

            # 获取相交图斑的几何
            intersecting_geom = sorted_gdf.iloc[intersecting_idx].geometry

            # 计算差集
            if current_geom.intersects(intersecting_geom):
                diff_geom = current_geom.difference(intersecting_geom)

                # 更新当前几何（如果差集为空则保留原几何）
                if not diff_geom.is_empty:
                    current_geom = diff_geom

        # 更新回DataFrame
        sorted_gdf.at[sorted_gdf.index[idx], 'geometry'] = current_geom

    return sorted_gdf


def gdf_type_n_merge(gdf):
    """
    提取 type 字段为 None 的图斑，遍历并使用 0.01 缓冲区判断是否相交，相交则合并。

    参数:
        gdf (GeoDataFrame): 输入的 GeoDataFrame，包含多边形几何对象和 'type' 字段。

    返回:
        GeoDataFrame: 合并后的 GeoDataFrame。
    """
    # 提取 'type' 字段为 None 的图斑
    gdf_none = gdf[gdf['type'].isna()].copy()

    gdf_fei = gdf[~gdf['type'].isna()].copy()

    if gdf_none.empty:
        return gdf

    # 添加缓冲区
    gdf_none['buffered'] = gdf_none.geometry.buffer(0.01)

    # 初始化已处理索引集合和结果列表
    processed = set()
    merged_geometries = []

    # 遍历每个图斑
    for idx, row in gdf_none.iterrows():
        if idx in processed:
            continue

        buffer_geom = row['buffered']
        group = [row.geometry]
        to_process = {idx}

        while to_process:
            current_idx = to_process.pop()
            if current_idx in processed:
                continue
            processed.add(current_idx)

            # 查询可能相交的图斑
            possible_matches = gdf_none[gdf_none['buffered'].intersects(buffer_geom)]
            for neighbor_idx, neighbor_row in possible_matches.iterrows():
                if neighbor_idx != current_idx and neighbor_idx not in processed:
                    neighbor_buffer = neighbor_row['buffered']
                    if buffer_geom.intersects(neighbor_buffer):
                        group.append(neighbor_row.geometry)
                        to_process.add(neighbor_idx)
                        buffer_geom = neighbor_buffer  # 更新缓冲区以扩展搜索范围

        # 合并该组几何
        merged_geometry = unary_union(group)

        # 再次查找相交图斑
        possible_matches = gdf_fei[gdf_fei['geometry'].intersects(buffer_geom)]

        for neighbor_idx, neighbor_row in possible_matches.iterrows():
            if neighbor_idx != current_idx and neighbor_idx not in processed:
                neighbor_buffer = neighbor_row['geometry']
                if buffer_geom.intersects(neighbor_buffer):
                    merged_geometry = merged_geometry.difference(neighbor_buffer)

        merged_geometries.append(merged_geometry)

    # 构建结果 GeoDataFrame
    merged_gdf = gpd.GeoDataFrame(geometry=merged_geometries, crs=gdf.crs)
    merged_gdf['type'] = None  # 继承原始字段值

    merged_gdf['geometry'] = merged_gdf.geometry.apply(
        lambda geom: make_valid(geom) if not geom.is_valid else geom
    )

    # 将合并后的 None 类型图斑与原数据中非 None 图斑合并
    non_none_gdf = gdf[~gdf.index.isin(gdf_none.index)]

    final_gdf = pd.concat([non_none_gdf, merged_gdf], ignore_index=True)

    return final_gdf


def remove_fully_contained_features(gdf):
    """
    删除完全被其他图斑包含的图斑

    参数:
        gdf (geopandas.GeoDataFrame): 输入的地理数据框架

    返回:
        geopandas.GeoDataFrame: 处理后仅包含非完全被包含图斑的数据框架
    """
    # 复制输入数据以免修改原始数据
    processed_gdf = gdf.copy()

    # 添加唯一ID用于跟踪原始要素
    processed_gdf['original_idx'] = processed_gdf.index

    # 创建空间索引以加速空间查询
    spatial_index = processed_gdf.sindex

    # 存储需要删除的索引
    idx_to_remove = set()
    processed_gdf['area'] = processed_gdf.geometry.area

    # 按面积升
    sorted_gdf = processed_gdf.sort_values(by='area', ascending=True)

    for idx, row in sorted_gdf.iterrows():
        # 跳过已标记删除的要素
        if idx in idx_to_remove:
            continue

        # 获取当前要素边界
        bounds = row.geometry.bounds
        # 使用空间索引查询可能的候选要素
        possible_matches_index = list(spatial_index.intersection(bounds))
        possible_matches = processed_gdf.iloc[possible_matches_index]

        # 检查每个候选要素
        for match_idx, match_row in possible_matches.iterrows():
            # 跳过自身
            if match_idx == idx:
                continue
            geo_in = row.geometry.intersection(match_row.geometry)
            if geo_in.is_empty:
                continue
            if geo_in.area >= row.geometry.area:
                idx_to_remove.add(idx)

            if geo_in.area >= match_row.geometry.area:
                idx_to_remove.add(match_idx)

    # 删除被完全包含的图斑
    result_gdf = processed_gdf[~processed_gdf.index.isin(idx_to_remove)]

    # 清理临时列并重置索引
    result_gdf = result_gdf.drop(columns=['original_idx']).reset_index(drop=True)

    return result_gdf


def rationalization(geo_gdf):
    geo_gdf = geo_gdf.explode(index_parts=True).reset_index(drop=True)
    geo_gdf = geo_gdf[geo_gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
    return geo_gdf


def remove_long_narrow(gdf):
    gdf_copy = gdf.copy()
    gdf_copy = rationalization(gdf_copy)

    retult_rows = []

    for idx in range(len(gdf_copy)):
        row = gdf_copy.iloc[idx].copy()
        row_geo = row.geometry
        geo_buffer = row.geometry.buffer(-0.3)
        if geo_buffer.geom_type == "MultiPolygon":
            geos = geo_buffer.geoms
            for geo in geos:
                new_row = row.copy()
                new_row['geometry'] = geo.buffer(0.3, cap_style="flat", join_style="mitre")
                retult_rows.append(new_row)
        else:
            new_row = row.copy()
            new_row['geometry'] = row_geo
            retult_rows.append(new_row)

    return gpd.GeoDataFrame(retult_rows, columns=gdf.columns, crs=gdf.crs)
