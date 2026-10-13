import geopandas as gpd
import pandas as pd
from shapely import make_valid, union_all
from shapely.ops import unary_union
import os

from tools.shp_utils import remove_small_polygons


def clip_and_merge(fwx_path, gdf_input):
    """
    参数说明:
    范围线路径: 范围线 shapefile 路径 (单多边形)
    输入shp路径: 输入图斑的 shapefile 路径
    输出路径: 结果 shapefile 的输出路径
    """
    # 读取数据
    gdf_range = gpd.read_file(fwx_path)  # 范围线

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


def aa(gdf_input):
    # gdf_input = gdf_input.explode(index_parts=False).reset_index(drop=True)
    # 创建空间索引以便快速查询邻接关系
    # gdf_input = gdf_input.to_crs("4524")

    # 获取所有面积小于20的图斑
    gdf_input['area'] = gdf_input.geometry.area
    small_areas = gdf_input[gdf_input['area'] < 40]
    indices_to_drop = []
    # 遍历所有小面积图斑，尝试将其合并到相邻的更大幅员图斑中
    for idx, small_row in small_areas.iterrows():
        geom = small_row.geometry
        geom_buffer = geom.buffer(0.1)
        possible_matches_index = list(gdf_input.sindex.intersection(geom_buffer.bounds))
        intersecting_indices = [
            i for i in possible_matches_index
            if gdf_input.iloc[i].geometry.intersects(geom_buffer) and not (i == idx)
        ]
        neighbor_idx = max(intersecting_indices, key=lambda i: gdf_input.iloc[i]['area'])
        # 寻找最近邻或相交的较大图斑

        neighbor_geom = gdf_input.iloc[neighbor_idx].geometry

        # 再次寻找相交图斑
        intersecting_indices_1 = [
            i for i in possible_matches_index
            if gdf_input.iloc[i].geometry.intersects(geom_buffer) and not (i == neighbor_idx)
               and not (i == idx)
        ]

        if len(intersecting_indices_1) > 0:
            for i in intersecting_indices_1:
                ll = gdf_input.iloc[i].geometry
                geom_buffer = geom_buffer.difference(ll)

        new_geom = unary_union([neighbor_geom, geom])

        if new_geom.is_valid:
            gdf_input.at[neighbor_idx, 'geometry'] = new_geom
        else:
            gdf_input.at[neighbor_idx, 'geometry'] = make_valid(new_geom)
        indices_to_drop.append(idx)

    # 删除已经合并的小图斑
    gdf_input = gdf_input.drop(indices_to_drop).reset_index(drop=True)
    gdf_input = gdf_input.explode(index_parts=False).reset_index(drop=True)

    return gdf_input

def bb(gdf_input):
    # gdf_input = gpd.read_file(gdf_input)
    # gdf_input = gdf_input.to_crs("4524")

    gdf_input = remove_small_polygons(gdf_input, 30)

    gdf_input["geometry"] = gdf_input.buffer(0.5)
    gdf_input["geometry"] = gdf_input.simplify(tolerance=1)

    # 计算面积并添加为新列
    gdf_input['area'] = gdf_input.geometry.area
    # 按面积从小到大排序
    gdf_input = gdf_input.sort_values('area').reset_index(drop=True)

    # 遍历图斑（从小到大）
    for idx, row in gdf_input.iterrows():
        geom = row['geometry']
        geom = geom.buffer(0.5)
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


# 使用示例
if __name__ == "__main__":
    fwx_path = r"F:\工作文件\目标识别与分割\范围线.shp"
    shp_path = r"F:\工作文件\目标识别与分割\绿茵湖2\out8\output_mask-1749096321.3874753.shp"
    reture_path = r"F:\工作文件\目标识别与分割\绿茵湖2\out8\第三步处理3.shp"
    gdf_input = gpd.read_file(shp_path)
    gdf_input = gdf_input.to_crs(4524)
    # 第一次处理，初步共边
    gdf_input = bb(gdf_input)

    gdf_input["geometry"] = gdf_input["geometry"].buffer(0.1)
    gdf_input["geometry"] = gdf_input["geometry"].buffer(-0.1)

    gdf_input = clip_and_merge(fwx_path, gdf_input)

    # 对于共边的二次处理 - 合并面积小于20的与相邻图斑
    # gdf_input = gdf_input[gdf_input.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]

    gdf_input = aa(gdf_input)
    # gdf_input["geometry"] = gdf_input["geometry"].buffer(0.1)
    # gdf_input["geometry"] = gdf_input["geometry"].buffer(-0.1)
    gdf_input = aa(gdf_input)

    gdf_input = gdf_input[gdf_input.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]

    gdf_input.to_file(reture_path, encoding='UTF-8')
    print(f"处理完成! 结果已保存至: {reture_path}")
