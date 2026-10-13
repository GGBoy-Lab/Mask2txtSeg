import geopandas as gpd
import numpy as np
from shapely.ops import snap, unary_union
from shapely.validation import make_valid
from shapely.geometry import LineString
from shapely.geometry import Polygon, MultiPolygon


def get_exterior_points(geom):
    """
    提取几何对象的外环顶点坐标
    支持 Polygon 和 MultiPolygon
    返回 list of (x, y) tuples
    """
    points = []

    if isinstance(geom, Polygon):
        points.extend(list(geom.exterior.coords))
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            points.extend(get_exterior_points(poly))
    else:
        raise ValueError(f"Unsupported geometry type: {type(geom)}")

    return points


# 使用示例
if __name__ == "__main__":
    # 1. 加载数据

    input_shp = 'F:\工作文件\目标识别与分割\绿茵湖2\out\output_mask-1748922601.5845938.shp'  # 替换为你的输入文件路径
    output_shp = r'F:\工作文件\目标识别与分割\绿茵湖2\out\8.shp'  # 输出文件路径

    gdf = gpd.read_file(input_shp)
    gdf = gdf.to_crs("EPSG:4524")
    buffer_tolerance = 2
    gdf_buffer = gdf.copy()
    gdf_buffer["geometry"] = gdf.buffer(buffer_tolerance)
    # 2. 自动计算容差
    tolerance = 2

    for index, row in gdf_buffer.iterrows():
        geom_buffer = row['geometry']
        geom = gdf.iloc[index].geometry
        geom = geom.simplify(tolerance=0.5)

        if geom.is_valid:
            geom = geom
        else:
            geom = make_valid(geom)
        # 获取该面的外环所有点
        # exterior_points = get_exterior_points(geom)
        # for point in exterior_points:
        #     # 找最近的图斑，排除当前图斑
        #     gdf["point_distance"] = gdf.geometry.distance(point)
        #     # 排除当前图斑
        #     filtered_gdf = gdf[gdf.index != index]
        #     min_distance = filtered_gdf["point_distance"].min()
        #     if min_distance < 5:
        #         nearest_index = gdf["point_distance"].idxmin()
        #         nearest_polygon = gdf.loc[nearest_index].geometry
        #         print(f"最近图斑索引: {nearest_index}")

        # 使用空间索引查找潜在相交的图斑索引
        possible_matches_index = list(gdf_buffer.sindex.intersection(geom_buffer.bounds))
        # 在这些候选中进一步判断是否真正相交
        intersecting_indices = [
            i for i in possible_matches_index
            if gdf_buffer.iloc[i].geometry.intersects(geom_buffer) and not (i == index)
        ]
        print(f"图斑 {index} 相交的图斑索引: {intersecting_indices}")
        # 边界吸附
        reference_geom_list = []
        for intersecting_index in intersecting_indices:
            reference_geom = gdf.iloc[intersecting_index].geometry
            reference_geom_list.append(reference_geom)
        feature = {
            "geometry": geom,
            "properties": {}
        }
        if reference_geom_list:
            s = gpd.GeoDataFrame([feature])
            s2 = gpd.GeoSeries(reference_geom_list)
            s = s.snap(s2, 1, align=True)
            # geom_after = snap(geom, reference_geom, tolerance)
            ll = s.iloc[0]
            gdf.loc[index, 'geometry'] = ll

    # 3. 执行边界对齐
    # aligned_gdf = align_polygon_edges(gdf, tolerance)

    # 4. 保存结果
    gdf.to_file(output_shp)
