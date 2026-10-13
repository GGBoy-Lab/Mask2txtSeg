import math

import geopandas as gpd
import numpy as np
from shapely import Polygon, MultiPolygon, make_valid


def adjust_sharp_angles(gdf, angle_threshold=30):
    """
    调整面要素中的锐角顶点，优先尝试a点的垂直线，不行再尝试c点的垂直线

    参数:
    gdf -- GeoDataFrame，包含面要素数据
    angle_threshold -- 角度阈值（度），默认30度

    返回:
    新的GeoDataFrame，包含调整后的几何
    """
    # 创建结果的深拷贝
    result_gdf = gdf.copy()

    # 处理每个面要素
    for idx, row in gdf.iterrows():
        geom = row.geometry

        # 处理多边形和多多边形
        if geom.geom_type in ['Polygon', 'MultiPolygon']:
            # 处理多多边形中的每个多边形
            if geom.geom_type == 'MultiPolygon':
                polygons = list(geom.geoms)
            else:
                polygons = [geom]

            adjusted_polygons = []

            for polygon in polygons:
                try:
                    # 处理外环
                    adjusted_exterior = process_ring(polygon.exterior.coords[:], angle_threshold)
                    # 处理所有内环
                    adjusted_interiors = []
                    for interior in polygon.interiors:
                        coords = interior.coords[:]
                        adjusted_interiors.append(Polygon(process_ring(coords, angle_threshold)))
                    if len(adjusted_interiors) == 0:
                        adjusted_polygons.append(Polygon(adjusted_exterior, adjusted_interiors))
                    else:
                        polygon = Polygon(adjusted_exterior)
                        if not polygon.is_valid:
                            polygon = polygon.buffer(0.0001)
                            polygon = polygon.buffer(-0.0001)
                            polygon = make_valid(polygon)

                        mult_hole = MultiPolygon(adjusted_interiors)

                        if not mult_hole.is_valid:
                            mult_hole = mult_hole.buffer(0.0001)
                            mult_hole = mult_hole.buffer(-0.0001)
                        polygon = polygon.difference(mult_hole)
                        adjusted_polygons.append(polygon)
                except Exception:
                    print("dfdfdf")
                    raise

            # 更新几何
            if not adjusted_polygons == []:
                if geom.geom_type == 'MultiPolygon':
                    new_geom = MultiPolygon(adjusted_polygons)
                else:
                    new_geom = adjusted_polygons[0]

                result_gdf.at[idx, 'geometry'] = new_geom

    return result_gdf


def process_ring(coords, angle_threshold):
    """
    处理单个环（外环或内环）
    """
    # 移除最后一个重复点（闭合点）
    points = coords[:-1] if len(coords) > 1 and coords[0] == coords[-1] else coords
    num_points = len(points)

    if num_points < 3:
        return points

    adjusted_points = list(points)
    changed = True

    while changed:
        changed = False
        i = 0

        while i < num_points:
            a = adjusted_points[i]
            b = adjusted_points[(i + 1) % num_points]
            c = adjusted_points[(i + 2) % num_points]

            angle = calculate_angle(a, b, c)

            if angle < angle_threshold:
                H_a = calculate_foot_from_point_to_edge(a, b, c)

                if H_a is not None:
                    adjusted_points[(i + 1) % num_points] = H_a
                    changed = True

                    i += 1
                else:
                    H_c = calculate_foot_from_point_to_edge(c, a, b)

                    if H_c is not None:
                        adjusted_points[(i + 1) % num_points] = H_c
                        changed = True
                        i += 1

            i += 1

    # 添加闭合点
    return adjusted_points + [adjusted_points[0]]


def calculate_angle(a, b, c):
    """
    计算三点之间在b点的夹角（度）
    """
    # 向量BA和BC
    BA = (a[0] - b[0], a[1] - b[1])
    BC = (c[0] - b[0], c[1] - b[1])

    dot_product = BA[0] * BC[0] + BA[1] * BC[1]

    len_BA = math.sqrt(BA[0] ** 2 + BA[1] ** 2)
    len_BC = math.sqrt(BC[0] ** 2 + BC[1] ** 2)

    # 避免除零错误
    if len_BA == 0 or len_BC == 0:
        return 180

    # 计算夹角的余弦值
    cos_theta = dot_product / (len_BA * len_BC)

    # 处理余弦值超出范围的情况
    cos_theta = max(-1.0, min(cos_theta, 1.0))

    # 转换为角度（0-180度）
    angle_rad = math.acos(cos_theta)
    angle_deg = math.degrees(angle_rad)

    return angle_deg


def calculate_foot_from_point_to_edge(point, edge_start, edge_end):
    """
    计算从一个点到一条边的垂足
    返回垂足坐标（如果垂足在边段上），否则返回None
    """
    P = np.array(point)
    A = np.array(edge_start)
    B = np.array(edge_end)

    AB = B - A

    len_ab_sq = np.dot(AB, AB)

    if len_ab_sq == 0:
        return None

    AP = P - A
    t = np.dot(AP, AB) / len_ab_sq

    if 0.0 <= t <= 1.0:
        # 计算垂足坐标
        H = A + t * AB
        return (H[0], H[1])

    return None


# 使用示例
if __name__ == "__main__":
    # 读取原始SHP文件
    input_shp = "path/to/your/input_shapefile.shp"
    gdf = gpd.read_file(input_shp)

    # 处理锐角
    result_gdf = adjust_sharp_angles(gdf, angle_threshold=30)

    # 保存结果
    output_shp = "path/to/your/output_shapefile.shp"
    result_gdf.to_file(output_shp)
