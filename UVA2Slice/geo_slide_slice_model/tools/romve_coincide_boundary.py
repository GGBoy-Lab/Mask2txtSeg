import math
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
import numpy as np


def romve_coincide_boundary(gdf):
    """
    调整面要素中的锐角顶点，优先尝试a点的垂直线，不行再尝试c点的垂直线

    参数:
    gdf -- GeoDataFrame，包含面要素数据

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
                # 处理外环
                adjusted_exterior = process_ring(polygon.exterior.coords[:])

                if adjusted_exterior is not None:

                    # 处理所有内环
                    adjusted_interiors = []
                    for interior in polygon.interiors:
                        coords = interior.coords[:]
                        adjusted_interiors.append(Polygon(process_ring(coords)))
                        # adjusted_interiors.append((interior.coords[:], angle_threshold))

                    try:
                        if len(adjusted_interiors) == 0:
                            adjusted_polygons.append(Polygon(adjusted_exterior, adjusted_interiors))
                        else:
                            polygon = Polygon(adjusted_exterior)
                            mult_hole = MultiPolygon(adjusted_interiors)
                            polygon = polygon.difference(mult_hole)
                            adjusted_polygons.append(polygon)
                    except:
                        print("dfdfdf")
                        raise

            if not adjusted_polygons == []:
                # 更新几何
                try:
                    if geom.geom_type == 'MultiPolygon':
                        new_geom = MultiPolygon(adjusted_polygons)
                    else:
                        new_geom = adjusted_polygons[0]

                    result_gdf.at[idx, 'geometry'] = new_geom
                except:
                    print("dfdfdf")
                    raise

    return result_gdf


def process_ring(coords):
    """
    处理单个环（外环或内环）
    """
    # 移除最后一个重复点（闭合点）
    points = coords[:-1] if len(coords) > 1 and coords[0] == coords[-1] else coords
    num_points = len(points)

    if num_points < 3:
        return points

    adjusted_points = list(points)
    indices = []

    for i in range(num_points):
        a = adjusted_points[i]
        b = adjusted_points[(i + 1) % num_points]
        c = adjusted_points[(i + 2) % num_points]

        if are_point_collinear(a, b, c, 0.000001):
            indices.append((i + 1) % num_points)

    for index in sorted(indices, reverse=True):
        del adjusted_points[index]

    if len(adjusted_points) < 3:
        return None

    # 添加闭合点
    return adjusted_points + [adjusted_points[0]]


def are_point_collinear(p1, p2, p3, tolerance):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)

    ab = b - a
    ac = c - a
    ab_length = np.linalg.norm(ab)
    if ab_length < tolerance:
        return True

    proj_length = np.dot(ac, ab) / ab_length

    projection = a + (ab / ab_length) * proj_length

    distance = np.linalg.norm(c - projection)

    is_between = 0 <= proj_length <= ab_length

    return distance < tolerance and is_between


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
