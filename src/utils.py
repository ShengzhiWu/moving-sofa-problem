import time
import numpy as np
from PIL import Image
import taichi as ti
from scipy.ndimage import zoom
import os
import inspect

def get_parameter_count(f):  # 获取函数参数数量
    sig = inspect.signature(f)
    return len(sig.parameters)

def resample_polyline(polyline, n):  # 在一条折线上等弧长间距地分布若干个点
    segments = np.diff(polyline, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)  # 各段长度
    total_length = np.sum(segment_lengths)  # 折线总长度
    cumulative_lengths = np.concatenate(([0], np.cumsum(segment_lengths)))  # 每个分段末端的累计距离
    target_distances = np.linspace(0, total_length, n)

    result = [polyline[0]]
    for i, distance in enumerate(target_distances):
        if i == 0:
            continue
        segment_idx = np.searchsorted(cumulative_lengths, distance) - 1  # 目标距离所在的分段
        segment_distance = distance - cumulative_lengths[segment_idx]  # 在该分段内的距离
        t = segment_distance / segment_lengths[segment_idx] if segment_lengths[segment_idx] > 0 else 0  # 在分段上的位置
        result.append(polyline[segment_idx] + t * segments[segment_idx])

    return np.array(result)

@ti.kernel
def normalize_quaternions(quaternion_field: ti.template()):  # 归一化四元数 # type: ignore
    for i in range(quaternion_field.shape[0]):
        w = quaternion_field[i, 0]
        x = quaternion_field[i, 1]
        y = quaternion_field[i, 2]
        z = quaternion_field[i, 3]
        norm = max(ti.sqrt(w * w + x * x + y * y + z * z), 1e-8)  # 防止除以零
        quaternion_field[i, 0] = w / norm
        quaternion_field[i, 1] = x / norm
        quaternion_field[i, 2] = y / norm
        quaternion_field[i, 3] = z / norm

def quaternion_to_rotation_matrix(quaternion):
    w, x, y, z = quaternion
    rotation_matrix = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
    ], dtype=np.float32)
    return rotation_matrix

def rotation_matrix_to_quaternion(rotation_matrix):
    m = rotation_matrix
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z], dtype=np.float32)

@ti.kernel
def quaternions_to_rotation_matrices(quaternion_field: ti.template(), rotation_matrix_field: ti.template()):  # 四元数转旋转矩阵 # type: ignore
    for i in range(quaternion_field.shape[0]):
        w = quaternion_field[i, 0]
        x = quaternion_field[i, 1]
        y = quaternion_field[i, 2]
        z = quaternion_field[i, 3]

        rotation_matrix_field[i][0, 0] = 1 - 2 * (y * y + z * z)
        rotation_matrix_field[i][0, 1] = 2 * (x * y - z * w)
        rotation_matrix_field[i][0, 2] = 2 * (x * z + y * w)

        rotation_matrix_field[i][1, 0] = 2 * (x * y + z * w)
        rotation_matrix_field[i][1, 1] = 1 - 2 * (x * x + z * z)
        rotation_matrix_field[i][1, 2] = 2 * (y * z - x * w)

        rotation_matrix_field[i][2, 0] = 2 * (x * z - y * w)
        rotation_matrix_field[i][2, 1] = 2 * (y * z + x * w)
        rotation_matrix_field[i][2, 2] = 1 - 2 * (x * x + y * y)

@ti.func
def transform_local_to_world(
    local_x: ti.f32,  # type: ignore
    local_y: ti.f32,  # type: ignore
    x: ti.f32,  # type: ignore
    y: ti.f32,  # type: ignore
    theta: ti.f32  # type: ignore
):
    c = ti.cos(theta)
    s = ti.sin(theta)
    world_x = c * local_x - s * local_y + x
    world_y = s * local_x + c * local_y + y
    return world_x, world_y

@ti.func
def transform_local_to_world_3d(
    local_x: ti.f32,  # type: ignore
    local_y: ti.f32,  # type: ignore
    local_z: ti.f32,  # type: ignore
    x: ti.f32,  # type: ignore
    y: ti.f32,  # type: ignore
    z: ti.f32,  # type: ignore
    rotation_matrix  # 3 * 3 matrix
):
    world_x = rotation_matrix[0, 0] * local_x + rotation_matrix[0, 1] * local_y + rotation_matrix[0, 2] * local_z + x
    world_y = rotation_matrix[1, 0] * local_x + rotation_matrix[1, 1] * local_y + rotation_matrix[1, 2] * local_z + y
    world_z = rotation_matrix[2, 0] * local_x + rotation_matrix[2, 1] * local_y + rotation_matrix[2, 2] * local_z + z
    return world_x, world_y, world_z

@ti.kernel
def compute_survive_domain(x_min: float, x_max: float, y_min: float, y_max: float, x_field: ti.template(), y_field: ti.template(), rotation_field: ti.template(), survive_mask: ti.template()):  # 根据轨迹计算能通过的图形 # type: ignore
    for i, j in survive_mask:
        survive_mask[i, j] = -1  # -1表示幸存

        # local coords
        local_x = (i + 0.5) / survive_mask.shape[0] * (x_max - x_min) + x_min
        local_y = (j + 0.5) / survive_mask.shape[1] * (y_max - y_min) + y_min
        for t in range(x_field.shape[0]):
            world_x, world_y = transform_local_to_world(local_x, local_y, x_field[t], y_field[t], rotation_field[t])
            # check forbidden region
            if is_forbidden(world_x, world_y):
                survive_mask[i, j] = t  # 非负整数表示未幸存，值为这个像素被削掉的时刻
                break

@ti.kernel
def compute_survive_domain_3d(x_min: float, x_max: float, y_min: float, y_max: float, z_min: float, z_max: float, x_field: ti.template(), y_field: ti.template(), z_field: ti.template(), rotation_matrix_field: ti.template(), survive_mask: ti.template()):  # 根据轨迹计算能通过的图形 # type: ignore
    for i, j, k in survive_mask:
        survive_mask[i, j, k] = -1  # -1表示幸存

        # local coords
        local_x = (i + 0.5) / survive_mask.shape[0] * (x_max - x_min) + x_min
        local_y = (j + 0.5) / survive_mask.shape[1] * (y_max - y_min) + y_min
        local_z = (k + 0.5) / survive_mask.shape[2] * (z_max - z_min) + z_min
        for t in range(x_field.shape[0]):
            world_x, world_y, world_z = transform_local_to_world_3d(local_x, local_y, local_z, x_field[t], y_field[t], z_field[t], rotation_matrix_field[t])
            # check forbidden region
            if is_forbidden(world_x, world_y, world_z):
                survive_mask[i, j, k] = t  # 非负整数表示未幸存，值为这个像素被削掉的时刻
                break

@ti.kernel
def compute_survive_domain_2(x_min: float, x_max: float, y_min: float, y_max: float, x_field: ti.template(), y_field: ti.template(), rotation_field: ti.template(), survive_mask: ti.template()):  # 根据轨迹计算能通过的图形，其中进入墙壁的判定函数需要输入相邻时间步的两个点 # type: ignore
    for i, j in survive_mask:
        survive_mask[i, j] = -1  # -1表示幸存

        # local coords
        local_x = (i + 0.5) / survive_mask.shape[0] * (x_max - x_min) + x_min
        local_y = (j + 0.5) / survive_mask.shape[1] * (y_max - y_min) + y_min
        world_x, world_y = transform_local_to_world(local_x, local_y, x_field[0], y_field[0], rotation_field[0])
        for t in range(1, x_field.shape[0]):
            world_x_new, world_y_new = transform_local_to_world(local_x, local_y, x_field[t], y_field[t], rotation_field[t])
            # check forbidden region
            if is_forbidden(world_x, world_y, world_x_new, world_y_new):
                survive_mask[i, j] = t  # 非负整数表示未幸存，值为这个像素被削掉的时刻
                break
            world_x, world_y = world_x_new, world_y_new

@ti.kernel
def compute_survive_domain_3d_2(x_min: float, x_max: float, y_min: float, y_max: float, z_min: float, z_max: float, x_field: ti.template(), y_field: ti.template(), z_field: ti.template(), rotation_matrix_field: ti.template(), survive_mask: ti.template()):  # 根据轨迹计算能通过的图形，其中进入墙壁的判定函数需要输入相邻时间步的两个点 # type: ignore
    for i, j, k in survive_mask:
        survive_mask[i, j, k] = -1  # -1表示幸存

        # local coords
        local_x = (i + 0.5) / survive_mask.shape[0] * (x_max - x_min) + x_min
        local_y = (j + 0.5) / survive_mask.shape[1] * (y_max - y_min) + y_min
        local_z = (k + 0.5) / survive_mask.shape[2] * (z_max - z_min) + z_min
        world_x, world_y, world_z = transform_local_to_world_3d(local_x, local_y, local_z, x_field[0], y_field[0], z_field[0], rotation_matrix_field[0])
        for t in range(1, x_field.shape[0]):
            world_x_new, world_y_new, world_z_new = transform_local_to_world_3d(local_x, local_y, local_z, x_field[t], y_field[t], z_field[t], rotation_matrix_field[t])
            # check forbidden region
            if is_forbidden(world_x, world_y, world_z, world_x_new, world_y_new, world_z_new):
                survive_mask[i, j, k] = t  # 非负整数表示未幸存，值为这个像素被削掉的时刻
                break
            world_x, world_y, world_z = world_x_new, world_y_new, world_z_new

def compute_survive(sofa_w, sofa_h, x_field, y_field, rotation_field, survive_mask, center_x=0, center_y=0):  # 根据轨迹计算能通过的图形
    compute_survive_domain(
        -sofa_w * 0.5 + center_x, sofa_w * 0.5 + center_x,
        -sofa_h * 0.5 + center_y, sofa_h * 0.5 + center_y,
        x_field, y_field, rotation_field, survive_mask
    )

def compute_survive_3d(sofa_w, sofa_h, sofa_d, x_field, y_field, z_field, rotation_matrix_field, survive_mask, center_x=0, center_y=0, center_z=0):  # 根据轨迹计算能通过的图形
    compute_survive_domain_3d(
        -sofa_w * 0.5 + center_x, sofa_w * 0.5 + center_x,
        -sofa_h * 0.5 + center_y, sofa_h * 0.5 + center_y,
        -sofa_d * 0.5 + center_z, sofa_d * 0.5 + center_z,
        x_field, y_field, z_field, rotation_matrix_field, survive_mask
    )

def compute_survive_2(sofa_w, sofa_h, x_field, y_field, rotation_field, survive_mask, center_x=0, center_y=0):  # 根据轨迹计算能通过的图形，其中进入墙壁的判定函数需要输入相邻时间步的两个点
    compute_survive_domain_2(
        -sofa_w * 0.5 + center_x, sofa_w * 0.5 + center_x,
        -sofa_h * 0.5 + center_y, sofa_h * 0.5 + center_y,
        x_field, y_field, rotation_field, survive_mask
    )

def compute_survive_3d_2(sofa_w, sofa_h, sofa_d, x_field, y_field, z_field, rotation_field, survive_mask, center_x=0, center_y=0, center_z=0):  # 根据轨迹计算能通过的图形，其中进入墙壁的判定函数需要输入相邻时间步的两个点
    compute_survive_domain_3d_2(
        -sofa_w * 0.5 + center_x, sofa_w * 0.5 + center_x,
        -sofa_h * 0.5 + center_y, sofa_h * 0.5 + center_y,
        -sofa_d * 0.5 + center_z, sofa_d * 0.5 + center_z,
        x_field, y_field, z_field, rotation_field, survive_mask
    )

def generate_fields(
    sofa_w = 3.5,   # 求解域的尺寸
    sofa_h = 1.0,   # 求解域的尺寸
    resolution = 512,  # 用一个位图表示沙发形状，这个参数是横向分辨率
    control_point_num = 30,  # 轨迹的控制点数量
    trajectory_upsampling = 10  # 在控制点之间插值。注意如果不在控制点之间差值将会得到不合理的结果，因为算法会尝试在控制点之间引入跃变而跳过障碍
):
    # Trajectories: 1D fields
    x_field = ti.field(dtype=ti.f32, shape=control_point_num * trajectory_upsampling)
    y_field = ti.field(dtype=ti.f32, shape=control_point_num * trajectory_upsampling)
    rotation_field = ti.field(dtype=ti.f32, shape=control_point_num * trajectory_upsampling)

    # world sampling resolution for sofa mask (we sample in sofa-local coordinates at this resolution)
    mask_w = resolution
    mask_h = int(resolution * (sofa_h / sofa_w))

    # For each sofa local pixel, mark surviving (1) or cut (0)
    survive_mask = ti.field(dtype=ti.int32, shape=(mask_w, mask_h))  # -1表示像素留存，非负整数表示像素被切削掉的时间

    return x_field, y_field, rotation_field, survive_mask

def generate_fields_3d(
    sofa_w = 3.5,   # 求解域的尺寸
    sofa_h = 1.0,   # 求解域的尺寸
    sofa_d = 1.0,   # 求解域的尺寸
    resolution = 64,  # 用一个位图表示沙发形状，这个参数是横向分辨率
    control_point_num = 30,  # 轨迹的控制点数量
    trajectory_upsampling = 10  # 在控制点之间插值。注意如果不在控制点之间差值将会得到不合理的结果，因为算法会尝试在控制点之间引入跃变而跳过障碍
):
    # Trajectories: 1D fields
    x_field = ti.field(dtype=ti.f32, shape=control_point_num * trajectory_upsampling)
    y_field = ti.field(dtype=ti.f32, shape=control_point_num * trajectory_upsampling)
    z_field = ti.field(dtype=ti.f32, shape=control_point_num * trajectory_upsampling)
    rotation_field = ti.field(dtype=ti.f32, shape=(control_point_num * trajectory_upsampling, 4))  # 四元数数组
    rotation_matrix_field = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(control_point_num * trajectory_upsampling,))  # 旋转矩阵数组

    # world sampling resolution for sofa mask (we sample in sofa-local coordinates at this resolution)
    mask_w = resolution
    mask_h = int(resolution * (sofa_h / sofa_w))
    mask_d = int(resolution * (sofa_d / sofa_w))

    # For each sofa local pixel, mark surviving (1) or cut (0)
    survive_mask = ti.field(dtype=ti.int32, shape=(mask_w, mask_h, mask_d))  # -1表示像素留存，非负整数表示像素被切削掉的时间

    return x_field, y_field, z_field, rotation_field, rotation_matrix_field, survive_mask

def get_sofa(  # 根据轨迹计算沙发形状，返回一张位图
    forbidden_function,  # 一个函数，参数数量可以是2或4，分别对应(x, y)或(x0, y0, x1, y1)
    xs,
    ys,
    rotations,
    sofa_w = 3.5,  # 求解域的尺寸
    sofa_h = 1.0,  # 求解域的尺寸
    center_x = 0,  # 画面中心对应的真实世界的坐标 
    center_y = 0,  # 画面中心对应的真实世界的坐标 
    resolution = 512,  # 用一个位图表示沙发形状，这个参数是横向分辨率
    trajectory_upsampling = 10,  # 在控制点之间插值。注意如果不在控制点之间差值将会得到不合理的结果，因为算法会尝试在控制点之间引入跃变而跳过障碍
    trajectory_upsampling_order = 1,  # 插值阶数
    return_hitting_time = False
):
    global is_forbidden
    is_forbidden = forbidden_function
    assert get_parameter_count(is_forbidden) in [2, 4]
    
    # Taichi fields
    x_field, y_field, rotation_field, survive_mask = generate_fields(sofa_w, sofa_h, resolution, len(xs), trajectory_upsampling)

    x_field.from_numpy(zoom(xs, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
    y_field.from_numpy(zoom(ys, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
    rotation_field.from_numpy(zoom(rotations, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
    if get_parameter_count(is_forbidden) == 2:
        compute_survive(sofa_w, sofa_h, x_field, y_field, rotation_field, survive_mask, center_x=center_x, center_y=center_y)
    else:
        compute_survive_2(sofa_w, sofa_h, x_field, y_field, rotation_field, survive_mask, center_x=center_x, center_y=center_y)

    survive_mask_numpy = survive_mask.to_numpy()

    if return_hitting_time:
        return survive_mask_numpy < 0, survive_mask_numpy
    else:
        return survive_mask_numpy < 0

def get_sofa_3d(  # 根据轨迹计算沙发形状，返回一张位图
    forbidden_function,  # 一个函数，参数数量可以是3或6，分别对应(x, y, z)或(x0, y0, z0, x1, y1, z1)
    xs,
    ys,
    zs,
    rotations,
    sofa_w = 1.0,  # 求解域的尺寸
    sofa_h = 1.0,  # 求解域的尺寸
    sofa_d = 1.0,  # 求解域的尺寸
    center_x = 0,  # 画面中心对应的真实世界的坐标 
    center_y = 0,  # 画面中心对应的真实世界的坐标 
    center_z = 0,  # 画面中心对应的真实世界的坐标 
    resolution = 64,  # 用一个位图表示沙发形状，这个参数是X向分辨率
    trajectory_upsampling = 10,  # 在控制点之间插值。注意如果不在控制点之间差值将会得到不合理的结果，因为算法会尝试在控制点之间引入跃变而跳过障碍
    trajectory_upsampling_order = 1,  # 插值阶数
    return_hitting_time = False
):
    global is_forbidden
    is_forbidden = forbidden_function
    assert get_parameter_count(is_forbidden) in [3, 6]
    
    # Taichi fields
    x_field, y_field, z_field, rotation_field, rotation_matrix_field, survive_mask = generate_fields_3d(sofa_w, sofa_h, sofa_d, resolution, len(xs), trajectory_upsampling)

    x_field.from_numpy(zoom(xs, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
    y_field.from_numpy(zoom(ys, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
    z_field.from_numpy(zoom(zs, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
    rotation_field.from_numpy(zoom(rotations, zoom=(trajectory_upsampling, 1), order=trajectory_upsampling_order))
    normalize_quaternions(rotation_field)
    quaternions_to_rotation_matrices(rotation_field, rotation_matrix_field)  # 四元数 -> 旋转矩阵
    if get_parameter_count(is_forbidden) == 3:
        compute_survive_3d(sofa_w, sofa_h, sofa_d, x_field, y_field, z_field, rotation_matrix_field, survive_mask, center_x=center_x, center_y=center_y, center_z=center_z)
    else:
        compute_survive_3d_2(sofa_w, sofa_h, sofa_d, x_field, y_field, z_field, rotation_matrix_field, survive_mask, center_x=center_x, center_y=center_y, center_z=center_z)

    survive_mask_numpy = survive_mask.to_numpy()

    if return_hitting_time:
        return survive_mask_numpy < 0, survive_mask_numpy
    else:
        return survive_mask_numpy < 0

@ti.kernel
def get_area_kernel(sofa_w: float, sofa_h: float, survive_mask: ti.template()) -> ti.f32:  # type: ignore
    s = 0
    for i, j in survive_mask:
        s += 1 if survive_mask[i, j] < 0 else 0
    return float(s) / survive_mask.shape[0] / survive_mask.shape[1] * sofa_w * sofa_h

@ti.kernel
def get_volume_kernel(sofa_w: float, sofa_h: float, sofa_d: float, survive_mask: ti.template()) -> ti.f32:  # type: ignore
    s = 0
    for i, j, k in survive_mask:
        s += 1 if survive_mask[i, j, k] < 0 else 0
    return float(s) / survive_mask.shape[0] / survive_mask.shape[1] / survive_mask.shape[2] * sofa_w * sofa_h * sofa_d

def get_area(
    forbidden_function,  # 一个函数，参数数量可以是2或4，分别对应(x, y)或(x0, y0, x1, y1)
    xs,
    ys,
    rotations,
    sofa_w = 3.5,   # 求解域的尺寸
    sofa_h = 1.0,   # 求解域的尺寸
    resolution = 512,  # 用一个位图表示沙发形状，这个参数是横向分辨率
    trajectory_upsampling = 10,  # 在控制点之间插值。注意如果不在控制点之间差值将会得到不合理的结果，因为算法会尝试在控制点之间引入跃变而跳过障碍
    trajectory_upsampling_order = 1  # 插值阶数
):
    global is_forbidden
    is_forbidden = forbidden_function
    assert get_parameter_count(is_forbidden) in [2, 4]
    
    # Taichi fields
    x_field, y_field, rotation_field, survive_mask = generate_fields(sofa_w, sofa_h, resolution, len(xs), trajectory_upsampling)

    x_field.from_numpy(zoom(xs, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
    y_field.from_numpy(zoom(ys, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
    rotation_field.from_numpy(zoom(rotations, zoom=trajectory_upsampling, order=trajectory_upsampling_order))

    if get_parameter_count(is_forbidden) == 2:
        compute_survive(sofa_w, sofa_h, x_field, y_field, rotation_field, survive_mask)
    else:
        compute_survive_2(sofa_w, sofa_h, x_field, y_field, rotation_field, survive_mask)

    return get_area_kernel(sofa_w, sofa_h, survive_mask)

def get_volume(
    forbidden_function,  # 一个函数，参数数量可以是3或6，分别对应(x, y, z)或(x0, y0, z0, x1, y1, z1)
    xs,
    ys,
    zs,
    rotations,
    sofa_w = 1.0,   # 求解域的尺寸
    sofa_h = 1.0,   # 求解域的尺寸
    sofa_d = 1.0,   # 求解域的尺寸
    resolution = 64,  # 用一个位图表示沙发形状，这个参数是X向分辨率
    trajectory_upsampling = 10,  # 在控制点之间插值。注意如果不在控制点之间差值将会得到不合理的结果，因为算法会尝试在控制点之间引入跃变而跳过障碍
    trajectory_upsampling_order = 1  # 插值阶数
):
    global is_forbidden
    is_forbidden = forbidden_function
    assert get_parameter_count(is_forbidden) in [3, 6]
    
    # Taichi fields
    x_field, y_field, z_field, rotation_field, rotation_matrix_field, survive_mask = generate_fields_3d(sofa_w, sofa_h, sofa_d, resolution, len(xs), trajectory_upsampling)

    x_field.from_numpy(zoom(xs, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
    y_field.from_numpy(zoom(ys, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
    z_field.from_numpy(zoom(zs, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
    rotation_field.from_numpy(zoom(rotations, zoom=(trajectory_upsampling, 1), order=trajectory_upsampling_order))
    quaternions_to_rotation_matrices(rotation_field, rotation_matrix_field)

    if get_parameter_count(is_forbidden) == 3:
        compute_survive_3d(sofa_w, sofa_h, sofa_d, x_field, y_field, z_field, rotation_matrix_field, survive_mask)
    else:
        compute_survive_3d_2(sofa_w, sofa_h, sofa_d, x_field, y_field, z_field, rotation_matrix_field, survive_mask)

    return get_volume_kernel(sofa_w, sofa_h, sofa_d, survive_mask)

def mutate(
    xs, ys, rs, mutation_sigma_pos, mutation_sigma_rotation,
    boundary_conditions='fixed'  # 边界条件，可以是'fixed'、'periodic'、'free'
):
    # 经过测试，这种变异方式进化效率极低，因为稍微一改变就很容易变差
    # TODO: 这里有待重新测试
    # new_xs = xs + np.random.randn(steps) * 0.0001
    # new_ys = ys + np.random.randn(steps) * 0.0001
    # new_rs = rs + np.random.randn(steps) * 0.0001

    # 随机挑一个控制点变异
    new_xs = xs.copy()
    new_ys = ys.copy()
    new_rs = rs.copy()
    if boundary_conditions == 'fixed':
        i = np.random.randint(1, len(xs) - 1)
    elif boundary_conditions == 'periodic':
        i = np.random.randint(0, len(xs) - 1)
    elif boundary_conditions == 'free':
        i = np.random.randint(0, len(xs))
    else:
        raise ValueError(f"Unknown boundary condition: {boundary_conditions}")
    new_xs[i] += np.random.randn() * mutation_sigma_pos
    new_ys[i] += np.random.randn() * mutation_sigma_pos
    new_rs[i] += np.random.randn() * mutation_sigma_rotation
    if boundary_conditions == 'periodic' and i == 0:
        new_xs[-1] = new_xs[0]
        new_ys[-1] = new_ys[0]
        new_rs[-1] = round((new_rs[-1] - new_rs[0]) / (np.pi * 2)) * np.pi * 2 + new_rs[0]  # 保持首尾旋转角度一致

    return new_xs, new_ys, new_rs

def mutate_3d(
    xs, ys, zs, rs, mutation_sigma_pos, mutation_sigma_rotation,
    boundary_conditions='fixed'  # 边界条件，可以是'fixed'、'periodic'、'free'
):
    # 经过测试，这种变异方式进化效率极低，因为稍微一改变就很容易变差
    # TODO: 这里有待重新测试
    # new_xs = xs + np.random.randn(steps) * 0.0001
    # new_ys = ys + np.random.randn(steps) * 0.0001
    # new_zs = zs + np.random.randn(steps) * 0.0001
    # new_rs = rs + np.random.randn(steps) * 0.0001

    # 随机挑一个控制点变异
    new_xs = xs.copy()
    new_ys = ys.copy()
    new_zs = zs.copy()
    new_rs = rs.copy()
    if boundary_conditions == 'fixed':
        i = np.random.randint(1, len(xs) - 1)
    elif boundary_conditions == 'periodic':
        i = np.random.randint(0, len(xs) - 1)
    elif boundary_conditions == 'free':
        i = np.random.randint(0, len(xs))
    else:
        raise ValueError(f"Unknown boundary condition: {boundary_conditions}")
    new_xs[i] += np.random.randn() * mutation_sigma_pos
    new_ys[i] += np.random.randn() * mutation_sigma_pos
    new_zs[i] += np.random.randn() * mutation_sigma_pos
    new_rs[i] += np.random.randn(4) * mutation_sigma_rotation
    new_rs[i] /= np.linalg.norm(new_rs[i])  # 归一化（只有单位四元数才能表征旋转）
    if boundary_conditions == 'periodic' and i == 0:
        new_xs[-1] = new_xs[0]
        new_ys[-1] = new_ys[0]
        new_zs[-1] = new_zs[0]

        # 保持首尾旋转角度一致
        norm_0 = np.linalg.norm(new_rs[-1] - new_rs[0])
        norm_1 = np.linalg.norm(new_rs[-1] + new_rs[0])
        if norm_0 < norm_1:
            new_rs[-1] = new_rs[0]
        else:
            new_rs[-1] = -new_rs[0]

    return new_xs, new_ys, new_zs, new_rs

def run_optimization(
    forbidden_function,  # 一个函数，参数数量可以是2或4，分别对应(x, y)或(x0, y0, x1, y1)
    initial_xs,
    initial_ys,
    initial_rotations,
    sofa_w = 3.5,   # 求解域的尺寸
    sofa_h = 1.0,   # 求解域的尺寸
    resolution = 512,  # 用一个位图表示沙发形状，这个参数是横向分辨率
    iterations = 10000,  # 迭代次数
    auto_divide_trajectory = False,  # 自动细分轨迹
    auto_divide_threshold_10000 = 0.0003,  # 自动细分轨迹的阈值。如果10000步内相对增长量小于此值则进行一次细分
    auto_divide_upsampling_order = 1,  # 自动细分轨迹时的插值阶数
    mutation_sigma_pos = 0.1,  # 变异率
    mutation_sigma_rotation = 0.02,  # 变异率
    auto_adjust_mutation_rate = False,  # 自动调整变异率
    symmetrize_function = None,  # 对称化函数
    trajectory_upsampling = 10,  # 在控制点之间插值。注意如果不在控制点之间差值将会得到不合理的结果，因为算法会尝试在控制点之间引入跃变而跳过障碍
    trajectory_upsampling_order = 1,  # 插值阶数
    regularization = 0.,  # 正则化
    regularization_mode = 'L2',  # 正则化模式，可以是'L1'或'L2'。L1能更好地抑制毛刺，L2能让轨迹点间距更均匀
    print_every = None,
    save_image_every = None,
    save_image_path = 'images/sofa_',
    save_image_start_id = 0,
    save_trajectory_path = 'trajectory/sofa_',  # 自动在每次输出图片时输出路径的npy文件。如果设成None，则不输出npy文件
    boundary_conditions = 'fixed'  # 边界条件，可以是'fixed'、'periodic'、'free'
):
    assert regularization >= 0, "Regularization must be non-negative"
    if regularization > 0:
        assert regularization_mode in ['L1', 'L2'], "Regularization_mode must be 'L1' or 'L2'"

    best_xs, best_ys, best_rotations = initial_xs, initial_ys, initial_rotations
    if callable(symmetrize_function):
        best_xs, best_ys, best_rotations = symmetrize_function(best_xs, best_ys, best_rotations)
    global is_forbidden
    is_forbidden = forbidden_function
    assert get_parameter_count(is_forbidden) in [2, 4]

    if not save_image_every is None:
        save_image_path_pure = save_image_path[:save_image_path.rfind('/') + 1]
        if save_image_path_pure != '':
            if not os.path.exists(save_image_path_pure):
                os.makedirs(save_image_path_pure)
        if not save_trajectory_path is None:
            save_trajectory_path_pure = save_trajectory_path[:save_trajectory_path.rfind('/') + 1]
            if save_trajectory_path_pure != '':
                if not os.path.exists(save_trajectory_path_pure):
                    os.makedirs(save_trajectory_path_pure)

    # Taichi fields
    x_field, y_field, rotation_field, survive_mask = generate_fields(sofa_w, sofa_h, resolution, len(best_xs), trajectory_upsampling)

    # evaluate initial score
    x_field.from_numpy(zoom(best_xs, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
    y_field.from_numpy(zoom(best_ys, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
    rotation_field.from_numpy(zoom(best_rotations, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
    if get_parameter_count(is_forbidden) == 2:
        compute_survive(sofa_w, sofa_h, x_field, y_field, rotation_field, survive_mask)
    else:
        compute_survive_2(sofa_w, sofa_h, x_field, y_field, rotation_field, survive_mask)
    maximal_area = get_area_kernel(sofa_w, sofa_h, survive_mask)

    loss = -maximal_area
    if regularization > 0:
        if regularization_mode == 'L1':
            loss += regularization * np.sum((np.diff(best_xs) ** 2 + np.diff(best_ys) ** 2 + np.diff(best_rotations) ** 2) ** 0.5)
        else:
            loss += regularization * (np.sum(np.diff(best_xs) ** 2) + np.sum(np.diff(best_ys) ** 2) + np.sum(np.diff(best_rotations) ** 2)) ** 0.5
    print(f"initial area = {maximal_area}, initial loss = {loss}")

    if not save_image_every is None:
        sofa = survive_mask.to_numpy() < 0
        image = (sofa * 255).astype(np.uint8)
        image = Image.fromarray(image.T[::-1])
        image.save(save_image_path + f'{save_image_start_id}.png')
        np.save(save_trajectory_path + f'{save_image_start_id}.npy', np.array([best_xs, best_ys, best_rotations]))

    maximal_area_record = []

    t0 = time.time()
    for iteration in range(iterations):
        new_xs, new_ys, new_rs = mutate(best_xs, best_ys, best_rotations, mutation_sigma_pos, mutation_sigma_rotation, boundary_conditions=boundary_conditions)
        if callable(symmetrize_function):
            new_xs, new_ys, new_rs = symmetrize_function(new_xs, new_ys, new_rs)
        # copy to taichi fields
        x_field.from_numpy(zoom(new_xs, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
        y_field.from_numpy(zoom(new_ys, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
        rotation_field.from_numpy(zoom(new_rs, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
        if get_parameter_count(is_forbidden) == 2:
            compute_survive(sofa_w, sofa_h, x_field, y_field, rotation_field, survive_mask)
        else:
            compute_survive_2(sofa_w, sofa_h, x_field, y_field, rotation_field, survive_mask)
        new_area = get_area_kernel(sofa_w, sofa_h, survive_mask)
        new_loss = -new_area
        if regularization > 0:
            if regularization_mode == 'L1':
                new_loss += regularization * np.sum((np.diff(new_xs) ** 2 + np.diff(new_ys) ** 2 + np.diff(new_rs) ** 2) ** 0.5)
            else:
                new_loss += regularization * (np.sum(np.diff(new_xs) ** 2) + np.sum(np.diff(new_ys) ** 2) + np.sum(np.diff(new_rs, axis=0) ** 2)) ** 0.5

        if new_loss < loss:
            loss = new_loss
            maximal_area = new_area
            best_xs = new_xs
            best_ys = new_ys
            best_rotations = new_rs
            # print(f"best_score={best_score}")
        maximal_area_record.append(maximal_area)

        if (not print_every is None) and (iteration + 1) % print_every == 0:
            print(f"iter {iteration + 1} / {iterations}, maximal_area = {maximal_area}, loss = {loss}")
        
        if (not save_image_every is None) and (iteration + 1) % save_image_every == 0:
            x_field.from_numpy(zoom(best_xs, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
            y_field.from_numpy(zoom(best_ys, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
            rotation_field.from_numpy(zoom(best_rotations, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
            if get_parameter_count(is_forbidden) == 2:
                compute_survive(sofa_w, sofa_h, x_field, y_field, rotation_field, survive_mask)
            else:
                compute_survive_2(sofa_w, sofa_h, x_field, y_field, rotation_field, survive_mask)
            sofa = survive_mask.to_numpy() < 0
            image = (sofa * 255).astype(np.uint8)
            image = Image.fromarray(image.T[::-1])
            id = (iteration + 1) // save_image_every + save_image_start_id
            image.save(save_image_path + f'{id}.png')
            np.save(save_trajectory_path + f'{id}.npy', np.array([best_xs, best_ys, best_rotations]))

        # 自动调整变异率
        if auto_adjust_mutation_rate and (iteration + 1) % 1000 == 0:  # 每1000步考虑一次调整变异率
            getting_better_rate = len(set(maximal_area_record[-1000:])) / 1000
            if getting_better_rate < 0.005:  # 大量迭代步都没有找到更优解，说明可能变异率过大了
                mutation_sigma_pos *= 0.9
                mutation_sigma_rotation *= 0.9
            elif getting_better_rate > 0.01:  # 大量迭代步都找到了更优解，说明可能变异率过小了
                mutation_sigma_pos *= 1.1
                mutation_sigma_rotation *= 1.1

        # 自动细分轨迹
        if auto_divide_trajectory and (iteration + 1) % 10000 == 0 and iteration + 1 < iterations:  # 每10000步考虑一次细分轨迹
            if maximal_area_record[-1] / maximal_area_record[-10000] < (1 + auto_divide_threshold_10000):  # 进展缓慢，说明当前离散化分辨率下可能已经收敛到最优了
                resolution_now = len(best_xs)
                best_xs = zoom(best_xs, zoom=(resolution_now * 2 - 1) / resolution_now, order=auto_divide_upsampling_order)
                best_ys = zoom(best_ys, zoom=(resolution_now * 2 - 1) / resolution_now, order=auto_divide_upsampling_order)
                best_rotations = zoom(best_rotations, zoom=(resolution_now * 2 - 1) / resolution_now, order=auto_divide_upsampling_order)
                control_point_num = len(best_xs)
                x_field = ti.field(dtype=ti.f32, shape=control_point_num * trajectory_upsampling)
                y_field = ti.field(dtype=ti.f32, shape=control_point_num * trajectory_upsampling)
                rotation_field = ti.field(dtype=ti.f32, shape=control_point_num * trajectory_upsampling)

    t1 = time.time()
    print(f"Done. Time: {t1-t0:.2f}s")

    # produce final survive mask from best
    x_field.from_numpy(zoom(best_xs, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
    y_field.from_numpy(zoom(best_ys, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
    rotation_field.from_numpy(zoom(best_rotations, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
    if get_parameter_count(is_forbidden) == 2:
        compute_survive(sofa_w, sofa_h, x_field, y_field, rotation_field, survive_mask)
    else:
        compute_survive_2(sofa_w, sofa_h, x_field, y_field, rotation_field, survive_mask)
    final_sofa = survive_mask.to_numpy() < 0

    return maximal_area, best_xs, best_ys, best_rotations, final_sofa, maximal_area_record

def run_optimization_3d(
    forbidden_function,  # 一个函数，参数数量可以是3或6，分别对应(x, y, z)或(x0, y0, z0, x1, y1, z1)
    initial_xs,
    initial_ys,
    initial_zs,
    initial_rotations,  # 每个元素是一个单位四元数
    sofa_w = 1.0,   # 求解域的尺寸
    sofa_h = 1.0,   # 求解域的尺寸
    sofa_d = 1.0,   # 求解域的尺寸
    resolution = 64,  # 用一个位图表示沙发形状，这个参数是X方向的分辨率
    iterations = 10000,  # 迭代次数
    auto_divide_trajectory = False,  # 自动细分轨迹
    auto_divide_threshold_10000 = 0.0003,  # 自动细分轨迹的阈值。如果10000步内相对增长量小于此值则进行一次细分
    auto_divide_upsampling_order = 1,  # 自动细分轨迹时的插值阶数
    mutation_sigma_pos = 0.1,  # 变异率
    mutation_sigma_rotation = 0.1,  # 变异率 0.02
    auto_adjust_mutation_rate = False,  # 自动调整变异率
    symmetrize_function = None,  # 对称化函数
    trajectory_upsampling = 10,  # 在控制点之间插值。注意如果不在控制点之间差值将会得到不合理的结果，因为算法会尝试在控制点之间引入跃变而跳过障碍
    trajectory_upsampling_order = 1,  # 插值阶数
    regularization = 0.,  # 正则化
    regularization_mode = 'L2',  # 正则化模式，可以是'L1'或'L2'。L1能更好地抑制毛刺，L2能让轨迹点间距更均匀
    print_every = None,
    save_image_every = None,
    save_image_path = 'images/sofa_',
    save_image_start_id = 0,
    save_trajectory_path = 'trajectory/sofa_',  # 自动在每次输出图片时输出路径的npy文件。如果设成None，则不输出npy文件
    boundary_conditions = 'fixed'  # 边界条件，可以是'fixed'、'periodic'、'free'
):
    assert regularization >= 0, "Regularization must be non-negative"
    if regularization > 0:
        assert regularization_mode in ['L1', 'L2'], "Regularization_mode must be 'L1' or 'L2'"

    best_xs, best_ys, best_zs, best_rotations = initial_xs, initial_ys, initial_zs, initial_rotations
    if callable(symmetrize_function):
        best_xs, best_ys, best_zs, best_rotations = symmetrize_function(best_xs, best_ys, best_zs, best_rotations)  # 对称化
    global is_forbidden
    is_forbidden = forbidden_function
    assert get_parameter_count(is_forbidden) in [3, 6]

    if not save_image_every is None:
        save_image_path_pure = save_image_path[:save_image_path.rfind('/') + 1]
        if save_image_path_pure != '':
            if not os.path.exists(save_image_path_pure):
                os.makedirs(save_image_path_pure)
        if not save_trajectory_path is None:
            save_trajectory_path_pure = save_trajectory_path[:save_trajectory_path.rfind('/') + 1]
            if save_trajectory_path_pure != '':
                if not os.path.exists(save_trajectory_path_pure):
                    os.makedirs(save_trajectory_path_pure)

    # Taichi fields
    x_field, y_field, z_field, rotation_field, rotation_matrix_field, survive_mask = generate_fields_3d(sofa_w, sofa_h, sofa_d, resolution, len(best_xs), trajectory_upsampling)

    # evaluate initial score
    x_field.from_numpy(zoom(best_xs, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
    y_field.from_numpy(zoom(best_ys, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
    z_field.from_numpy(zoom(best_zs, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
    rotation_field.from_numpy(zoom(best_rotations, zoom=(trajectory_upsampling, 1), order=trajectory_upsampling_order))
    normalize_quaternions(rotation_field)
    quaternions_to_rotation_matrices(rotation_field, rotation_matrix_field)  # 四元数 -> 旋转矩阵
    if get_parameter_count(is_forbidden) == 3:
        compute_survive_3d(sofa_w, sofa_h, sofa_d, x_field, y_field, z_field, rotation_matrix_field, survive_mask)
    else:
        compute_survive_3d_2(sofa_w, sofa_h, sofa_d, x_field, y_field, z_field, rotation_matrix_field, survive_mask)
    maximal_volume = get_volume_kernel(sofa_w, sofa_h, sofa_d, survive_mask)

    loss = -maximal_volume
    if regularization > 0:
        if regularization_mode == 'L1':
            loss += regularization * np.sum((np.diff(best_xs) ** 2 + np.diff(best_ys) ** 2 + np.diff(best_zs) ** 2 + np.sum(np.diff(best_rotations, axis=0) ** 2, axis=-1)) ** 0.5)
        else:
            loss += regularization * (np.sum(np.diff(best_xs) ** 2) + np.sum(np.diff(best_ys) ** 2) + np.sum(np.diff(best_zs) ** 2) + np.sum(np.diff(best_rotations, axis=0) ** 2)) ** 0.5
        

    print(f"initial volume = {maximal_volume}, initial loss = {loss}")
    if not save_image_every is None:
        np.save(save_trajectory_path + f'{save_image_start_id}.npy', np.concatenate([best_xs[:, np.newaxis], best_ys[:, np.newaxis], best_zs[:, np.newaxis], best_rotations], axis=1))
        # TODO: 保存图片

    maximal_volume_record = []

    t0 = time.time()
    for iteration in range(iterations):
        new_xs, new_ys, new_zs, new_rs = mutate_3d(best_xs, best_ys, best_zs, best_rotations, mutation_sigma_pos, mutation_sigma_rotation, boundary_conditions=boundary_conditions)
        if callable(symmetrize_function):
            new_xs, new_ys, new_zs, new_rs = symmetrize_function(new_xs, new_ys, new_zs, new_rs)  # 对称化
        # copy to taichi fields
        x_field.from_numpy(zoom(new_xs, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
        y_field.from_numpy(zoom(new_ys, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
        z_field.from_numpy(zoom(new_zs, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
        rotation_field.from_numpy(zoom(new_rs, zoom=(trajectory_upsampling, 1), order=trajectory_upsampling_order))
        normalize_quaternions(rotation_field)
        quaternions_to_rotation_matrices(rotation_field, rotation_matrix_field)  # 四元数 -> 旋转矩阵
        if get_parameter_count(is_forbidden) == 3:
            compute_survive_3d(sofa_w, sofa_h, sofa_d, x_field, y_field, z_field, rotation_matrix_field, survive_mask)
        else:
            compute_survive_3d_2(sofa_w, sofa_h, sofa_d, x_field, y_field, z_field, rotation_matrix_field, survive_mask)
        new_volume = get_volume_kernel(sofa_w, sofa_h, sofa_d, survive_mask)
        new_loss = -new_volume
        if regularization > 0:
            if regularization_mode == 'L1':
                new_loss += regularization * np.sum((np.diff(new_xs) ** 2 + np.diff(new_ys) ** 2 + np.diff(new_zs) ** 2 + np.sum(np.diff(new_rs, axis=0) ** 2, axis=-1)) ** 0.5)
            else:
                new_loss += regularization * (np.sum(np.diff(new_xs) ** 2) + np.sum(np.diff(new_ys) ** 2) + np.sum(np.diff(new_zs) ** 2) + np.sum(np.diff(new_rs, axis=0) ** 2)) ** 0.5

        if new_loss < loss:
            loss = new_loss
            maximal_volume = new_volume
            best_xs = new_xs
            best_ys = new_ys
            best_zs = new_zs
            best_rotations = new_rs
            # print(f"best_score={best_score}")
        maximal_volume_record.append(maximal_volume)

        if (not print_every is None) and (iteration + 1) % print_every == 0:
            print(f"iter {iteration + 1} / {iterations}, maximal_volume = {maximal_volume}, loss = {loss}")
        
        if (not save_image_every is None) and (iteration + 1) % save_image_every == 0:
            id = (iteration + 1) // save_image_every + save_image_start_id
            np.save(save_trajectory_path + f'{id}.npy', np.concatenate([best_xs[:, np.newaxis], best_ys[:, np.newaxis], best_zs[:, np.newaxis], best_rotations], axis=1))
            # x_field.from_numpy(zoom(best_xs, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
            # y_field.from_numpy(zoom(best_ys, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
            # z_field.from_numpy(zoom(best_zs, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
            # rotation_field.from_numpy(zoom(best_rotations, zoom=(trajectory_upsampling, 1), order=trajectory_upsampling_order))
            # if get_parameter_count(is_forbidden) == 3:
            #     compute_survive_3d(sofa_w, sofa_h, sofa_d, x_field, y_field, z_field, rotation_matrix_field, survive_mask)
            # else:
            #     compute_survive_3d_2(sofa_w, sofa_h, sofa_d, x_field, y_field, z_field, rotation_matrix_field, survive_mask)
            # sofa = survive_mask.to_numpy() < 0
            # TODO: 保存图片

        # 自动调整变异率
        if auto_adjust_mutation_rate and (iteration + 1) % 1000 == 0:  # 每1000步考虑一次调整变异率
            getting_better_rate = len(set(maximal_volume_record[-1000:])) / 1000
            if getting_better_rate < 0.005:  # 大量迭代步都没有找到更优解，说明可能变异率过大了
                mutation_sigma_pos *= 0.9
                mutation_sigma_rotation *= 0.9
            elif getting_better_rate > 0.01:  # 大量迭代步都找到了更优解，说明可能变异率过小了
                mutation_sigma_pos *= 1.1
                mutation_sigma_rotation *= 1.1

        # 自动细分轨迹
        if auto_divide_trajectory and (iteration + 1) % 10000 == 0 and iteration + 1 < iterations:  # 每10000步考虑一次细分轨迹
            if maximal_volume_record[-1] / maximal_volume_record[-10000] < (1 + auto_divide_threshold_10000):  # 进展缓慢，说明当前离散化分辨率下可能已经收敛到最优了
                resolution_now = len(best_xs)
                best_xs = zoom(best_xs, zoom=(resolution_now * 2 - 1) / resolution_now, order=auto_divide_upsampling_order)
                best_ys = zoom(best_ys, zoom=(resolution_now * 2 - 1) / resolution_now, order=auto_divide_upsampling_order)
                best_zs = zoom(best_zs, zoom=(resolution_now * 2 - 1) / resolution_now, order=auto_divide_upsampling_order)
                best_rotations = zoom(best_rotations, zoom=((resolution_now * 2 - 1) / resolution_now, 1), order=auto_divide_upsampling_order)
                control_point_num = len(best_xs)
                x_field = ti.field(dtype=ti.f32, shape=control_point_num * trajectory_upsampling)
                y_field = ti.field(dtype=ti.f32, shape=control_point_num * trajectory_upsampling)
                z_field = ti.field(dtype=ti.f32, shape=control_point_num * trajectory_upsampling)
                rotation_field = ti.field(dtype=ti.f32, shape=(control_point_num * trajectory_upsampling, 4))  # 四元数数组
                rotation_matrix_field = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(control_point_num * trajectory_upsampling,))  # 旋转矩阵数组

    t1 = time.time()
    print(f"Done. Time: {t1-t0:.2f}s")

    # produce final survive mask from best
    x_field.from_numpy(zoom(best_xs, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
    y_field.from_numpy(zoom(best_ys, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
    z_field.from_numpy(zoom(best_zs, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
    rotation_field.from_numpy(zoom(best_rotations, zoom=(trajectory_upsampling, 1), order=trajectory_upsampling_order))
    normalize_quaternions(rotation_field)
    quaternions_to_rotation_matrices(rotation_field, rotation_matrix_field)  # 四元数 -> 旋转矩阵
    best_rotation_matrices = rotation_matrix_field.to_numpy()
    if get_parameter_count(is_forbidden) == 3:
        compute_survive_3d(sofa_w, sofa_h, sofa_d, x_field, y_field, z_field, rotation_matrix_field, survive_mask)
    else:
        compute_survive_3d_2(sofa_w, sofa_h, sofa_d, x_field, y_field, z_field, rotation_matrix_field, survive_mask)
    final_sofa = survive_mask.to_numpy() < 0

    return maximal_volume, best_xs, best_ys, best_zs, best_rotations, best_rotation_matrices, final_sofa, maximal_volume_record

@ti.kernel
def test_forbidden_function_kernel(image: ti.template(), x_min: float, x_max: float, y_min: float, y_max: float):  # 测试双参数版的墙函数 # type: ignore
    for i, j in image:
        x = x_min + (x_max - x_min) * (i + 0.5) / image.shape[0]
        y = y_min + (y_max - y_min) * (j + 0.5) / image.shape[1]
        if is_forbidden(x, y):
            image[i, j] = 1.0
        else:
            image[i, j] = 0.0

@ti.kernel
def test_forbidden_function_3d_kernel(image: ti.template(), x_min: float, x_max: float, y_min: float, y_max: float, z_min: float, z_max: float):  # 测试3参数版的墙函数 # type: ignore
    for i, j, k in image:
        x = x_min + (x_max - x_min) * (i + 0.5) / image.shape[0]
        y = y_min + (y_max - y_min) * (j + 0.5) / image.shape[1]
        z = z_min + (z_max - z_min) * (k + 0.5) / image.shape[2]
        if is_forbidden(x, y, z):
            image[i, j, k] = 1.0
        else:
            image[i, j, k] = 0.0

@ti.kernel
def test_forbidden_function_2_kernel(image: ti.template(), x_min: float, x_max: float, y_min: float, y_max: float, wall_normal_x: float, wall_normal_y: float):  # 测试四参数版的墙函数 # type: ignore
    for i, j in image:
        x = x_min + (x_max - x_min) * (i + 0.5) / image.shape[0]
        y = y_min + (y_max - y_min) * (j + 0.5) / image.shape[1]
        if is_forbidden(x - wall_normal_x * 0.5, y - wall_normal_y * 0.5, x + wall_normal_x * 0.5, y + wall_normal_y * 0.5):
            image[i, j] = 1.0
        else:
            image[i, j] = 0.0

@ti.kernel
def test_forbidden_function_3d_2_kernel(image: ti.template(), x_min: float, x_max: float, y_min: float, y_max: float, z_min: float, z_max: float, wall_normal_x: float, wall_normal_y: float, wall_normal_z: float):  # 测试四参数版的墙函数 # type: ignore
    for i, j, k in image:
        x = x_min + (x_max - x_min) * (i + 0.5) / image.shape[0]
        y = y_min + (y_max - y_min) * (j + 0.5) / image.shape[1]
        z = z_min + (z_max - z_min) * (k + 0.5) / image.shape[2]
        if is_forbidden(x - wall_normal_x * 0.5, y - wall_normal_y * 0.5, z - wall_normal_z * 0.5, x + wall_normal_x * 0.5, y + wall_normal_y * 0.5, z + wall_normal_z * 0.5):
            image[i, j, k] = 1.0
        else:
            image[i, j, k] = 0.0

def test_forbidden_function(
        forbidden_function,  # 一个函数，参数数量可以是2或4，分别对应(x, y)或(x0, y0, x1, y1)
        x_min, x_max, y_min, y_max,
        resolution=512,
        wall_normal = [0, 0.01]  # 此参数仅适用于4参数版本墙函数的测试
    ):
    """测试障碍函数，返回一个NumPy数组"""
    if isinstance(resolution, int):
        resolution = (resolution, resolution)

    global is_forbidden
    is_forbidden = forbidden_function
    parameter_count = get_parameter_count(is_forbidden)
    image = ti.field(ti.f32, shape=resolution)
    if parameter_count == 2:  # (x, y)
        test_forbidden_function_kernel(image, x_min, x_max, y_min, y_max)
    elif parameter_count == 4:  # (x0, y0, x1, y1)
        test_forbidden_function_2_kernel(image, x_min, x_max, y_min, y_max, wall_normal[0], wall_normal[1])
    else:
        raise ValueError("forbidden_function must take 2 or 4 parameters")

    return image.to_numpy()

def test_forbidden_function_3d(
        forbidden_function,  # 一个函数，参数数量可以是3或6，分别对应(x, y, z)或(x0, y0, z0, x1, y1, z1)
        x_min, x_max, y_min, y_max, z_min, z_max,
        resolution=64,
        wall_normal = [0.01, 0, 0]  # 此参数仅适用于6参数版本墙函数的测试
    ):
    """测试障碍函数（3d版本），返回一个NumPy数组"""
    if isinstance(resolution, int):
        resolution = (resolution, resolution, resolution)

    global is_forbidden
    is_forbidden = forbidden_function
    parameter_count = get_parameter_count(is_forbidden)
    image = ti.field(ti.f32, shape=resolution)
    if parameter_count == 3:  # (x, y, z)
        test_forbidden_function_3d_kernel(image, x_min, x_max, y_min, y_max, z_min, z_max)
    elif parameter_count == 6:  # (x0, y0, z0, x1, y1, z1)
        test_forbidden_function_3d_2_kernel(image, x_min, x_max, y_min, y_max, z_min, z_max, wall_normal[0], wall_normal[1], wall_normal[2])
    else:
        raise ValueError("forbidden_function must take 3 or 6 parameters")

    return image.to_numpy()

def monotonicly_interpolate(keys, values, samples=10, key_min=None, key_max=None):  # 重单调采样（采样使得keys对应的函数单调递增）
    if key_min is None:
        key_min = np.min(keys)
    if key_max is None:
        key_max = np.max(keys)
    
    result = []
    for key in np.linspace(key_min, key_max, samples):
        i = np.argmax(keys >= key)
        if keys[i] == key:
            result.append(values[i])
        else:
            t = (key - keys[i - 1]) / (keys[i]- keys[i - 1])
            result.append(values[i - 1] * (1 - t) + values[i] * t)
    
    return np.array(result)
