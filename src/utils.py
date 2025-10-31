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

@ti.kernel
def compute_survive(sofa_w: float, sofa_h: float, x_field: ti.template(), y_field: ti.template(), rotation_field: ti.template(), survive_mask: ti.template()):  # 根据轨迹计算能通过的图形 # type: ignore
    for i, j in survive_mask:
        survive_mask[i, j] = -1  # -1表示幸存

        # local coords
        local_x = (i + 0.5) / survive_mask.shape[0] * sofa_w - sofa_w * 0.5
        local_y = (j + 0.5) / survive_mask.shape[1] * sofa_h - sofa_h * 0.5
        for t in range(x_field.shape[0]):
            world_x, world_y = transform_local_to_world(local_x, local_y, x_field[t], y_field[t], rotation_field[t])
            # check forbidden region
            if is_forbidden(world_x, world_y):
                survive_mask[i, j] = t  # 非负整数表示未幸存，值为这个像素被削掉的时刻
                break

@ti.kernel
def compute_survive_2(sofa_w: float, sofa_h: float, x_field: ti.template(), y_field: ti.template(), rotation_field: ti.template(), survive_mask: ti.template()):  # 根据轨迹计算能通过的图形，其中进入墙壁的判定函数需要输入相邻时间步的两个点 # type: ignore
    for i, j in survive_mask:
        survive_mask[i, j] = -1  # -1表示幸存

        # local coords
        local_x = (i + 0.5) / survive_mask.shape[0] * sofa_w - sofa_w * 0.5
        local_y = (j + 0.5) / survive_mask.shape[1] * sofa_h - sofa_h * 0.5
        world_x, world_y = transform_local_to_world(local_x, local_y, x_field[0], y_field[0], rotation_field[0])
        for t in range(1, x_field.shape[0]):
            world_x_new, world_y_new = transform_local_to_world(local_x, local_y, x_field[t], y_field[t], rotation_field[t])
            # check forbidden region
            if is_forbidden(world_x, world_y, world_x_new, world_y_new):
                survive_mask[i, j] = t  # 非负整数表示未幸存，值为这个像素被削掉的时刻
                break
            world_x, world_y = world_x_new, world_y_new

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

def get_sofa(  # 根据轨迹计算沙发形状，返回一张位图
    forbidden_function,  # 一个函数，参数数量可以是2或4，分别对应(x, y)或(x0, y0, x1, y1)
    xs,
    ys,
    rotations,
    sofa_w = 3.5,   # 求解域的尺寸
    sofa_h = 1.0,   # 求解域的尺寸
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
        compute_survive(sofa_w, sofa_h, x_field, y_field, rotation_field, survive_mask)
    else:
        compute_survive_2(sofa_w, sofa_h, x_field, y_field, rotation_field, survive_mask)

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

def mutate(xs, ys, rs, mutation_sigma_pos, mutation_sigma_rotation):
    # 经过测试，这种变异方式进化效率极低，因为稍微一改变就很容易变差
    # TODO: 这里有待重新测试
    # new_xs = xs + np.random.randn(steps) * 0.0001
    # new_ys = ys + np.random.randn(steps) * 0.0001
    # new_rs = rs + np.random.randn(steps) * 0.0001

    # 随机挑一个控制点变异
    new_xs = xs.copy()
    new_ys = ys.copy()
    new_rs = rs.copy()
    i = np.random.randint(1, len(xs) - 1)
    new_xs[i] += np.random.randn() * mutation_sigma_pos
    new_ys[i] += np.random.randn() * mutation_sigma_pos
    new_rs[i] += np.random.randn() * mutation_sigma_rotation

    # keep endpoints exact
    new_xs[0] = xs[0]
    new_xs[-1] = xs[-1]
    new_ys[0] = ys[0]
    new_ys[-1] = ys[-1]
    new_rs[0] = rs[0]
    new_rs[-1] = rs[-1]

    return new_xs, new_ys, new_rs

def run_optimization(
    forbidden_function,  # 一个函数，参数数量可以是2或4，分别对应(x, y)或(x0, y0, x1, y1)
    initial_xs,
    initial_ys,
    initial_rotations,
    sofa_w = 3.5,   # 求解域的尺寸
    sofa_h = 1.0,   # 求解域的尺寸
    resolution = 512,  # 用一个位图表示沙发形状，这个参数是横向分辨率
    iterations = 10000,  # 迭代次数
    mutation_sigma_pos = 0.1,  # 变异率
    mutation_sigma_rotation = 0.02,  # 变异率
    trajectory_upsampling = 10,  # 在控制点之间插值。注意如果不在控制点之间差值将会得到不合理的结果，因为算法会尝试在控制点之间引入跃变而跳过障碍
    trajectory_upsampling_order = 1,  # 插值阶数
    print_every = None,
    save_image_every = None,
    save_image_path = 'images/sofa_',
    save_image_start_id = 0,
    save_trajectory_path = 'trajectory/sofa_'  # 自动在每次输出图片时输出路径的npy文件。如果设成None，则不输出npy文件
):
    best_xs, best_ys, best_rotations = initial_xs, initial_ys, initial_rotations
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
    print(f"Initial survivors: {maximal_area}")
    if not save_image_every is None:
        sofa = survive_mask.to_numpy() < 0
        image = (sofa * 255).astype(np.uint8)
        image = Image.fromarray(image.T[::-1])
        image.save(save_image_path + f'{save_image_start_id}.png')
        np.save(save_trajectory_path + f'{save_image_start_id}.npy', np.array([best_xs, best_ys, best_rotations]))

    maximal_area_record = []

    t0 = time.time()
    for iteration in range(iterations):
        new_xs, new_ys, new_rs = mutate(best_xs, best_ys, best_rotations, mutation_sigma_pos, mutation_sigma_rotation)
        # copy to taichi fields
        x_field.from_numpy(zoom(new_xs, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
        y_field.from_numpy(zoom(new_ys, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
        rotation_field.from_numpy(zoom(new_rs, zoom=trajectory_upsampling, order=trajectory_upsampling_order))
        if get_parameter_count(is_forbidden) == 2:
            compute_survive(sofa_w, sofa_h, x_field, y_field, rotation_field, survive_mask)
        else:
            compute_survive_2(sofa_w, sofa_h, x_field, y_field, rotation_field, survive_mask)
        new_area = get_area_kernel(sofa_w, sofa_h, survive_mask)
        if new_area >= maximal_area:
            maximal_area = new_area
            best_xs = new_xs
            best_ys = new_ys
            best_rotations = new_rs
            # print(f"best_score={best_score}")
        maximal_area_record.append(maximal_area)

        if (not print_every is None) and (iteration + 1) % print_every == 0:
            print(f"iter {iteration + 1} / {iterations}, maximal_area={maximal_area}")
        
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
            id = (iteration + 1) // save_image_every - 1 + save_image_start_id
            image.save(save_image_path + f'{id}.png')
            np.save(save_trajectory_path + f'{id}.npy', np.array([best_xs, best_ys, best_rotations]))

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

@ti.kernel
def test_forbidden_function_kernel(image: ti.template(), x_min: float, x_max: float, y_min: float, y_max: float):  # type: ignore
    for i, j in image:
        x = x_min + (x_max - x_min) * (i + 0.5) / image.shape[0]
        y = y_min + (y_max - y_min) * (j + 0.5) / image.shape[1]
        if is_forbidden(x, y):
            image[i, j] = 1.0
        else:
            image[i, j] = 0.0

def test_forbidden_function(  # 测试障碍函数，返回一个位图
        forbidden_function,  # 一个函数，参数数量可以是2或4，分别对应(x, y)或(x0, y0, x1, y1)
        x_min, x_max, y_min, y_max,
        resolution
    ):
    if isinstance(resolution, int):
        resolution = (resolution, resolution)

    global is_forbidden
    is_forbidden = forbidden_function
    assert get_parameter_count(is_forbidden) == 2

    image = ti.field(ti.f32, shape=resolution)
    test_forbidden_function_kernel(image, x_min, x_max, y_min, y_max)

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
