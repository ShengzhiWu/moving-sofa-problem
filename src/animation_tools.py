import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils import get_sofa, monotonicly_interpolate

def draw_pattern(image, id, min_pattern_size=6):  # 绘制随机花纹
    np.random.seed(id + 1)

    # 底色
    image[:, :, 0] = np.random.randint(200)
    image[:, :, 1] = np.random.randint(200)
    image[:, :, 2] = np.random.randint(200)

    # 随机花纹
    a = np.random.rand()
    if a < 0.5:  # 纯色
        pass
    elif a < 0.7:  # 格子
        pattern_size = int(min_pattern_size * (int(image.shape[1]) / min_pattern_size) ** np.random.rand())
        for i in range(3):
            b = np.random.rand() * 0.8 + 0.6
            image[np.arange(image.shape[0], dtype=int) % pattern_size < pattern_size // 2, :, i] *= b
            image[:, np.arange(image.shape[1], dtype=int) % pattern_size < pattern_size // 2, i] *= b
        image[:, :, :3] = np.minimum(255, image[:, :, :3])
    elif a < 0.8:  # 横条纹
        pattern_size = int(min_pattern_size * (int(image.shape[1]) / min_pattern_size) ** np.random.rand())
        for i in range(3):
            image[np.arange(image.shape[0], dtype=int) % pattern_size < pattern_size // 2, :, i] *= np.random.rand() * 0.8 + 0.6
        image[:, :, :3] = np.minimum(255, image[:, :, :3])
    elif a < 0.95:  # 竖条纹
        pattern_size = int(min_pattern_size * (int(image.shape[1]) / min_pattern_size) ** np.random.rand())
        for i in range(3):
            image[:, np.arange(image.shape[1], dtype=int) % pattern_size < pattern_size // 2, i] *= np.random.rand() * 0.8 + 0.6
        image[:, :, :3] = np.minimum(255, image[:, :, :3])
    else:  # 圆点
        x, y = np.mgrid[0:image.shape[0] - 1:image.shape[0] * (1j), 0:image.shape[1] - 1:image.shape[1] * (1j)]
        x = x.flatten()
        y = y.flatten()
        pattern_size = min_pattern_size * (int(image.shape[1]) * 0.3 / min_pattern_size) ** np.random.rand()
        original_shape = image.shape
        image = np.reshape(image, (-1, image.shape[-1]))
        mask = np.zeros(image.shape[0], dtype=bool)
        for i in range(int(0.7 * np.random.rand() * image.shape[0] * image.shape[1] / pattern_size ** 2)):
            center = (
                np.random.rand() * (original_shape[0] + pattern_size) - pattern_size / 2,
                np.random.rand() * (original_shape[1] + pattern_size) - pattern_size / 2
            )
            mask[(x - center[0]) ** 2 + (y - center[1]) ** 2 < (pattern_size / 2) ** 2] = True
        for i in range(3):
            image[mask, i] *= np.random.rand() * 0.8 + 0.6
        image = np.reshape(image, original_shape)
        image[:, :, :3] = np.minimum(255, image[:, :, :3])

def generate_sofa_image(
    is_forbidden,
    path,
    id = 0,
    draw_text = True,
    text_row_1 = '',
    test_row_2 = None,
    resolution = 500,
    trajectory_upsampling = 50,
    text_size = 35
):
    xs, ys, rotations = np.load(path + f'{id}.npy')
    sofa = get_sofa(is_forbidden, xs, ys, rotations, resolution=resolution, trajectory_upsampling=trajectory_upsampling)
    center = [
        np.sum(sofa * np.arange(sofa.shape[0])[:, np.newaxis]) / np.sum(sofa),
        np.sum(sofa * np.arange(sofa.shape[1])[np.newaxis, :]) / np.sum(sofa)
    ]
    colume = sofa[int(center[0])]
    center[1] = (np.argmax(colume[::-1]) + (len(colume) - np.argmax(colume))) / 2

    image = np.ones(list(sofa.T.shape) + [4])
    draw_pattern(image, id)  # 绘制随机花纹
    image[:, :, 3] = sofa.T[::-1] * 255
    image = Image.fromarray(image.astype(np.uint8))

    if draw_text:
        draw = ImageDraw.Draw(image)
        text_row_2 = f'面积：{np.sum(sofa) / sofa.shape[1] ** 2:.4f}'
        if text_row_2 is None:
            font = ImageFont.truetype("simhei.ttf", text_size)
            draw.text((center[0] - draw.textlength(text_row_1, font=font) / 2,  center[1] - text_size / 2), text_row_1, fill=(255, 255, 255), font=font)
        else:  # 有两行文本要绘制
            font = ImageFont.truetype("simhei.ttf", text_size)
            draw.text((center[0] - draw.textlength(text_row_1, font=font) / 2,  center[1] + text_size * -0.8), text_row_1, fill=(255, 255, 255), font=font)
            font = ImageFont.truetype("simhei.ttf", text_size * 0.5)
            draw.text((center[0] - draw.textlength(text_row_2, font=font) / 2,  center[1] + text_size * 0.3), text_row_2, fill=(255, 255, 255), font=font)

    return image

def generate_animated_sofa(
    is_forbidden,
    path,
    id,  # 14453
    start_time,
    duration,
    resolution,
    rotation_speed_parameter = 0.4,  # 控制旋转与平移速度比例的参数，值越小，旋转相对于平移越快，但注意取0时旋转速度仍然 > 0
    key_function = None,
    draw_text = True,
    id_factor = 83,  # 显示的编号是输入的编号的多少倍
    text_size_factor = 0.06,
    trajectory_updampling = 1  # 在对轨迹进行单调化处理前先重新采样
):
    image = generate_sofa_image(
        is_forbidden, path,
        id=id,
        draw_text=draw_text,
        text_row_1=f'#{id * id_factor}',
        resolution=resolution,
        trajectory_upsampling=101,
        text_size=resolution * text_size_factor
    )
    
    xs, ys, rotations = np.load(path + f'{id}.npy')
    if trajectory_updampling != 1:
        xs = np.interp(np.linspace(0, len(xs) - 1, num=(len(xs) - 1) * trajectory_updampling + 1), np.arange(len(xs)), xs)
        ys = np.interp(np.linspace(0, len(ys) - 1, num=(len(ys) - 1) * trajectory_updampling + 1), np.arange(len(ys)), ys)
        rotations = np.interp(np.linspace(0, len(rotations) - 1, num=(len(rotations) - 1) * trajectory_updampling + 1), np.arange(len(rotations)), rotations)
    if callable(key_function):
        keys = key_function(xs, ys, rotations)
    else:
        keys = xs - ys - rotations * rotation_speed_parameter  # 要求这个函数单调递增来重新采样
    xs_monotonic, ys_monotonic, rotations_monotonic = monotonicly_interpolate(
        keys,
        np.array([xs, ys, rotations]).T,
        duration,
        key_min=keys[0],
        key_max=keys[-1]
    ).T
    
    return {'image': image, 'start_time': start_time, 'xs': xs_monotonic, 'ys': ys_monotonic, 'rotations': rotations_monotonic}

def draw_sofa(
    rendering_target,
    origin,
    scale,
    sofa_image,
    xs,
    ys,
    rotations, process  # 运动进度，可以超过数组范围，超过的部分会自动线性延拓（注意延拓方向要么水平要么竖直，不支持斜向运动）
):
    if process < 0:  # 延拓
        if abs(xs[1] - xs[0]) > abs(ys[1] - ys[0]):  # 横向移动
            x = xs[0] + (xs[1] - xs[0]) * process
            y = ys[0]
        else:  # 纵向移动
            x = xs[0]
            y = ys[0] + (ys[1] - ys[0]) * process
        rotation = rotations[0]
    elif process < len(xs):
        x = xs[process]
        y = ys[process]
        rotation = rotations[process]
    else:  # 延拓
        if abs(xs[-1] - xs[-2]) > abs(ys[-1] - ys[-2]):  # 横向移动
            x = xs[-1] + (xs[-1] - xs[-2]) * (process - (len(xs) - 1))
            y = ys[-1]
        else:  # 纵向移动
            x = xs[-1]
            y = ys[-1] + (ys[-1] - ys[-2]) * (process - (len(xs) - 1))
        rotation = rotations[-1]

    sofa_image_transformed = sofa_image.rotate(rotation / np.pi * 180, expand=True, resample=Image.Resampling.BILINEAR)
    sofa_image_transformed = sofa_image_transformed.resize(
        (round(sofa_image_transformed.width * scale / sofa_image.height), round(sofa_image_transformed.height * scale / sofa_image.height))
    )

    rendering_target.paste(
        sofa_image_transformed,
        (
            int(origin[0] + x * scale - sofa_image_transformed.width / 2),
            int(origin[1] - y * scale - sofa_image_transformed.height / 2)
        ),
        sofa_image_transformed
    )
