import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import find_objects, center_of_mass
import cv2
from typing import Optional, Tuple, Union, List, Any, Dict
import logging
import configparser
import os 
from utils.calibration import Convert
import math
from scipy.spatial.transform import Rotation as R
from datetime import datetime

# 将以逗号和冒号分割的名词分开
def sort_label(input_str) -> tuple[str, str]:

    parts = input_str.split(',')
    processed_parts = []
    current_key = None
    
    for part in parts:
        if ':' in part:
            # 这是一个新的键值对
            if current_key is not None:
                # 保存前一个键值对
                processed_parts.append(f"{current_key}:{current_value}")
            key, value = part.split(':', 1)
            current_key = key
            current_value = value
        else:
            # 这是前一个值的延续
            if current_key is not None:
                current_value += '-' + part
    
    # 添加最后一个键值对
    if current_key is not None:
        processed_parts.append(f"{current_key}:{current_value}")
    
    label = ','.join(processed_parts)


    grounding_label = ''
    noun_label = ''
    parts = [item.strip() for item in label.split(',') if item.strip()]
    for part in parts:
        # 同时支持英文冒号 `:` 和中文冒号 `：`
        segments = [s.strip() for s in part.replace('：', ':').split(':') if s.strip()]
        if len(segments) != 2:
            raise ValueError(f"标签部分 '{part}' 必须包含一个冒号分隔符（英文或中文）")
        
        if not grounding_label:
            grounding_label = segments[0]
            noun_label = segments[1]
        else:
            grounding_label = ','.join([grounding_label, segments[0]])
            noun_label = ','.join([noun_label, segments[1]])
    return grounding_label, noun_label

# 在逗号分割的字符串和字符串列表之间转化
def str2list(input_string):
    return [item.strip() for item in input_string.split(',') if item.strip()]
def list2str(input_list):
    return ', '.join(input_list)

# 将逗号分割的字符串中的空格替换为下划线
def union_str(g_label_temps: str) -> str:
    text_prompts = [item.strip() for item in g_label_temps.strip().split(',') if item.strip()]
    for i in range(len(text_prompts)):
        text_prompts[i] = '_'.join(text_prompts[i].split())
    text_prompts = ', '.join(text_prompts) # str
    return text_prompts

# 将字符串中的下划线替换为空格
def separate_str(label :str) -> str:
    return ' '.join(label.split(' _ '))


def load_ini_to_globals(ini_path: str, globals_dict: Dict[str, Any]) -> None:
    """
    自动读取INI文件中的所有配置并转换为全局变量
    
    :param ini_path: INI文件路径
    :param globals_dict: globals()字典
    """
    config = configparser.ConfigParser()
    config.read(ini_path)
    
    for section in config.sections():
        # 将节名转换为大写作为变量前缀
        section_prefix = f"{section.upper()}_"
        
        for key, value in config.items(section):
            # 将键名转换为大写作为变量名
            # var_name = f"{section_prefix}{key.upper()}"
            var_name = f"{key.upper()}"
            
            try:
                # 尝试将逗号分隔的值转换为浮点数列表
                value_list = str2list(value)
                if value_list:  # 如果有有效值
                    globals_dict[var_name] = list(map(float, value_list))
                    continue
            except ValueError:
                pass
                
            # 如果转换失败，保持原始字符串值
            globals_dict[var_name] = value

def save_and_pause(image, save_dir="vis", pause=True, filename=None):
    """
    适用于远程环境的图像保存与暂停函数（支持自动时间戳命名）
    
    参数:
        image: 输入图像 (numpy.ndarray 或 PIL.Image)
        save_dir: 保存目录 (默认 "output_images")
        filename: 保存文件名 (可选，未指定时自动按时间生成)
    """
    try:
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 自动生成时间戳文件名（如果未指定）
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}.png"
        
        save_path = os.path.join(save_dir, filename)
        
        # 统一转换为PIL格式保存
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8 and len(image.shape) == 3:
                # OpenCV BGR转RGB
                if image.shape[2] == 3:
                    image = image[..., ::-1]  # BGR to RGB
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError("输入必须是numpy.ndarray或PIL.Image")
        
        # 保存图像（优化压缩质量）
        pil_image.save(save_path, optimize=True, quality=95)
        
        # 纯终端暂停
        if pause:
            _pause_remote()
        
        return save_path  # 返回保存路径以便后续使用
    
    except Exception as e:
        print(f"❌ 保存失败: {str(e)}")
        raise

def _pause_remote():
    """适用于远程环境的暂停函数"""
    print("\n>>> 按回车键继续...", end='', flush=True)
    try:
        # 尝试使用input()（兼容性最好）
        input()
    except EOFError:
        # 如果input不可用（如某些脚本环境），使用timeout等待
        time.sleep(5)  # 默认等待5秒

def clean_directory(dir_path="output_images"):
    """清理目录（保持不变）"""
    if os.path.exists(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"删除 {file_path} 失败: {e}")
        print(f"已清理目录: {dir_path}")
    else:
        print(f"目录不存在: {dir_path}")

# pil和array之间的转换
def pil2np(pil_image: Image.Image, bgr2rgb: bool = True) -> np.ndarray:
    """
    将PIL图像转换为NumPy数组（支持RGB/BGR格式转换）
    
    Args:
        pil_image: PIL.Image对象
        bgr2rgb: 是否将BGR格式转换为RGB（OpenCV默认使用BGR）
    
    Returns:
        np.ndarray: 形状为 (H, W, 3) 的uint8数组
    """
    np_image = np.array(pil_image)  # 直接转换（可能为RGB或RGBA）
    
    # 处理RGBA图像（移除Alpha通道）
    if np_image.shape[-1] == 4:
        np_image = np_image[..., :3]
    
    # BGR ↔ RGB 转换
    if bgr2rgb:
        np_image = np_image[..., ::-1]  # 反转最后一维（通道）
    
    return np_image

def np2pil(np_image: np.ndarray, rgb2bgr: bool = True) -> Image.Image:
    """
    将NumPy数组转换为PIL图像（支持RGB/BGR格式转换）
    
    Args:
        np_image: 形状为 (H, W, 3) 的uint8数组
        rgb2bgr: 是否将RGB格式转换为BGR（OpenCV默认使用BGR）
    
    Returns:
        PIL.Image.Image: RGB模式的PIL图像
    """
    # 确保输入为uint8类型
    if np_image.dtype != np.uint8:
        np_image = np.clip(np_image, 0, 255).astype(np.uint8)
    
    # BGR ↔ RGB 转换
    if rgb2bgr:
        np_image = np_image[..., ::-1]
    
    # 转换为PIL图像（自动处理RGB顺序）
    return Image.fromarray(np_image)


def replace_objects(tool_object_str: str, new_objects: List) -> str:
    """
    用新的物体列表替换原字符串中的物体
    :param tool_object_str: 原始工具-物体字符串
    :param new_objects: 新的物体列表
    :return: 替换后的新字符串
    """
    pairs = tool_object_str.split(',')
    if len(pairs) != len(new_objects):
        raise ValueError("新物体列表长度与原物体数量不匹配")
    
    new_pairs = []
    for i, pair in enumerate(pairs):
        if ':' in pair:
            tool, _ = pair.split(':', 1)
            new_pairs.append(f"{tool}:{new_objects[i]}")
        else:
            new_pairs.append(pair)
    
    return ','.join(new_pairs)

def annotate(image_pil, masks, boxes, mode=["blend_masks", "blend_boxes", "number_masks", "outline_masks"], output_path="visualized_image.png"):
    """
    可视化函数：将 masks 和 boxes blend 到原图像上，并保存结果。
    
    参数:
        image_pil (PIL.Image): 原始图像。
        masks (np.ndarray): N x H x W 的 masks 数组。
        boxes (np.ndarray): N x 4 的 boxes 数组，每行为 (x1, y1, x2, y2)。
        mode (str): 模式，可选 "blend_masks", "blend_boxes", "blend_masks_boxes", "number_masks", "outline_masks"。
        output_path (str): 保存路径。
    """
    # 转换原图像为 RGBA 格式（支持透明度）
    image = image_pil.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))  # 创建透明图层
    
    if "blend_masks" in mode:
        # 遍历每个 mask，生成伪随机颜色并 blend 到 overlay 上
        for mask in masks:
            # 随机生成伪随机颜色
            color = tuple([random.randint(0, 255) for _ in range(3)] + [int(255 * 0.3)])  # 0.3 不透明度
            mask_image = Image.fromarray((mask * 255).astype(np.uint8))  # 转换为 8 位灰度图
            mask_colored = Image.new("RGBA", image.size, color)  # 创建彩色 mask
            overlay.paste(mask_colored, (0, 0), mask_image)  # 使用 mask 作为 alpha 通道进行叠加
    
    if "blend_boxes" in mode:
        # 在 overlay 上绘制 boxes
        draw = ImageDraw.Draw(overlay)
        for box in boxes:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0, 255), width=2)  # 绿色边框，宽度为 2
    
    if "number_masks" in mode:
        # 在每个 mask 的中心添加编号
        draw = ImageDraw.Draw(overlay)
        for i, mask in enumerate(masks):
            # 计算 mask 的中心
            center_y, center_x = center_of_mass(mask)
            text = str(i + 1)
            font_size = 20
            # font = ImageFont.truetype("arial.ttf", font_size)  # 替换为系统可用字体
            try:
                font = ImageFont.truetype("arial.ttf", 20)  # 替换为具体字体路径
            except OSError:
                print("Custom font not found, using default font.")
                font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # 添加黑色底的白色数字
            rect_start = (int(center_x - text_width / 2) - 2, int(center_y - text_height / 2) - 2)
            rect_end = (int(center_x + text_width / 2) + 2, int(center_y + text_height / 2) + 2)
            draw.rectangle([rect_start, rect_end], fill=(0, 0, 0, 255))
            draw.text((center_x - text_width / 2, center_y - text_height / 2), text, fill=(255, 255, 255, 255), font=font)
    
    if "outline_masks" in mode:
        # 绘制 mask 的边界
        draw = ImageDraw.Draw(overlay)
        for mask in masks:
            # 使用 OpenCV 提取轮廓
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                # 转换轮廓为坐标列表
                points = [(int(point[0][0]), int(point[0][1])) for point in contour]
                # 绘制边界
                draw.line(points + [points[0]], fill=(0, 0, 0, 255), width=3)
    
    # 将 overlay 和原图像 blend
    blended = Image.alpha_composite(image, overlay)
    
    # 转换为 RGB 格式（去除 alpha 通道）并保存
    blended = blended.convert("RGB")
    blended.save(output_path)
    print(f"Visualized image saved to {output_path}")
    return blended

# ==============================================================================================
# ==============================================================================================
# ==============================================================================================

    # 旋转矩阵
def rotate_matrix(axis, theta):
    if axis == 'x':
        T = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
    elif axis == 'y':
        T = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
    elif axis == 'z':
        T = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError()

    return T

def depth2pc(depth_image, fx=605.2769, fy=604.6624, cx=325.3693, cy=252.1953, vis_flag=1):
    '''
    将像素索引的深度图通过计算得到像素索引的点云
    '''

    # 深度单位变为m
    depth_image = depth_image / 1000
    rows, cols = depth_image.shape

    # 使用 meshgrid 生成行和列的网格坐标
    rows_grid, cols_grid = np.meshgrid(np.arange(rows), np.arange(cols))
    # 打印行和列的网格坐标
    pc = np.concatenate(
        [rows_grid.T[:, :, np.newaxis], cols_grid.T[:, :, np.newaxis], depth_image[:, :, np.newaxis]], axis=-1) # (H, W, 3)
    # 这时pc是一个三维array，最后一个维度的长度是3，其内容与索引值存在重叠，如pc[x, y]的值是[x, y, depth]

    pc = pc.reshape(-1, 3) # (H*W, 3)
    # 计算在相机坐标系下的坐标值
    pc[:, 0] = (pc[:, 0] - cy) / fy * pc[:, 2]
    pc[:, 1] = (pc[:, 1] - cx) / fx * pc[:, 2]


    # 显示整幅图像的点云
    if vis_flag in [2]:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pc)  # 将点转换为Open3D的Vector3dVector格式
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([point_cloud, mesh_frame])

    pc = pc.reshape(rows, cols, 3)
    pc = pc[:, :, [1, 0, 2]]  # 切换为行列索引

    return pc

def get_spatial_relation(pos1, pos2):
    '''
    根据两个物体的三维坐标，得到两者信息量最大的方向描述
    '''
    # 计算相对向量
    relative_vector = [pos2[i] - pos1[i] for i in range(3)]
    
    # 计算每个方向的绝对值（权重）
    direction_weights = {
        'on the right of': abs(relative_vector[0]) if relative_vector[0] > 0 else 0, # 
        'on the left of': abs(relative_vector[0]) if relative_vector[0] < 0 else 0, # 
        'behind': abs(relative_vector[1]) if relative_vector[1] < 0 else 0, # 
        'in front of': abs(relative_vector[1]) if relative_vector[1] > 0 else 0, # 
        'under': abs(relative_vector[2]) if relative_vector[2] > 0 else 0, # 上
        'above': abs(relative_vector[2]) if relative_vector[2] < 0 else 0, # 下
    }
    
    # 过滤掉权重为0的方向
    direction_weights = {k: v for k, v in direction_weights.items() if v > 0}
    
    # 如果没有显著方向，则认为是同一位置
    if not direction_weights:
        return 'at the same position with'
    
    # 按权重从大到小排序
    sorted_directions = sorted(direction_weights.keys(), key=lambda x: direction_weights[x], reverse=True)
    
    # 选择至多两个最显著的方向
    selected_directions = sorted_directions[0]
    
    # 组合方向
    return ''.join(selected_directions)

def get_description_from_depth(depth_image, masks, label, erosion_scale=0.1, vis_flag=1):
    '''
    通过点云数据和每个物体的mask得到物体之间的相对位置关系
        depth_image: array (H, W)
        masks: array (N, H, W) dtype=bool
        label: 逗号分割的字符串，顺序与mask一致
        erosion_scale: 腐蚀掉多少比例的mask面积
    '''
    # 预处理============================
    # 目前得到的点云没有xy方向上的距离信息，只有索引
    pc = depth2pc(depth_image, vis_flag=vis_flag) # (H, W, 3)
    position = {}
    label_list = [item.strip() for item in label.split(',')]
    for label_ in label_list:
        position[label_] = None

    # 为可视化提前准备各个mask的颜色数组
    # 基于 z 坐标设置颜色
    z_min, z_max = pc[:, :, 2].min(), pc[:, :, 2].max()
    pc_color = (pc[:, :, 2] - z_min) / (z_max - z_min)  # 归一化到 [0, 1]
    pc_color = np.tile(pc_color[:, :, np.newaxis], (1, 1, 3))  # 扩展到 RGB

    for i, mask in enumerate(masks):
        # 对每个mask进行腐蚀
        mask_area = np.sum(mask)
        # 将每个mask抽象成正方形，计算结构元素大小
        kernal_size = max(1, int((np.sqrt(mask_area) - np.sqrt((1 - erosion_scale) * mask_area)) / 2))
        kernel = np.ones((kernal_size, kernal_size), np.uint8)
        # 进行腐蚀操作
        int_mask = mask.astype(np.uint8) * 255 # dtype=int
        erode_mask = cv2.erode(int_mask, kernel, iterations=1)
        # 根据腐蚀后的mask计算平均的xy
        erode_mask = erode_mask.astype(np.bool_)
        xy = np.where(erode_mask==True) # (array([x1, x2, x3]), array([y1, y2, y3]))
        # 计算平均坐标
        x_mean = np.mean(pc[xy[0], xy[1], 0]) 
        y_mean = np.mean(pc[xy[0], xy[1], 1]) 
        z_mean = np.mean(pc[xy[0], xy[1], 2]) 
        position[label_list[i]] = np.array([x_mean, y_mean, z_mean])
        # 为mask打上黑色
        pc_color[xy[0], xy[1], :] = [0, 0, 0]

        # 创建o3d点云类，并进行可视化================================
        # 可视化每一个mask
        if vis_flag in [2]:

            # 实例化一个 PointCloud 对象
            pcd = o3d.geometry.PointCloud()
            # 创建坐标轴
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
            # 将 NumPy 数组转换为 Open3D 点云对象
            pcd.points = o3d.utility.Vector3dVector(pc[xy])
            # 可视化点云
            o3d.visualization.draw_geometries([pcd, mesh_frame])
    # 可视化所有点云，为每个mask赋予不同的颜色
    if vis_flag in [1, 2]:

        # 实例化一个 PointCloud 对象
        pcd = o3d.geometry.PointCloud()
        # 创建坐标轴
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
        

        # 将 NumPy 数组转换为 Open3D 点云对象
        pcd.points = o3d.utility.Vector3dVector(pc.reshape(-1, 3))
        pcd.colors = o3d.utility.Vector3dVector(pc_color.reshape(-1, 3))
        # 可视化点云
        o3d.visualization.draw_geometries([pcd, mesh_frame])

    res = ''
    # 描述阶段==============
    for i in range(len(label_list) - 1):
        for j in range(i + 1, len(label_list)):
            relation = get_spatial_relation(position[label_list[i]], position[label_list[j]])
            res = ','.join([res, f'{label_list[j]}在{label_list[i]}的{relation}方']).strip(',')
            print(f'{label_list[j]}在{label_list[i]}的{relation}方')
    return res

def get_short_edge_angle(mask, vis_flag=False) -> float:
    """
    计算物体最小外接矩形的短边方向角度（长边的法向，相对于x轴）。
    
    参数:
        mask (np.ndarray): 二值化掩码，背景=0，物体>0（支持bool/int/float类型）。
    
    返回:
        float: 短边角度（单位：度），范围 [-90, 90]。
              如果 mask 全零或无轮廓，返回 None。
    """
    # 1. 转换 mask 为 uint8 类型
    mask_uint8 = np.uint8(mask > 0) * 255

    # 2. 提取轮廓
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)

    # 3. 计算最小外接旋转矩形
    (center, (width, height), angle) = cv2.minAreaRect(largest_contour)  # angle ∈ [0, 90]

    # 4. 计算短边方向（长边的法向）
    if width > height:
        short_angle = angle + 90  # 长边是width，短边是height，法向=angle+90°
    else:
        short_angle = angle - 90  # 长边是height，短边是width，法向=angle-90°

    # 5. 规范化角度到 [0°, 180°]
    short_angle = short_angle % 180

    if vis_flag:
        # 创建白底图像
        vis = np.ones((*mask.shape, 3), dtype=np.uint8) * 255
        
        # 绘制旋转矩形（绿色）
        box = cv2.boxPoints((center, (width, height), angle))
        box = np.int0(box)
        cv2.drawContours(vis, [box], 0, (0, 255, 0), 2)
        
        # 绘制短边方向箭头（红色）
        arrow_length = max(width, height) * 0.6  # 箭头长度为长边的60%
        end_point = (
            int(center[0] + arrow_length * np.cos(np.deg2rad(short_angle))),
            int(center[1] + arrow_length * np.sin(np.deg2rad(short_angle))))
        cv2.arrowedLine(vis, (int(center[0]), int(center[1])), end_point, (0, 0, 255), 3, tipLength=0.2)
        
        save_and_pause(vis)

    return short_angle

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================

def erode_mask(
    mask: Optional[Union[np.ndarray, None]], 
    size: int = 3, 
    iterations: int = 1
) -> Optional[Union[np.ndarray, None]]:
    """
    对输入的mask进行腐蚀操作，并返回与输入类型相同的mask
    
    Args:
        mask: 输入的mask，可能是None或NumPy数组（可以是bool、uint8、float等类型）
        size: 腐蚀核的大小（默认3）
        iterations: 腐蚀操作的迭代次数（默认1）
    
    Returns:
        腐蚀后的mask，类型与输入相同（如果输入是None，返回None）
    """
    # 如果mask是None或空数组，直接返回
    if mask is None or (isinstance(mask, np.ndarray) and mask.size == 0):
        return mask
    
    # 确保mask是NumPy数组
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"Expected mask to be a numpy.ndarray or None, got {type(mask)}")
    
    # 保存原始数据类型
    original_dtype = mask.dtype
    
    # 将mask转换为uint8类型（cv2.erode要求输入是uint8或float32）
    if mask.dtype == bool:
        mask_uint8 = mask.astype(np.uint8) * 255
    elif np.issubdtype(mask.dtype, np.integer):
        mask_uint8 = mask.astype(np.uint8)
    elif np.issubdtype(mask.dtype, np.floating):
        mask_uint8 = (mask * 255).astype(np.uint8)
    else:
        raise TypeError(f"Unsupported mask dtype: {mask.dtype}")
    
    # 创建腐蚀核
    kernel = np.ones((size, size), np.uint8)
    
    # 执行腐蚀操作
    eroded = cv2.erode(mask_uint8, kernel, iterations=iterations)
    
    # 转换回原始数据类型
    if original_dtype == bool:
        eroded_mask = (eroded > 0).astype(bool)
    elif np.issubdtype(original_dtype, np.integer):
        eroded_mask = eroded.astype(original_dtype)
    elif np.issubdtype(original_dtype, np.floating):
        eroded_mask = (eroded / 255.0).astype(original_dtype)
    else:
        eroded_mask = eroded  # 默认返回uint8
    
    return eroded_mask