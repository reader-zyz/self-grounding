from utils.robot_sim import *
from utils.episode import Episode
from utils.annotate import *
from utils.calibration import Convert
from PIL import Image
import configparser
from utils.logger import setup_logging
import logging
import numpy as np
import cv2
import time

def load_image_pairs(directory: str) -> List[List[np.ndarray]]:
    """
    读取指定目录下的PNG和同名NPY文件，生成spawn_images格式的数据
    
    参数:
        directory: 包含PNG和NPY文件的目录路径
        
    返回:
        spawn_images格式的列表，每个元素是[image_array, depth_array]对
        
    异常:
        FileNotFoundError: 如果目录不存在
        ValueError: 如果PNG和NPY文件不匹配
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"目录不存在: {directory}")
    
    # 获取目录下所有PNG文件
    png_files = [f for f in os.listdir(directory) if f.lower().endswith('.png')]
    if not png_files:
        raise ValueError(f"目录中没有PNG文件: {directory}")
    
    spawn_images = []
    
    for png_file in png_files:
        # 构建对应的NPY文件名
        base_name = os.path.splitext(png_file)[0]
        npy_file = base_name + '.npy'
        npy_path = os.path.join(directory, npy_file)
        
        # 检查NPY文件是否存在
        if not os.path.exists(npy_path):
            raise ValueError(f"找不到与 {png_file} 对应的NPY文件: {npy_path}")
        
        # 读取PNG文件
        png_path = os.path.join(directory, png_file)
        try:
            img_array = pil2np(Image.open(png_path))
        except Exception as e:
            raise ValueError(f"无法读取PNG文件 {png_file}: {str(e)}")
        
        # 读取NPY文件
        try:
            depth_array = np.load(npy_path)
        except Exception as e:
            raise ValueError(f"无法读取NPY文件 {npy_file}: {str(e)}")
        
        # 检查尺寸是否匹配
        if img_array.shape[:2] != depth_array.shape[:2]:
            raise ValueError(
                f"图像和深度图尺寸不匹配: {png_file} {img_array.shape[:2]} vs {npy_file} {depth_array.shape[:2]}"
            )
        
        # 添加到结果列表
        spawn_images.append([img_array, depth_array])
    
    return spawn_images

def get_corresponding_npy_path(image_path: str) -> str:
    """
    根据图像路径获取同目录下同名的npy文件路径
    
    参数:
        image_path: 图像文件路径 (如: '/path/to/image.png')
        
    返回:
        对应的npy文件路径 (如: '/path/to/image.npy')
        
    异常:
        ValueError: 如果输入路径不是有效的图像文件路径
    """
    # 检查输入路径是否有效
    if not os.path.isfile(image_path):
        raise ValueError(f"输入路径不是有效的文件: {image_path}")
    
    # 分离目录、基本名和扩展名
    dir_name = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 构建npy文件路径
    npy_path = os.path.join(dir_name, f"{base_name}.npy")
    
    return npy_path

if __name__ == '__main__':
    # 读取 config.ini
    config = configparser.ConfigParser()
    config.read('./utils/simulate.ini')  # 自动处理文件不存在的情况

    api_key = config.get('LLM', 'api_key')
    api_base = config.get('LLM', 'api_base')
    gpt_model = config.get('LLM', 'gpt_model')
    ckpt_path = config.get('PATH', 'ckpt_path')
    manual_path = config.get('PATH', 'manual_path')
    dataset_path = config.get('PATH', 'dataset_path')
    log_dir = config.get('PATH', 'log_dir')
    manual_save_path = config.get('PATH', 'manual_save_path')
    erode_size = config.getint('ROBOT', 'erode_size')
    grounding_enable = config.getboolean('MODE', 'grounding_enable')
    point_cloud_enable = config.getboolean('MODE', 'point_cloud_enable')
    plan_mode = [item.strip() for item in config.get('MODE', 'plan_mode').split(',') if item.strip()] # List[str]

    # init
    # 初始化日志（自动命名）
    setup_logging('./log')
    logging.getLogger("httpx").propagate = False
    logging.getLogger("openai._base_client").propagate = False

    episode = Episode(dataset_path)
    # 根据mode在init中初始化硬件设备
    robot = Robot(ckpt_path, api_key, api_base, gpt_model, manual_path, mode='simulate', manual_save_path=manual_save_path)
    robot.conversation_dir = config.get('PATH', 'conversation_dir')
    
    # 写入日志
    logging.info(f"start, mode: sim")
    # 清理可视化文件夹
    clean_directory('vis')

# ==========================================================================================
# 出生点
# ==========================================================================================

    spawn_images = load_image_pairs(dataset_path)
    
        
    # 处理所有出生点观测的图像
    for i in range(len(spawn_images)):
        label_temp = robot.llm_infer('navigation.txt', [np2pil(spawn_images[i][0])], [], True, True, mode=[])

        if label_temp != 'None':
            g_label_temps, noun_label_temps = sort_label(label_temp)

            g_label_temps = [temp.strip() for temp in g_label_temps.split(',')]
            noun_label_temps = [temp.strip() for temp in noun_label_temps.split(',')]
            for g_label_temp, noun_label_temp in zip(g_label_temps, noun_label_temps):
                robot.nav_mem.add(g_label_temp, label=noun_label_temp)

            logging.info(f'original navigation points: {robot.nav_mem.get_values_by_key("label")}')
            logging.info(f'valid navigation points: {list(episode.scenes.keys())}')
            # 对齐
            robot.align_navigation_point(','.join(list(episode.scenes.keys())))
            logging.info(f'aligned navigation points: {robot.nav_mem.get_values_by_key("label")}')

            # grounding
            if grounding_enable and robot.nav_mem.get_values_by_key("label"):
                # infer_img 会自动按照逗号分割字符串label
                logging.info(f'going to grounding, labels: {union_str(",".join(robot.nav_mem.get_values_by_key("g_label")))}, len: {len(robot.nav_mem.get_values_by_key("g_label"))}')
                # Langsam的输出顺序是按照label的字母排序来的
                blend_img, labels, masks = robot.infer_img(spawn_images[i][0], union_str(','.join(robot.nav_mem.get_values_by_key("g_label"))), output_path=f'vis/navigation_paint_{i}.png')
                
                # 将labels还原回输入
                for i in range(len(labels)):
                    labels[i] = separate_str(labels[i])

                # 按照label将mask添加到记忆中
                for label, mask in zip(labels, masks):
                    robot.nav_mem.update('g_label', label, 'mask', mask)
                
                # 删除所有没有mask的物体
                for g_label in robot.scene_mem.get_values_by_key('g_label'):
                    if robot.scene_mem.get('g_label', g_label)[0]['mask'] is None:
                        robot.scene_mem.delete('g_label', g_label)
            
    logging.info(f'navigation points: {list2str(robot.nav_mem.get_values_by_key("label"))}')
    
# ==========================================================================================
# 导航点
# ==========================================================================================

    # 遍历每一个导航点
    for navigation_point in robot.nav_mem.get_values_by_key("label"):
        # 清空scene mem
        robot.scene_mem.clear_all()
        episode.state = navigation_point
        logging.info(f'going to: {navigation_point}')
        # 导航到导航点
        # robot.move_to(navigation_point)

        while(True):

            # 近距离观察
            # 获取图像
            observation_path = episode.get_image()
            color_image = pil2np(Image.open(observation_path))

            # perception
            label_1 = robot.llm_infer('perception.txt', [np2pil(color_image)], [], False, True)

            # 解析标签，分流为grounding用和分析用
            g_labels, labels = sort_label(label_1)
            for g_label, label in zip(str2list(g_labels), str2list(labels)):
                robot.scene_mem.add(g_label, label=label)
            before_label = list2str(robot.scene_mem.get_values_by_key("label"))
            logging.info(f'before objects: {list2str(robot.scene_mem.get_values_by_key("label"))}')

            if grounding_enable:
                logging.info(f"going to grounding, labels: {g_labels}, len: {len(g_labels.split(','))}")
                blend_img, g_labels, masks = robot.infer_img(color_image, union_str(g_labels), output_path=f'vis/{navigation_point}.png')
                
                # 将labels还原回输入
                for i in range(len(g_labels)):
                    g_labels[i] = separate_str(g_labels[i])
                
                logging.info(f"after SAM labels: {g_labels}, len masks: {len(masks)}")
                for g_label, mask in zip(g_labels, masks):
                    # 对图像进行腐蚀
                    if erode_size > 1:
                        mask = erode_mask(mask, erode_size)
                    robot.scene_mem.update('g_label', g_label, 'mask', mask)
                # 删除所有没有mask的物体
                for g_label in robot.scene_mem.get_values_by_key('g_label'):
                    if robot.scene_mem.get('g_label', g_label)[0]['mask'] is None:
                        robot.scene_mem.delete('g_label', g_label)

                if point_cloud_enable:
                    depth_img = np.load(get_corresponding_npy_path(observation_path))
                    label_keys = robot.scene_mem.get_values_by_key('label')

                    for label_key in label_keys:
                        mask_bool = robot.scene_mem.get('label', label_key)[0]['mask'].astype(np.bool_)
                        pc = depth2pc(depth_img)  # (H, W, 3)
                        masked_pc = pc[mask_bool]

                        # 计算中心坐标
                        mean_xyz = np.mean(masked_pc, axis=0)
                        
                        # 计算中心到原点的欧氏距离
                        distance_to_origin = np.linalg.norm(mean_xyz)

                        if len(masked_pc) > 1:
                            # 计算AABB边界框的尺寸（X/Y/Z方向的边长）
                            min_coords = np.min(masked_pc, axis=0)
                            max_coords = np.max(masked_pc, axis=0)
                            ranges = max_coords - min_coords  # [ΔX, ΔY, ΔZ]

                            # 计算平均尺寸（三个轴向的平均边长）
                            avg_size = np.mean(ranges) / 2
                        else:
                            avg_size = 0.0

                        # 存储结果
                        robot.scene_mem.update('label', label_key, 'cor', mean_xyz)
                        robot.scene_mem.update('label', label_key, 'size', avg_size)
                        robot.scene_mem.update('label', label_key, 'dis', distance_to_origin)
                        logging.info(f'Added cor: {mean_xyz}, size: {avg_size:.3f}, dis: {distance_to_origin:.3f} to scene mem: {label_key}')

                # 分析空间关系
                label_list = label_keys
                spatial_description = ''
                # 描述阶段==============
                for i in range(len(label_list) - 1):
                    for j in range(i + 1, len(label_list)):
                        relation = get_spatial_relation(robot.scene_mem.get('label', label_list[i])[0]['cor'], robot.scene_mem.get('label', label_list[j])[0]['cor'])
                        spatial_description = ','.join([spatial_description, f'{label_list[j]} is {relation} {label_list[i]}']).strip(',')
                        print(f'{label_list[j]} is {relation} {label_list[i]}')
            extra_txt = []
            if 'space' in plan_mode:
                extra_txt.append(spatial_description)

            original_actions = robot.plan_only(','.join(robot.scene_mem.get_values_by_key("label")), blend_img, mode=['space', 'size'], extra_txt=extra_txt)

            logging.info(f"original_actions: {original_actions}")
            if original_actions == '0':
                break

            actions, objects = sort_label(original_actions)

            try_ = 0
            aligned_objects = None
            while(try_ < 3):
                # 对齐决策中的物体
                aligned_objects = robot.llm_infer('synonym.txt', [], [objects, list2str(episode.scenes[episode.state]['id'])], False, True)
                aligned_objects = str2list(aligned_objects)
                objects = str2list(objects)
                if len(aligned_objects) != len(objects):
                    try_ += 1
                    objects = list2str(objects)
                    continue
                else:
                    break

            # 替换动作-物体对
            action_objects = replace_objects(original_actions, aligned_objects)
            # 替换secne mem
            for i, res_ in enumerate(aligned_objects):
                # 场景中没有对应的物体
                if res_ == '0':
                    # 从scene_mem中删除
                    robot.scene_mem.delete('label', objects[i])
                    pass
                # 场景中有对应的物体，将这个物体对齐
                else:
                    # 更新scene_mem
                    robot.scene_mem.update('label', objects[i], 'label', res_)

            logging.info(f'aligned actions: {action_objects}')

            last_action = None
            action_list = action_objects.split(',')
            total_actions = len(action_list)

            # 遍历每个动作物体对
            for i, action_object in enumerate(action_list, 1):
                action = action_object.split(':')[0].strip()
                object_ = action_object.split(':')[1].strip()

                # 执行操作
                logging.info(f'execute decision: {action} {object_}')
                logging.info(f'before execute: {episode.scenes[episode.state]["objects"]}')
                after_img = episode.action([action], [object_])
                logging.info(f'after execute: {episode.scenes[episode.state]["objects"]}')
                
                # 当当前动作与上一次不同时，或者是最后一个动作时执行compare
                if action != last_action or i == total_actions:
                    # 分析之后图像
                    robot.compare(color_image, after_img, before_label, action)
                
                # 更新上一次的动作
                last_action = action

            if len(episode.scenes[episode.state]["objects"]) == 0:
                break



