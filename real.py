from utils.robot import *
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

FAR_OBS_JOINT = [2.768, 44.95, -69.414, -6.492, -84.351, 0.95]
V = 15

if __name__ == '__main__':
    # 读取 config.ini
    config = configparser.ConfigParser()
    config.read('./utils/real.ini')  # 自动处理文件不存在的情况

    api_key = config.get('LLM', 'api_key')
    api_base = config.get('LLM', 'api_base')
    gpt_model = config.get('LLM', 'gpt_model')
    ckpt_path = config.get('PATH', 'ckpt_path')
    manual_path = config.get('PATH', 'manual_path')
    dataset_path = config.get('PATH', 'dataset_path')
    log_dir = config.get('PATH', 'log_dir')
    manual_save_path = config.get('PATH', 'manual_save_path')
    grounding_enable = config.getboolean('MODE', 'grounding_enable')
    point_cloud_enable = config.getboolean('MODE', 'point_cloud_enable')
    vis_flag = config.getint('MODE', 'vis_flag')
    spawn_n = config.getint('ROBOT', 'spawn_n')
    erode_size = config.getint('ROBOT', 'erode_size')
    plan_mode = [item.strip() for item in config.get('MODE', 'plan_mode').split(',') if item.strip()] # List[str]
    
    if point_cloud_enable:
        grounding_enable = True

    # init
    # 初始化日志（自动命名）
    setup_logging('./log')
    logging.getLogger("httpx").propagate = False
    logging.getLogger("openai._base_client").propagate = False

    if not grounding_enable:
        ckpt_path = None

    episode = Episode(dataset_path)
    # 根据mode在init中初始化硬件设备
    robot = Robot(ckpt_path, api_key, api_base, gpt_model, manual_path,
     g_enable=grounding_enable, pc_enable=point_cloud_enable, mode='real', manual_save_path=manual_save_path)
    robot.conversation_dir = './conversation/real/'
    
    # 写入日志
    logging.info(f"start, mode: real")
    # 预热摄像头
    time.sleep(1) 
    # 清理可视化文件夹
    clean_directory('vis')

# ==========================================================================================
# 出生点
# ==========================================================================================

    # 移动到出生点（可选
    # 遍历出生点观测
    spawn_images = robot.spawn_observe(1, vis_flag=2)
    
        
    # 处理所有出生点观测的图像
    for i in range(len(spawn_images)):
        label_temp = robot.llm_infer('navigation.txt', [np2pil(spawn_images[i][0])], [], True, True, mode=[])
        
        # label_temp = 'red can:can, clear bottle:bottle'
        if label_temp != '无':
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

            # 点云处理
            if point_cloud_enable and robot.nav_mem.get_values_by_key("label"):
                depth_img = spawn_images[i][1] # array
                g_label_keys = robot.nav_mem.get_values_by_key('g_label')
                for g_label_key in g_label_keys:
                    mask_bool = robot.nav_mem.get('g_label', g_label_key)[0]['mask'].astype(np.bool_)
                    # 获取相机坐标系下的经过mask筛选的平均坐标
                    pc = robot.cam.get_point_cloud(depth_image=depth_img) # array (H, W, 3)
                    mean_xyz = np.mean(pc[mask_bool], axis=0)
                    # 将坐标转化为world坐标系
                    pass
                    # 在记忆中添加坐标
                    robot.nav_mem.update('g_label', g_label_key, 'cor', mean_xyz)
                    logging.info(f'added cor: {mean_xyz} to navigation mem: {g_label_key}')


                # 分析空间关系
                # res = get_description_from_depth(depth_img, masks.astype(np.bool_), label_1)
            
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
        robot.move_to(navigation_point)
        # 根据导航点设置近距离观察角度
        observation_joint = GROUND_OBS_JOINT
        temp = ''.join([navigation_point.upper(), '_OBS_JOINT'])
        if temp in globals():
            observation_joint = globals()[temp]
        # 近距离观察
        robot.arm.setJoints(observation_joint)
        # 获取图像
        color_image, depth_image, depth_frame = robot.cam.get_frames()

        # perception
        label_1 = robot.llm_infer('perception.txt', [np2pil(color_image)], [], False, True)
        
        # 解析标签，分流为grounding用和分析用
        g_labels, labels = sort_label(label_1)
        for g_label, label in zip(str2list(g_labels), str2list(labels)):
            robot.scene_mem.add(g_label, label=label)
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
        logging.info(f"action_objects: {action_objects}")
        if action_objects != '0':
            # 对于每一个动作
            for action_object in action_objects.split(','):
                action, object_ = action_object.split(':')
                    robot.action(depth_img=depth_image, label=object_)
                # 感知操作之后的场景
                before_img = color_image
                before_label = ','.join(robot.scene_mem.get_values_by_key("label"))
                color_image, depth_image, depth_frame = robot.cam.get_frames()
                after_img = color_image
                robot.compare(before_img, after_img, before_label, action)



