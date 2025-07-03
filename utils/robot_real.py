from utils.annotate import *
from utils.calibration import Convert
from utils.myapi import Conversation
from openai import OpenAI
from PIL import Image
from utils.manual_manager import ManualManager
from utils.camera import Realsense
from arm.arm import my_Arm
from agv.my_agv import Agv
import numpy as np
from typing import Optional, Tuple, Union, List, Any, Dict
from langsam.lang_sam import LangSAM
import cv2
import logging
import configparser
import math
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs
import json

load_ini_to_globals('./utils/joint.ini', globals())

class Robot:
    def __init__(self, ckpt_path, api_key, api_base, gpt_model, manual_path, g_enable=False, pc_enable=False, mode='simulate', manual_save_path=None):
        # mode
        self.grounding_enable = g_enable
        self.point_cloud_enable = pc_enable
        self.conversation_dir = './conversation/'
        self.retry = 3
        # 语言分割模型
        self.langsam: Optional[LangSAM] = None
        if ckpt_path:
            self.langsam = LangSAM(ckpt_path=ckpt_path)  
        # chatGPT
        self.gpt = Conversation(OpenAI(api_key=api_key, base_url=api_base), gpt_model)
        # 说明书
        self.manual = ManualManager(manual_path, manual_save_path)
        # 设置记忆存储，分别记录导航和场景中的物体
        self.nav_mem = Memory(['g_label', 'label', 'mask', 'cor'])
        self.scene_mem = Memory(['g_label', 'label', 'mask', 'cor'])
        # 机械臂
        self.arm: Optional[my_Arm] = None
        if mode == 'real':
            self.arm = my_Arm(wifi=False)
        # 摄像头
        self.cam: Optional[Realsense] = None
        if mode == 'real':
            self.cam = Realsense()
        # 底盘
        self.agv: Optional[Agv] = None
        if mode == 'real':
            self.agv = Agv()
        
    # 将 utils.annotate 转换为方法，是涂鸦的方法
    def paint(self, image_pil, masks, boxes, mode=["blend_masks", "blend_boxes", "number_masks", "outline_masks"], output_path="visualized_image.png"):
        return annotate(image_pil, masks, boxes, mode, output_path)

    def llm_infer(self, conversation_name, img_list: Union[List[str], List[Image.Image], List[np.ndarray]], text_list, print_flag=False, vis_flag=False, mode=[]):
        observation_list = []
        for img in img_list:
            if isinstance(img, str):  # 如果是路径字符串
                observation_list.append(Image.open(img))
            elif isinstance(img, Image.Image):  # 如果是PIL图像对象
                observation_list.append(img)
            elif isinstance(img, np.ndarray):  # 如果是array
                observation_list.append(np2pil(img))
            else:
                raise TypeError(f"Unsupported type: {type(img)}. Expected str or PIL.Image.")
        self.gpt.read_txt(self.conversation_dir + conversation_name, observation_list, text_list, mode=mode)
        if print_flag:
            print(self.gpt.get_lastest_ans())
        if vis_flag:
            self.gpt.get_message('./output')
        res = self.gpt.get_lastest_ans()
        self.gpt.clear_message()
        return res

    def infer_img(self, img, label: str, output_path, blend_mode=["number_masks", "outline_masks"]) -> Tuple[Image.Image, List[str], List[np.ndarray]]:
        '''
            img: PIL图像路径
            label: 标签，格式上是逗号分割的字符串
        '''
        if isinstance(img, str):  # 如果是路径字符串
            img = Image.open(img)
        elif isinstance(img, Image.Image):  # 如果是PIL图像对象
            pass
        elif isinstance(img, np.ndarray):  # 如果是NumPy数组（OpenCV格式）
            # OpenCV格式检查与转换
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            
            # 处理BGR->RGB转换（OpenCV默认BGR，PIL需要RGB）
            if len(img.shape) == 3 and img.shape[2] == 3:  # 如果是彩色图像
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img = Image.fromarray(img)
        else:
            raise TypeError(
                f"Unsupported type: {type(img)}. "
                "Expected: str (path), PIL.Image or numpy.ndarray (OpenCV format)."
            )

        # 通过语言分割模型对标签进行grounding
        text_prompt = '.'.join(label.split(',')) # 'item1. item2.'
        print(f'text_prompt {text_prompt}')
        results = self.langsam.predict([img], [text_prompt]) # results->list[dict{'scores', 'labels', 'boxes', 'masks', 'masks_score'}]
        # 对grounding的mask进行涂鸦
        labels = results[0]['labels'] 
        masks = results[0]['masks'] # dtype=float
        boxes = results[0]['boxes']
        image_blend = self.paint(img, masks, boxes, mode=blend_mode, output_path=output_path)
        return image_blend, labels, masks

    def plan_only(self, label, img): 
        # 进行决策
        feasibility = self.llm_infer('feasibility.txt', [img], [self.manual.convert_to_string(), label], False, False)
        consequence = self.llm_infer('consequence.txt', [img], [self.manual.convert_to_string(), label], False, False)
        security = self.llm_infer('security.txt', [img], [self.manual.convert_to_string(), label], False, False)
        res = self.llm_infer('synthesize.txt', [img], [self.manual.convert_to_string(), label, feasibility, consequence, security], False, True)

        return res

    def plan(self, img): 
        '''
        根据当前场景进行开放词汇检测
            img: 路径
        '''
        # 分析当前图像
        label_1 = self.llm_infer('perception.txt', [img], [], False, False)
        # 解析标签，分流为grounding用和分析用
        g_label_1, label_1 = sort_label(label_1)
        for g_label, label_1 in zip(g_label_1, label_1):
            self.scene_mem.add(g_label, {'label': label_1})
        # 进行task grounding
        if self.grounding_enable:
            pass
        if self.point_cloud_enable:
            pass
        # 进行决策
        feasibility = self.llm_infer('feasibility.txt', [img], [self.manual.convert_to_string(), label_1], False, False)
        consequence = self.llm_infer('consequence.txt', [img], [self.manual.convert_to_string(), label_1], False, False)
        security = self.llm_infer('security.txt', [img], [self.manual.convert_to_string(), label_1], False, False)
        res = self.llm_infer('synthesize.txt', [img], [self.manual.convert_to_string(), label_1, feasibility, consequence, security], False, True)

        return res

    def compare(self, before_img, after_img, label_1, tool):
        '''
        根据清洁前后的变化总结经验并更新说明书
            before_img: 路径
            after_img: 路径
            label_1: before图像的标签
            tool: 本次操作使用的工具
        '''
        # 分析之后图像
        label_2 = self.llm_infer('perception.txt', [after_img], [], True, True)
        # 解析标签，分流为grounding用和分析用
        _, label_2 = sort_label(label_2)
        # 分析出说明书的新格式
        label_1_new = []
        res = self.llm_infer('synonym.txt', [], [label_1, ','.join(self.manual.get_subkeys(f'{tool}.可供性'))], False, True)
        res = str2list(res)
        for i, res_ in enumerate(res):
            item = str2list(label_1)[i]
            # 如果说明书中不存在近义词
            if res_ == '0':
                self.manual.add_prompt(f'{tool}.可供性.{item}', ' ')
                label_1_new.append(item)
            # 说明书中存在近义词
            else:
                label_1_new.append(res_)

        label_1 = ','.join(label_1_new)

        # 对清洁问题进行分类
        res = self.llm_infer('synonym.txt', [], [label_1, label_2], False, True)
        res = str2list(res)
        for i, res_ in enumerate(res):
            item = str2list(label_1)[i]
            # 物体消失，清洁成功
            if res_ == '0':
                self.manual.add_prompt(f'{tool}.可供性.{item}', f'{tool}可以清理{item}')
            # 物体依然存在，清洁无效
            else:
                self.manual.add_prompt(f'{tool}.可供性.{item}', f'{tool}不能清理{item}')  

        res = self.llm_infer('synonym.txt', [], [label_2, label_1], False, True)
        res = str2list(res)
        for i, res_ in enumerate(res):
            # 新出现的物体，说明二次污染
            if res_ == '0':
                notice = self.llm_infer('secondary_contamination.txt', ["./before_1.jpg", after_img], [label_1, label_2, label_2, '夹爪'], False, True)
                item, notice = sort_label(notice)
                current_manual = self.manual.get_prompt(f'{tool}.可供性.{item}')
                # 如果本身存在注意事项，需要进行注意事项的融合
                notice = self.llm_infer('merge_notice.txt', [], [current_manual, notice], False, True)
                self.manual.add_prompt(f'{tool}.可供性.{item}', notice) 
            # 物体以前存在，清洁无效
            else:
                pass

    def align_navigation_point(self, valid_nav_points: str):
        # 设置重试次数
        retry = 0
        # 获取当前的导航点用于备份和检索
        # robot_nav_point = self.nav_mem.get_all_labels()
        robot_nav_point = self.nav_mem.get_values_by_key('label')
        while(retry < self.retry):
            retry += 1
            # 匹配导航点到场景
            res = self.llm_infer('synonym.txt', [], [list2str(self.nav_mem.get_values_by_key('label')), valid_nav_points], True, True)
            res = str2list(res)
            # 判断数量是否一致
            if len(res) != len(robot_nav_point):
                continue
            for i, res_ in enumerate(res):
                # 场景中没有对应的导航点
                if res_ == '0':
                    # 从nav mem中删除
                    # self.nav_mem.delete_by_label(robot_nav_point[i])
                    self.nav_mem.delete('label', robot_nav_point[i])
                # 场景中有对应的导航点，将这个导航点保存
                else:
                    # 更新nav mem
                    # self.nav_mem.update_label(robot_nav_point[i], res_)
                    self.nav_mem.update('label', robot_nav_point[i], 'label', res_)
            break
        logging.info(f'aligned navigation points, before: {robot_nav_point}, after: {self.nav_mem.get_values_by_key("label")}')

    def move_to_marker(self, marker):

        self.agv.Goto(marker)
        print("agv will walking to " + marker)
        time.sleep(0.5)
        while not self.agv.arrived(marker):
            time.sleep(0.1)
        print("agv have arrived at " + marker)
        return

    def move_to(self, delta_x, delta_y, dis=0., theta='target'):
        """

        delta_x, delta_y: 相对于机械臂坐标系移动的相对距离，单位米
        dis: 距离目标点dis米停下，不完全到达目标点
        theta: 设置停下后底盘的方向，'target'指向目标，'stay'保留原theta，其他数字则制定相对于机械臂坐标系的制定角度
        """

        coordinate = self.agv.get_target_location(delta_x, delta_y, dis)
        if theta in ['target']:
            self.agv.Goto_Location(*coordinate)
        if theta in ['stay']:
            phi = self.agv.getRobotStatus()['results']['current_pose']['theta'] # current_pose: {'x', 'y', 'theta'}，单位米、弧度
            self.agv.Goto_Location(coordinate[0], coordinate[1], phi)
        if isinstance(theta, (int, float, np.ndarray)):
            self.agv.Goto_Location(coordinate[0], coordinate[1], theta)

    def spawn_observe(self, spawn_n, vis_flag=0) -> List[Tuple[np.ndarray, np.ndarray, rs.depth_frame]]:
        spawn_images = []
        position_now = self.agv.getRobotStatus()['results']['current_pose']
        x, y, phi_0 = position_now['x'], position_now['y'], position_now['theta']
        # 保存现在的joint
        original_joint = self.arm.getState()[0]
        # 设置远距离观察joint
        self.arm.setJoints(FAR_OBS_JOINT)
        for i in range(spawn_n):
            # 将i加1可以保证回到原位
            phi = phi_0 + (i + 1) * (2 * math.pi / spawn_n)
            if phi > math.pi:
                phi -= 2 * math.pi
            # self.agv.Goto_Location(x, y, phi)
            spawn_images.append(self.cam.get_frames())
            if vis_flag == 1:
                save_and_pause(spawn_images[-1][0], pause=False)
            elif vis_flag == 2:
                save_and_pause(spawn_images[-1][0], pause=True)
        # 回调为原始的joint
        self.arm.setJoints(original_joint)
        return spawn_images

    def action(self, depth_img: np.ndarray, label=None, mask=None):
        """
            Due to commercial confidentiality, the code cannot be released publicly
        """
        pass

class Memory:
    def __init__(self, all_keys=None):
        """
        初始化存储系统（无索引版）
        :param all_keys: 所有可能的键列表，如 ['g_label', 'name', 'age']
                         必须包含主键 'g_label'，默认为 ['g_label']
        """
        if all_keys is None:
            all_keys = ['g_label']
        elif 'g_label' not in all_keys:
            raise ValueError("必须包含主键 'g_label'")

        self.all_keys = all_keys  # 所有预定义的键
        self.data = []  # 主数据存储（列表字典）
        
        # 为每个键初始化一个空列表（兼容旧代码）
        for key in all_keys:
            setattr(self, key, [])

    def clear_all(self):
        """清空所有数据"""
        self.data = []
        for key in self.all_keys:
            getattr(self, key).clear()

    def add(self, g_label: Any, **kwargs: Any) -> int:
        """
        添加新记录（自动补全缺失的键为None）
        :param g_label: 主键值（任意类型）
        :param kwargs: 其他键值对
        :return: 新记录的索引
        :raises KeyError: 如果尝试添加未定义的键
        """
        undefined_keys = set(kwargs.keys()) - set(self.all_keys)
        if undefined_keys:
            raise KeyError(f"尝试添加未定义的键: {undefined_keys}")

        # 创建完整记录
        record = {key: None for key in self.all_keys}
        record.update({'g_label': g_label, **kwargs})
        
        # 存储数据
        self.data.append(record)
        for key in self.all_keys:
            getattr(self, key).append(record[key])
        
        return len(self.data) - 1

    def get(self, key: str, value: Any) -> List[Dict]:
        """
        线性搜索获取指定键值对应的所有记录
        :param key: 查询键
        :param value: 查询值
        :return: 匹配的记录列表
        :raises KeyError: 如果键未定义
        """
        if key not in self.all_keys:
            raise KeyError(f"键 '{key}' 未定义")
        return [record for record in self.data if record.get(key) == value]

    def update(self, query_key: str, query_value: Any, 
               update_key: str, update_value: Any) -> int:
        """
        线性搜索并更新记录
        :return: 更新的记录数量
        """
        if query_key not in self.all_keys:
            raise KeyError(f"查询键 '{query_key}' 未定义")
        if update_key not in self.all_keys:
            raise KeyError(f"更新键 '{update_key}' 未定义")

        updated_count = 0
        for record in self.data:
            if record.get(query_key) == query_value:
                record[update_key] = update_value
                updated_count += 1
        
        # 同步更新属性列表（如果需要）
        if updated_count > 0:
            self._sync_attr_lists()
        
        return updated_count

    def _sync_attr_lists(self):
        """将data中的数据同步到各属性列表（仅在必要时调用）"""
        for key in self.all_keys:
            attr_list = getattr(self, key)
            attr_list.clear()
            attr_list.extend(record[key] for record in self.data)

    def delete(self, key: str, value: Any) -> int:
        """
        线性搜索并物理删除记录
        :return: 删除的记录数量
        """
        if key not in self.all_keys:
            raise KeyError(f"键 '{key}' 未定义")

        # 标记要删除的索引（倒序避免移位问题）
        indexes_to_delete = [
            i for i, record in enumerate(self.data) 
            if record.get(key) == value
        ]
        
        for i in sorted(indexes_to_delete, reverse=True):
            self.data.pop(i)
        
        # 同步属性列表
        self._sync_attr_lists()
        return len(indexes_to_delete)

    def get_values_by_key(self, key: str) -> List[Any]:
        """
        获取指定键的所有唯一值
        :param key: 查询键
        :return: 包含所有唯一值的列表
        """
        if key not in self.all_keys:
            raise KeyError(f"键 '{key}' 未定义")
        return list({record[key] for record in self.data if key in record})


class Memory_v2:
    def __init__(self, all_keys=None):
        """
        初始化存储系统
        :param all_keys: 所有可能的键列表，如 ['g_label', 'name', 'age', 'department']
                         必须包含主键 'g_label'，如果不指定则默认为 ['g_label']
        """
        if all_keys is None:
            all_keys = ['g_label']
        elif 'g_label' not in all_keys:
            raise ValueError("必须包含主键 'g_label'")

        self.data = []  # 存储所有记录
        self.indices = {}  # 多级索引字典 {key: {serialized_value: [index1, index2, ...]}}
        self.all_keys = set(all_keys)  # 所有预定义的键
        self._initialize_indices()

    def _serialize_value(self, value: Any) -> str:
        """将任意值序列化为可哈希的字符串（用于索引键）"""
        if isinstance(value, (list, dict, set)):
            return json.dumps(value, sort_keys=True)  # 保持一致性
        return str(value)

    def _initialize_indices(self):
        """初始化所有键的索引结构"""
        self.indices = {key: {} for key in self.all_keys}

    def clear_all(self):
        """清空所有数据和索引，但保留键结构"""
        self.data = []
        self._initialize_indices()

    def add(self, g_label: Any, **kwargs: Any) -> int:
        """
        添加新记录（自动补全缺失的键为None）
        :param g_label: 主键值（任意类型）
        :param kwargs: 其他键值对（值可以是任意类型）
        :return: 新记录的索引
        :raises KeyError: 如果尝试添加未定义的键
        """
        undefined_keys = set(kwargs.keys()) - self.all_keys
        if undefined_keys:
            raise KeyError(f"尝试添加未定义的键: {undefined_keys}")

        record = {key: None for key in self.all_keys}
        record.update({'g_label': g_label, **kwargs})
        
        index = len(self.data)
        self.data.append(record)
        self._add_to_indices(index, record)
        
        return index

    def _add_to_indices(self, index: int, record: Dict[str, Any]):
        """将记录添加到索引（自动处理值的序列化）"""
        for key, value in record.items():
            serialized_value = self._serialize_value(value)
            if serialized_value not in self.indices[key]:
                self.indices[key][serialized_value] = []
            self.indices[key][serialized_value].append(index)

    def get(self, key: str, value: Any) -> List[Dict]:
        """
        获取指定键值对应的所有记录
        :param key: 查询键
        :param value: 查询值（任意类型）
        :return: 匹配的记录列表
        :raises KeyError: 如果键未定义
        """
        if key not in self.all_keys:
            raise KeyError(f"键 '{key}' 未定义")
        
        serialized_value = self._serialize_value(value)
        return [self.data[idx] for idx in self.indices.get(key, {}).get(serialized_value, [])]

    def update(self, query_key: str, query_value: Any, update_key: str, update_value: Any) -> int:
        """
        更新记录（自动处理值的序列化）
        :param query_key: 查询键
        :param query_value: 查询值（任意类型）
        :param update_key: 要更新的键
        :param update_value: 新的值（任意类型）
        :return: 更新的记录数量
        """
        if query_key not in self.all_keys:
            raise KeyError(f"查询键 '{query_key}' 未定义")
        if update_key not in self.all_keys:
            raise KeyError(f"更新键 '{update_key}' 未定义")

        serialized_query_value = self._serialize_value(query_value)
        matched_indexes = self.indices.get(query_key, {}).get(serialized_query_value, [])
        updated_count = 0

        for idx in matched_indexes:
            record = self.data[idx]
            old_value = record[update_key]
            
            # 更新记录
            record[update_key] = update_value
            
            # 更新索引
            self._remove_from_index(update_key, old_value, idx)
            self._add_to_index(update_key, update_value, idx)
            
            updated_count += 1

        return updated_count

    def _remove_from_index(self, key: str, value: Any, index: int):
        """从索引中移除记录（自动处理值的序列化）"""
        serialized_value = self._serialize_value(value)
        if serialized_value in self.indices[key]:
            self.indices[key][serialized_value].remove(index)
            if not self.indices[key][serialized_value]:
                del self.indices[key][serialized_value]

    def _add_to_index(self, key: str, value: Any, index: int):
        """添加记录到索引（自动处理值的序列化）"""
        serialized_value = self._serialize_value(value)
        if serialized_value not in self.indices[key]:
            self.indices[key][serialized_value] = []
        self.indices[key][serialized_value].append(index)

    def delete(self, key: str, value: Any) -> int:
        """
        删除指定键值对应的记录（自动处理值的序列化）
        :param key: 删除键
        :param value: 删除值（任意类型）
        :return: 删除的记录数量
        """
        if key not in self.all_keys:
            raise KeyError(f"键 '{key}' 未定义")

        serialized_value = self._serialize_value(value)
        matched_indexes = sorted(
            self.indices.get(key, {}).get(serialized_value, []),
            reverse=True
        )
        
        if not matched_indexes:
            return 0

        deleted_count = 0
        for idx in matched_indexes:
            self.data.pop(idx)
            self._rebuild_indices_after_delete(idx)
            deleted_count += 1

        return deleted_count

    def _rebuild_indices_after_delete(self, deleted_index: int):
        """物理删除后重建索引"""
        self._initialize_indices()
        for new_index, record in enumerate(self.data):
            self._add_to_indices(new_index, record)

    def get_values_by_key(self, key: str) -> List[Any]:
        """
        获取指定键的所有可能取值（返回反序列化的原始值）
        :param key: 查询键
        :return: 包含所有唯一值的列表
        """
        if key not in self.all_keys:
            raise KeyError(f"键 '{key}' 未定义")
        
        # 从序列化的索引键还原原始值
        unique_values = []
        seen = set()
        for record in self.data:
            value = record[key]
            if value is not None and value not in seen:
                seen.add(value)
                unique_values.append(value)
        return unique_values

class Memory_v1:
    def __init__(self):
        self._data = {}  # 主存储，以g_label为键
        self._label_to_glabel = {}  # label到g_label的映射
        
    def add_data(self, g_label, label=None, mask=None, cor=None):
        """添加或更新数据"""
        if g_label in self._data:
            # 更新现有数据
            if label is not None:
                # 如果已有label且要修改，先清除旧映射
                old_label = self._data[g_label]['label']
                if old_label in self._label_to_glabel:
                    del self._label_to_glabel[old_label]
                # 建立新映射
                self._label_to_glabel[label] = g_label
                
            self._data[g_label].update({
                'label': label if label is not None else self._data[g_label].get('label'),
                'mask': mask if mask is not None else self._data[g_label].get('mask'),
                'cor': cor if cor is not None else self._data[g_label].get('cor')
            })
        else:
            # 添加新数据
            self._data[g_label] = {
                'label': label,
                'mask': mask,
                'cor': cor
            }
            if label is not None:
                self._label_to_glabel[label] = g_label
    
    def clear_all(self):
        """
        清除所有存储的数据
        :return: 无返回值
        """
        self._data.clear()  # 清空主存储
        self._label_to_glabel.clear()  # 清空label映射
    
    def update_label(self, old_label, new_label):
        """
        将指定的old_label修改为new_label
        :param old_label: 要修改的现有label
        :param new_label: 新的label值
        :return: 是否修改成功 (False表示old_label不存在或new_label已存在)
        """
        if old_label not in self._label_to_glabel:
            return False
            
        if new_label in self._label_to_glabel and new_label != old_label:
            return False
            
        g_label = self._label_to_glabel[old_label]
        
        del self._label_to_glabel[old_label]
        self._label_to_glabel[new_label] = g_label
        
        self._data[g_label]['label'] = new_label
        
        return True
    
    def delete_by_g_label(self, g_label):
        """
        通过g_label删除数据
        :param g_label: 要删除的数据的g_label
        :return: 被删除的数据项，如果不存在则返回None
        """
        if g_label not in self._data:
            return None
            
        deleted_item = self._data[g_label]
        
        if deleted_item['label'] is not None and deleted_item['label'] in self._label_to_glabel:
            del self._label_to_glabel[deleted_item['label']]
        
        del self._data[g_label]
        
        return deleted_item
    
    def delete_by_label(self, label):
        """
        通过label删除数据
        :param label: 要删除的数据的label
        :return: 被删除的数据项，如果不存在则返回None
        """
        if label not in self._label_to_glabel:
            return None
            
        g_label = self._label_to_glabel[label]
        return self.delete_by_g_label(g_label)
    
    def get_all_g_labels(self) -> List[str]:
        """获取所有g_label"""
        return list(self._data.keys())
    
    def get_all_labels(self) -> List[str]:
        """获取所有label（不包括None的label）"""
        return [item['label'] for item in self._data.values() if item['label'] is not None]
    
    def get_by_g_label(self, g_label):
        """通过g_label获取完整数据"""
        return self._data.get(g_label)
    
    def get_by_label(self, label):
        """通过label获取完整数据"""
        g_label = self._label_to_glabel.get(label)
        if g_label is not None:
            return self._data.get(g_label)
        return None
    
    def get_mask(self, identifier):
        """通过g_label或label获取mask"""
        data = self._get_data(identifier)
        return data.get('mask') if data else None
    
    def get_cor(self, identifier):
        """通过g_label或label获取cor"""
        data = self._get_data(identifier)
        return data.get('cor') if data else None
    
    def _get_data(self, identifier):
        """内部方法：通过g_label或label获取数据"""
        if identifier in self._data:
            return self._data[identifier]
        elif identifier in self._label_to_glabel:
            g_label = self._label_to_glabel[identifier]
            return self._data.get(g_label)
        return None
    
    # def __repr__(self):
    #     return f"Memory({len(self._data)} items)"

if __name__ == '__main__':
    # 初始化时定义所有可能的键
    mem = Memory(all_keys=['g_label', 'name', 'age', 'department', 'salary'])

    # 添加记录（可以只提供部分字段）
    mem.add(g_label="emp001", name="Alice")
    mem.add(g_label="emp002", age=30)
    mem.add(g_label="emp003", department="HR", salary=5000)

    # 通过查询更新记录
    # 将所有name为None的记录设置name为"Unknown"
    mem.update_by_query(
        query_key="name", 
        query_value=None, 
        update_key="name", 
        update_value="Unknown"
    )

    # 将年龄为30的员工部门设为"IT"
    mem.update_by_query(
        query_key="age",
        query_value=30,
        update_key="department",
        update_value="IT"
    )

    # 查询验证
    print(mem.get("department", "IT"))  # 查找IT部门的员工
    print(mem.get_values_by_key("name"))  # 查看所有姓名取值