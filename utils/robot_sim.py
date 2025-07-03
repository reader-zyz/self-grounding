from utils.annotate import *
from utils.myapi import Conversation
from openai import OpenAI
from PIL import Image
from utils.manual_manager import ManualManager
from langsam.lang_sam import LangSAM
import numpy as np
from typing import Optional, Tuple, Union, List, Any, Dict
import cv2
import logging
import configparser
import math
from scipy.spatial.transform import Rotation as R
import json


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
        self.scene_mem = Memory(['g_label', 'label', 'mask', 'cor', 'size', 'dis'])
        
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

    def plan_only(self, label, img, mode=[], extra_txt=[]): 
        if 'size' in mode:
            size = ''
            for label in self.scene_mem.get_values_by_key("label"):
                size = ''.join([size, f'{label} size: {self.scene_mem.get("label", label)[0]["size"]:.3f} '])
            extra_txt.append(size)
        if 'dis' in mode:
            dis = ''
            for label in self.scene_mem.get_values_by_key("label"):
                size = ''.join([size, f'{label} distance to robot: {self.scene_mem.get("label", label)[0]["dis"]:.3f} '])
            extra_txt.append(dis)
        # 进行决策
        feasibility = self.llm_infer('feasibility.txt', [img], [self.manual.convert_to_string(), label] + extra_txt, False, True, mode=mode)
        consequence = self.llm_infer('consequence.txt', [img], [self.manual.convert_to_string(), label] + extra_txt, False, True, mode=mode)
        security = self.llm_infer('security.txt', [img], [self.manual.convert_to_string(), label] + extra_txt, False, True, mode=mode)
        res = self.llm_infer('synthesize.txt', [img], [self.manual.convert_to_string(), label, feasibility, consequence, security] + extra_txt, False, True, mode=mode)

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
        res = self.llm_infer('synonym.txt', [], [label_1, ','.join(self.manual.get_subkeys(f'{tool}.affordances'))], False, True)
        res = str2list(res)
        for i, res_ in enumerate(res):
            item = str2list(label_1)[i]
            # 如果说明书中不存在近义词
            if res_ == '0':
                self.manual.add_prompt(f'{tool}.affordances.{item}', ' ')
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
                self.manual.add_prompt(f'{tool}.affordances.{item}', f'{tool}可以清理{item}')
            # 物体依然存在，清洁无效
            else:
                self.manual.add_prompt(f'{tool}.affordances.{item}', f'{tool}不能清理{item}')  

        res = self.llm_infer('synonym.txt', [], [label_2, label_1], False, True)
        res = str2list(res)
        for i, res_ in enumerate(res):
            # 新出现的物体，说明二次污染
            if res_ == '0':
                notice = self.llm_infer('secondary_contamination.txt', [before_img, after_img], [label_1, label_2, label_2, tool], False, True)
                item, notice = sort_label(notice)
                current_manual = self.manual.get_prompt(f'{tool}.affordances.{item}')
                # 如果本身存在注意事项，需要进行注意事项的融合
                notice = self.llm_infer('merge_notice.txt', [], [current_manual, notice], False, True)
                self.manual.add_prompt(f'{tool}.affordances.{item}', notice) 
            # 物体以前存在，清洁无效
            else:
                pass

    def align_navigation_point(self, valid_nav_points: str):
        # 设置重试次数
        retry = 0
        # 获取当前的导航点用于备份和检索
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
                    self.nav_mem.delete('label', robot_nav_point[i])
                # 场景中有对应的导航点，将这个导航点保存
                else:
                    # 更新nav mem
                    self.nav_mem.update('label', robot_nav_point[i], 'label', res_)
            break
        logging.info(f'aligned navigation points, before: {robot_nav_point}, after: {self.nav_mem.get_values_by_key("label")}')

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
