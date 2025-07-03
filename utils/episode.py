import os
import random
# 建立一个管理整个流程的类，该类初步设定两种运行方式
# 数据集模式下，不需要实体机器人，直接通过与LLM的交互完成任务
# 机器人模式下，需要设置spawn的坐标，给定每个item的坐标，机器人每观察一下，就将图像更新到该类中，与LLM的交互也通过该类实现
class Episode:
    def __init__(self, directory):
        self.directory = directory
        self.scenes = {}
        self.metarule = None
        self.spawn_images = []
        self.state = 'spawn'  # 初始状态为'spawn'
        
        # 支持的图像文件扩展名
        self.image_extensions = ('.jpg', '.jpeg', '.png')
        
        # 读取metarule.txt
        metarule_path = os.path.join(directory, 'metarule.txt')
        if os.path.exists(metarule_path):
            with open(metarule_path, 'r') as f:
                self.metarule = [line.strip().split(',') for line in f.readlines()]
        
        # 读取spawn images
        for filename in os.listdir(directory):
            if filename.startswith('spawn_') and filename.lower().endswith(self.image_extensions):
                # 存储相对路径
                self.spawn_images.append(os.path.join(directory, filename))
        
        # 读取每个scene的信息
        for scene_name in os.listdir(directory):
            scene_path = os.path.join(directory, scene_name)
            if os.path.isdir(scene_path):
                self.scenes[scene_name] = {
                    'objects': [],
                    'images': [],
                    'id': set(),
                    'rule': []
                }
                
                # 读取rule.txt
                rule_path = os.path.join(scene_path, 'rule.txt')
                if os.path.exists(rule_path):
                    with open(rule_path, 'r') as f:
                        # 将每条规则转换为列表形式
                        self.scenes[scene_name]['rule'] = [line.strip().split(',') for line in f.readlines()]
                
                # 读取scene下的所有图片
                for filename in os.listdir(scene_path):
                    if filename.lower().endswith(self.image_extensions):
                        # 存储相对路径
                        self.scenes[scene_name]['images'].append(os.path.join(scene_path, filename))
                        # 提取图片名称中的物体ID
                        name = os.path.splitext(filename)[0]  # 去掉扩展名
                        if name != 'None':
                            objects = name.split('_')
                            self.scenes[scene_name]['id'].update(objects)
                
                # 将id集合转换为列表
                self.scenes[scene_name]['id'] = list(self.scenes[scene_name]['id'])
                
                # 从a_b.txt中提取初始状态的对象
                initial_state_file = None
                for filename in os.listdir(scene_path):
                    if filename.lower().endswith('.txt') and filename.lower() != 'rule.txt':
                        initial_state_file = filename
                        break
                
                if initial_state_file:
                    # 提取文件名中的对象（去掉.txt后缀）
                    initial_objects = os.path.splitext(initial_state_file)[0].split('_')
                    self.scenes[scene_name]['objects'] = initial_objects
                else:
                    print(f"No initial state file found for scene '{scene_name}'.")

    def generate_rule(self):
        if not self.metarule:
            print("No metarule found.")
            return
        
        for scene_name, scene_data in self.scenes.items():
            scene_id = scene_data['id']
            applicable_rules = []
            
            # 遍历每条metarule
            for rule in self.metarule:
                # 解析metarule中的目标物体
                target_object = rule[0]  # 第一个元素是目标物体
                
                # 检查目标物体是否在scene的id中
                if target_object in scene_id:
                    applicable_rules.append(rule)
            
            # 如果有适用的规则，写入rule.txt并更新scene['rule']
            if applicable_rules:
                rule_path = os.path.join(self.directory, scene_name, 'rule.txt')
                # 将规则列表转换为逗号分隔的字符串
                rule_strings = [','.join(rule) for rule in applicable_rules]
                with open(rule_path, 'w') as f:
                    f.write('\n'.join(rule_strings))
                scene_data['rule'] = applicable_rules
                print(f"Updated rules for scene '{scene_name}'.")
            else:
                print(f"No applicable rules for scene '{scene_name}'.")

    def get_image(self):
        """
        根据当前state找到对应的scene，再根据scene['objects']确定状态，
        返回与objects一致的image（考虑任意顺序），如果没有匹配则返回同目录下的None.jpg。
        """
        if self.state == 'spawn':
            print("Current state is 'spawn'. No scene image to return.")
            return None
        
        # 获取当前scene
        if self.state not in self.scenes:
            print(f"Scene '{self.state}' not found.")
            return None
        
        scene_data = self.scenes[self.state]
        objects = scene_data['objects']
        images = scene_data['images']
        
        # 将objects排序并生成标准化的字符串
        normalized_objects = '_'.join(sorted(objects))
        none_image_path = None
        
        # 遍历images，找到与objects一致的图片
        for image_path in images:
            # 提取图片名称中的物体ID
            filename = os.path.basename(image_path)
            name = filename.split('.')[0]
            if name != 'None':
                # 将图片名称中的物体排序并生成标准化的字符串
                normalized_image_objects = '_'.join(sorted(name.split('_')))
                if normalized_image_objects == normalized_objects:
                    return image_path
            else:
                # 保存None.jpg的路径以备后用
                none_image_path = image_path
        
        # 如果找到了None.jpg就返回它，否则返回None
        return none_image_path


    def action(self, actions, objects):
        """
        Accepts lists of actions and objects, and updates state according to rules.
        Processes actions and objects in sequence.
        """
        if self.state == 'spawn':
            print("Current state is 'spawn'. No action can be performed.")
            return None
        
        # Check if the number of actions matches the number of objects
        if len(actions) != len(objects):
            print("Error: Number of actions must match number of objects.")
            return None
        
        # Get current scene
        if self.state not in self.scenes:
            print(f"Scene '{self.state}' not found.")
            return None
        
        scene_data = self.scenes[self.state]
        current_objects = scene_data['objects']
        rules = scene_data['rule']
        
        # Process each action-object pair sequentially
        for action, object_ in zip(actions, objects):
            # Check if object exists in current scene
            if object_ not in current_objects:
                print(f"Object '{object_}' not found in current scene. Skipping this action.")
                continue
            
            # Find matching rule
            rule_matched = False
            for rule in rules:
                if len(rule) >= 2 and rule[0] == object_ and rule[1] == action:
                    rule_matched = True
                    # Update state
                    if object_ in current_objects:
                        current_objects.remove(object_)
                    # Parse probability parts
                    # Generate random number
                    rand_num = random.randint(1, 100)
                    for prob_effect in rule[2:]:
                        prob_range, effect = prob_effect.split(':')
                        start, end = map(int, prob_range.split('-'))
                        if start <= rand_num <= end:
                            if effect != 'None':
                                current_objects.append(effect)
                            break
                    break
            
            if not rule_matched:
                print(f"No matching rule found for action '{action}' on object '{object_}'")
        
        # # Check if objects are empty after processing all actions
        # if len(current_objects) == 0:
        #     print("Scene objects are empty. Placeholder for additional logic.")
        # else:
        #     # Return the final image after processing all actions
        #     return self.get_image()
        return self.get_image()

    def action_old(self, action, object_):
        """
        接受action和object作为输入，根据规则更新状态。
        """
        if self.state == 'spawn':
            print("Current state is 'spawn'. No action can be performed.")
            return None
        
        # 获取当前scene
        if self.state not in self.scenes:
            print(f"Scene '{self.state}' not found.")
            return None
        
        scene_data = self.scenes[self.state]
        objects = scene_data['objects']
        rules = scene_data['rule']
        
        # 检查输入的object是否在当前scene的objects中
        if object_ not in objects:
            print(f"Object '{object_}' not found in current scene.")
            return None
        
        # 遍历规则，查找匹配的规则
        for rule in rules:
            if len(rule) >= 2 and rule[0] == object_ and rule[1] == action:
                # 解析概率部分
                for prob_effect in rule[2:]:
                    prob_range, effect = prob_effect.split(':')
                    start, end = map(int, prob_range.split('-'))
                    # 生成随机数
                    rand_num = random.randint(1, 100)
                    if start <= rand_num <= end:
                        # 执行状态更新
                        if object_ in objects:
                            objects.remove(object_)
                        if effect != 'None':
                            objects.append(effect)
                        break
                break
        
        # 检查objects的长度
        if len(objects) == 0:
            # 留出一个代码块的位置
            print("Scene objects are empty. Placeholder for additional logic.")
        else:
            # 调用get_image并返回结果
            return self.get_image()


class Episode_old:
    def __init__(self):
        self.item = {} # {'desk':{'coordinate':[x, y, theta], 'observation':['desk_0.jpg']}} 每张光学图像对应的点云信息在point_cloud的同名文件中

    # 添加新的item
    def add_item(self, item):
        if item in self.item.keys():
            raise ValueError(f'{item} already exists in items')
        self.item[item] = {'coordinate': [], 'observation': []} # item代表导航点的名称、cor代表场景中的坐标、obs代表详细观察得到的图像

    # 读取episode
    def read_episode(self, episode_dir):
        for file_name in os.listdir(episode_dir):
            if file_name.endswith('.jpg'):
                file_path = os.path.join(episode_dir, file_name)
                # 在字典中添加图片的路径
                item_name = ''.join(file_name.split('_')[:-1])
                if item_name not in self.item.keys():
                    self.add_item(item_name)
                self.item[item_name]['observation'].append(file_path)

            # 读取坐标文件
            elif file_name.endswith('.txt'):
                file_path = os.path.join(episode_dir, file_name)
                # 读取文件并处理所有行
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    
                    for line in lines:
                        parts = line.split(',')
                        # 去除空格并转换为数字
                        parts = [float(part.strip()) if part.strip().replace('.', '', 1).isdigit() else part.strip() for part in parts]
                    if parts[0] not in self.item.keys():
                        self.add_item(parts[0])
                    self.item[parts[0]]['coordinate'].append(parts[1:])
