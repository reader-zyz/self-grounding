# 创建一个message类方便问答内容的增删改查
from openai import OpenAI
import os
import re
# 常用函数
import base64
import io
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import matplotlib.pyplot as plt
import platform
import os

'''
### message template: ###
messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "这张图片里有什么?请详细描述。"},
                    {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]
'''
'''
### conversation template: ###
<s>user,,image_url,,<image>
<s>user,,text,,这张图片里都有什么东西，
你认为哪些区域需要清理？
<s><question><temperature=1.0>
'''

class Conversation:
    def __init__(self, client, model):
        self.message = []
        self.client = client
        self.model = model
        self.temperature_record = []

    # 增加内容，以role为单位,传入的图片需要是PIL 
    def add_message(self, role, type_list, detail_content_list, merge_flag=True):
        '''
            role: str
        '''
        # 输入合法检测
        if len(type_list) != len(detail_content_list):
            raise ValueError('type num and detail content num imcompatible when adding message')
        if role not in ['system', 'user', 'assistant']:
            raise ValueError('role illegal when adding message')

        for i in range(len(type_list)):
            if isinstance(detail_content_list[i], str):
                self.add_text(detail_content_list[i], role, merge_flag)
            elif isinstance(detail_content_list[i], Image.Image):
                self.add_img(detail_content_list[i], role, merge_flag)
            else:
                raise ValueError('content type is neither str nor Image')
    
    def add_text(self, text, role='user', merge_flag=True):
        # 需要合并role
        if merge_flag:
            # 当前message不为空
            if len(self.message):
                if self.message[-1]['role'] == role:
                    # 与上一条合并
                    self.message[-1]['content'].append({'type': 'text', 'text': text})
                    return True
        self.message.append({'role': role, 'content': [{'type': 'text', 'text': text}]})

    # 增加图片内容，输入为PIL的img
    def add_img(self, img, role='user', merge_flag=True):
        image_base64 = pil_to_base64(img)
        # 需要合并role
        if merge_flag:
            # 当前message不为空
            if len(self.message):
                if self.message[-1]['role'] == role:
                    # 与上一条合并
                    self.message[-1]['content'].append({'type': 'image_url', 'image_url': {"url":  f"data:image/jpeg;base64,{image_base64}"}})
                    return True
        self.message.append({'role': role, 'content': [{'type': 'image_url', 'image_url': {"url":  f"data:image/jpeg;base64,{image_base64}"}}]})

    # 将message做为输入进行问答，并更新message
    def question(self, temperature=1.0):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.message,
            temperature=temperature # 加入温度调节
            )
        # response.choices[0].message.content可以直接访问回答内容，response.choices[0].message.role可以直接访问'assistant'
        self.add_message(response.choices[0].message.role,
                        ['text' if type(response.choices[0].message.content) in [str] else None],
                        [response.choices[0].message.content])
        self.temperature_record.append(temperature)

    # 获取最新的回答
    def get_lastest_ans(self, print_flag=False):
        for message in self.message[::-1]:
            if message['role'] == 'assistant':
                if print_flag:
                    print(message['content'][0]['text'])
                return message['content'][0]['text']
        return None

    # 展示所有的message内容,以role:content的格式
    def get_message(self, output_path, mode='image', ):
        res_text = []
        res_image = []
        temperature_point = 0
        for message in self.message[:]: # message->dict
            for detail in message['content']: # detail->dict{'type':str, 'str':detail_content}
                if detail['type'] == 'text':
                    if message['role'] == 'assistant':
                        res_text.append(''.join([message['role'], f' temperature={self.temperature_record[temperature_point]}'': ', detail['text']]))
                        temperature_point += 1
                    else:
                        res_text.append(''.join([message['role'], ': ', detail['text']]))
                elif detail['type'] == 'image_url':
                    
                    image_base64 = detail['image_url']['url'].split(',')[-1]
                    image = base64.b64decode(image_base64)
                    image = Image.open(io.BytesIO(image))
                    res_text.append(''.join([message['role'], ': ']))
                    res_text.append('<image>') # 图片占位符
                    res_image.append(image)

        # 如果问答中有图片，则只能输出图片
        if len(res_image):
            mode = 'image'
        # 内容中只有文字信息且不打印图片
        if mode == 'text':
            for line in res_text:
                print(line)
        # 问答内容中有图片或者要求打印图片
        if mode == 'image':
            # 获取当前时间用于命名
            now = datetime.now()
            output_path = ''.join([output_path, '/', now.strftime("%m%d%H%M%S"), '.jpg'])
            visualize(res_text, res_image, output_path, font_size=24, image_max_width=700, image_max_height=400, max_chars_per_line=30)

    def read_txt(self, txt_path, image_list=[], text_list=[], mode=[]):
        """
        处理TXT文档并根据规则替换占位符内容，返回处理结果
        :param txt_path: TXT文档路径
        :param image_list: PIL图像列表
        :param text_list: 文本内容列表
        :param mode: 当前模式列表，决定哪些<mode:XXX>标记的内容会被处理
        :return: None
        mode说明：space：空间描述；size：尺寸、距离与工作范围；state：当前电量水箱等；locate：输出的决策与SOM标注的数字绑定；cut：当图像问答结束后删掉图像节省token
        """
        with open(txt_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 按照<s>分割
        segments = content.split('<s>')
        image_index = 0
        text_index = 0

        for segment in segments:
            segment = segment.strip()
            if not segment:  # 跳过空段
                continue
            
            # 初始化当前段的模式检查状态
            current_mode_check = True
            required_modes = []
            
            # 检查是否有<mode:XXX>标记（必须在<s>后立即出现）
            mode_match = re.match(r'<mode:([a-zA-Z_,]+)>(.*)', segment, re.DOTALL)
            if mode_match:
                # 提取模式要求和剩余内容
                required_modes = mode_match.group(1).split(',')
                segment = mode_match.group(2).strip()
                
                # 检查当前模式是否包含任一要求的模式
                if not any(req_mode in mode for req_mode in required_modes):
                    current_mode_check = False  # 不处理当前段
            
            # 如果模式检查不通过，跳过当前段
            if not current_mode_check:
                continue
            
            if '<question>' in segment:
                # 检查是否有<cut>标记
                cut_flag = '<cut>' in segment
                # 如果存在<cut>标记，检查是否在允许的模式中
                if cut_flag and 'cut' not in mode:
                    cut_flag = False
                
                # 使用正则表达式提取温度值
                match = re.search(r'<temperature=([0-9\.]+)>', segment)
                # 如果找到了匹配的结果，将其转换为浮点数
                if match:
                    temperature = float(match.group(1))
                else:
                    temperature = 0.6
                
                self.question(temperature)
                
                # 如果设置了cut_flag且模式允许，则删除历史对话中的图片记录
                if cut_flag:
                    self._cut_images()
            else:
                # 按照',,'分割
                parts = segment.split(',,')
                if len(parts) != 3:
                    continue  # 如果分割结果不是三个部分，则跳过
                role, type_, detail_content = parts[0].strip(), parts[1].strip(), parts[2].strip()
                
                if role not in ['system', 'user', 'assistant']:
                    raise ValueError('wrong role content when reading txt')

                if type_ == 'text':
                    placeholders = re.findall(r"<text>", detail_content)
                    for _ in range(len(placeholders)):
                        detail_content = detail_content.replace('<text>', text_list[text_index], 1)
                        text_index += 1
                    self.add_text(detail_content, role=role)
                elif type_ == 'image_url':
                    self.add_img(image_list[image_index], role=role)
                    image_index += 1
                else:
                    raise ValueError('wrong type content when reading txt')

    def _cut_images(self):
        """删除历史对话中的图片记录，保留文字内容"""
        new_messages = []
        
        for message in self.message[:-1]:  # 保留最后一条(assistant的回答)
            new_content = []
            for content_item in message['content']:
                if content_item['type'] == 'text':
                    new_content.append(content_item)
            
            if new_content:  # 只保留有文字内容的消息
                new_message = {
                    'role': message['role'],
                    'content': new_content
                }
                new_messages.append(new_message)
        
        # 保留最后一条assistant的回答
        if self.message and self.message[-1]['role'] == 'assistant':
            new_messages.append(self.message[-1])
        
        self.message = new_messages

    # 读取txt文档的内容并进行问答
    def read_txt_old(self, txt_path, image_list=[], text_list=[]):
        """
        处理TXT文档并根据规则替换占位符内容，返回处理结果
        :param txt_path: TXT文档路径
        :param image_list: PIL图像列表
        :return: 处理后的消息列表
        """
        with open(txt_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 按照<s>分割
        segments = content.split('<s>')
        messages = []
        image_index = 0
        text_index = 0

        for segment in segments:
            segment = segment.strip()
            if not segment:  # 跳过空段
                continue
            
            if '<question>' in segment:
                # 使用正则表达式提取温度值
                match = re.search(r'<temperature=([0-9\.]+)>', segment)
                # 如果找到了匹配的结果，将其转换为浮点数
                if match:
                    temperature = float(match.group(1))
                else:
                    temperature = 0.6
                self.question(temperature)
            else:
                # 按照',,'分割
                parts = segment.split(',,')
                if len(parts) != 3:
                    continue  # 如果分割结果不是三个部分，则跳过
                role, type_, detail_content = parts[0].strip(), parts[1].strip(), parts[2].strip()
                
                if role not in ['system', 'user', 'assistant']:
                    raise ValueError('wrong role content when reading txt')

                if type_ == 'text':
                    placeholders = re.findall(r"<text>", detail_content)
                    for _ in range(len(placeholders)):
                        detail_content = detail_content.replace('<text>', text_list[text_index], 1)
                        text_index += 1
                    self.add_text(detail_content, role=role)
                elif type_ == 'image_url':
                    self.add_img(image_list[image_index], role=role)
                    image_index += 1
                else:
                    raise ValueError('wrong type content when reading txt')

    # 清理message
    def clear_message(self):
        self.message = []
        self.temperature_record = []

# 将PIL编码成base64
def pil_to_base64(image, format='PNG'):
    """
    将 PIL.Image 转换为 Base64 字符串。
    
    :param image: PIL.Image 对象
    :param format: 图像格式，默认为 'PNG'
    :return: Base64 字符串
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)  # 将图像保存到内存中
    buffer.seek(0)
    img_bytes = buffer.read()  # 读取图像的字节数据
    base64_str = base64.b64encode(img_bytes).decode('utf-8')  # 编码为 Base64 字符串
    return base64_str

def get_default_chinese_font(font_size):
    """根据操作系统返回合适的中文字体"""
    system = platform.system()
    
    # Windows 默认使用 "等线" (Deng.ttf)
    if system == "Windows":
        font_path = r"C:/Windows/Fonts/Deng.ttf"
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, font_size)
    
    # Linux 默认使用 "文泉驿微米黑" 或 "Noto Sans CJK"
    elif system == "Linux":
        # 尝试常见的 Linux 中文字体路径
        linux_fonts = [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttf",  # 文泉驿微米黑
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Noto Sans CJK
            "/usr/share/fonts/truetype/arphic/uming.ttc",  # AR PL UMing (Linux 常见中文字体)
        ]
        for font_path in linux_fonts:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, font_size)
    
    # macOS 默认使用 "PingFang.ttc" 或 "STHeiti.ttf"
    elif system == "Darwin":
        mac_fonts = [
            "/System/Library/Fonts/PingFang.ttc",  # 苹方
            "/System/Library/Fonts/STHeiti.ttf",   # 华文黑体
        ]
        for font_path in mac_fonts:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, font_size)
    
    # 如果找不到任何中文字体，回退到默认字体
    return ImageFont.load_default()

def visualize(text_lines, image_list, output_path='output_image.png', font_size=20, image_max_width=700, image_max_height=300, max_chars_per_line=50):
    # 设置基本的图像宽高和背景颜色
    width = 800  # 图像的固定宽度
    height = 1500  # 初始的图像高度
    background_color = (255, 255, 255)  # 白色背景
    text_color = (0, 0, 0)  # 黑色文本
    
    # 获取适合当前系统的中文字体
    font = get_default_chinese_font(font_size)
    
    # 创建空白图像
    image = Image.new('RGB', (width, height), color=background_color)
    draw = ImageDraw.Draw(image)
    
    # 起始绘制位置
    y_offset = 20
    
    # 计算图像所需的高度，文本和图片都需要占据空间
    lines_height = 0
    images = []  # 用来存储图像对象
    
    # 处理文本和插入图像
    for line in text_lines:
        if '<image>' in line:
            # 插入图片的占位符
            if image_list:
                img = image_list.pop(0)  # 获取图片
                img.thumbnail((image_max_width, image_max_height))
                images.append(img)
                lines_height += img.height + 20  # 更新总高度
        else:
            # 使用 textbbox 获取文本的边界框，并计算文本的高度
            wrapped_lines = wrap_text(line, font, max_chars_per_line, width - 40)
            for wrapped_line in wrapped_lines:
                bbox = draw.textbbox((0, 0), wrapped_line, font=font)
                text_height = bbox[3] - bbox[1]  # 计算文本高度
                lines_height += text_height + 10  # 每行文本的高度 + 间隔
    
    # 动态调整图像的高度
    height = lines_height + 100
    image = Image.new('RGB', (width, height), color=background_color)
    draw = ImageDraw.Draw(image)

    # 再次计算y_offset和重新绘制文本和图像
    y_offset = 20
    image_idx = 0

    # 遍历文本和插入图像
    for line in text_lines:
        if '<image>' in line:
            # 插入图片
            if image_idx < len(images):
                img = images[image_idx]
                image.paste(img, (20, y_offset))
                y_offset += img.height + 20
                image_idx += 1
        else:
            # 自动换行文本
            wrapped_lines = wrap_text(line, font, max_chars_per_line, width - 40)
            for wrapped_line in wrapped_lines:
                draw.text((20, y_offset), wrapped_line, fill=text_color, font=font)
                bbox = draw.textbbox((0, 0), wrapped_line, font=font)
                text_height = bbox[3] - bbox[1]
                y_offset += text_height + 10

        # 防止文本超出图像边界
        if y_offset > height - 100:
            break

    # 保存输出的图像
    image.save(output_path)

def wrap_text(text, font, max_chars_per_line, max_width):
    """ 自动换行函数：根据最大字符数和最大宽度拆分文本，支持中文和英文 """
    lines = []
    current_line = ''
    
    # 遍历文本中的每个字符
    for char in text:
        # 将当前行与新字符组合
        test_line = current_line + char
        
        # 获取文本的宽度，判断是否超过最大宽度
        bbox = ImageDraw.Draw(Image.new('RGB', (max_width, 1))).textbbox((0, 0), test_line, font=font)
        text_width = bbox[2] - bbox[0]  # 计算文本的宽度
        
        if text_width <= max_width:
            # 如果未超过最大宽度，则添加字符到当前行
            current_line = test_line
        else:
            # 如果超过最大宽度，将当前行添加到lines，并开始新的一行
            lines.append(current_line)
            current_line = char
    
    if current_line:
        lines.append(current_line)  # 将最后一行添加到lines
    
    return lines


if __name__ == '__main__':

    api_key = "sk-zMfHOFfyelncr2o1onIXnKOJrvp0PLlzfZW0vmFVkjYwvjQg"
    api_base = "https://sg.uiuiapi.com/v1"
    client = OpenAI(api_key=api_key, base_url=api_base)

    # 进行清洁场景的简单问答
    topic = Conversation(client, "gpt-4o-mini")
    img = Image.open('./tidy_scene_1.jpg')
    # topic.read_txt('./conversation/test.txt', [img], ['bill'])
    # print(topic.message)
    # topic.get_lastest_ans(True)
    topic.add_text('hello')
    topic.add_img(img)
    topic.question()
    topic.get_lastest_ans(True)

