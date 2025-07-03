import json
import os

class ManualManager:
    def __init__(self, file_path, save_path=None):
        self.file_path = file_path
        if save_path:
            self.save_path = save_path
        else:
            self.save_path = file_path
        self.prompts = self.load_prompts()

    def load_prompts(self):
        """读取提示词文件"""
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {}

    def save_prompts(self):
        """将修改后的提示词保存到文件"""
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(self.prompts, f, ensure_ascii=False, indent=4)

    def get_prompt(self, key):
        """获取指定 key 的提示词"""
        keys = key.split('.')
        prompt = self.prompts
        for k in keys:
            prompt = prompt.get(k, None)
            if prompt is None:
                return "None"
        return prompt

    def set_prompt(self, key, value):
        """修改指定 key 的提示词"""
        keys = key.split('.')
        prompt = self.prompts
        for k in keys[:-1]:
            prompt = prompt.setdefault(k, {})
        prompt[keys[-1]] = value
        self.save_prompts()

    def add_prompt(self, key, value):
        """添加新提示词"""
        self.set_prompt(key, value)

    def delete_prompt(self, key):
        """删除指定 key 的提示词"""
        keys = key.split('.')
        prompt = self.prompts
        for k in keys[:-1]:
            prompt = prompt.get(k, {})
            if prompt is None:
                return False
        if keys[-1] in prompt:
            del prompt[keys[-1]]
            self.save_prompts()
            return True
        return False

    def convert_to_string(self, prompt_dict=None, indent=0):
        """将所有提示词转化为层级字符串，层级关系使用回车和制表符表示"""
        if prompt_dict is None:
            prompt_dict = self.prompts

        result = ""
        for key, value in prompt_dict.items():
            result += '\t' * indent + key + ": "  # 添加缩进和 key
            if isinstance(value, dict):
                result += "\n"  # 如果是字典，则换行并递归
                result += self.convert_to_string(value, indent + 1)
            elif value is None:
                result += "空值\n"  # 如果 value 是 None，标记为 "空值"
            elif value == "":
                result += "空字符串\n"  # 如果 value 是空字符串，标记为 "空字符串"
            else:
                result += value + "\n"  # 否则输出实际的 value
        return result

    def get_subkeys(self, key):
        """获取指定 key 下的所有子 key"""
        keys = key.split('.')
        prompt = self.prompts
        for k in keys:
            prompt = prompt.get(k, None)
            if prompt is None:
                return []

        if not isinstance(prompt, dict):
            return []

        def _get_subkeys_recursive(d):
            subkeys = []
            for k, v in d.items():
                subkeys.append(k)
                if isinstance(v, dict):
                    subkeys.extend(_get_subkeys_recursive(v))
            return subkeys

        return _get_subkeys_recursive(prompt)

# 示例使用
if __name__ == "__main__":
    manager = PromptManager('prompts.json')

    # 读取提示词
    print(manager.get_prompt("greeting.morning"))

    # 修改提示词
    manager.set_prompt("greeting.morning", "早上好！今天怎么样？")

    # 添加新提示词
    manager.add_prompt("farewell", "再见！")
    manager.add_prompt("empty_field", "")

    # 删除提示词
    manager.delete_prompt("task.create_report")

    # 打印修改后的提示词
    print(manager.prompts)

    # 获取层级化的提示词字符串
    print(manager.convert_to_string())
