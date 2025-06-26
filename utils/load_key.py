# utils/load_key.py

import json
import os
import getpass # 用于隐藏用户在命令行输入的密码或API key

def load_key(key_name: str) -> str:
    """
    从 keys.json 文件中加载指定的 API 密钥。
    如果文件不存在或密钥不存在，则提示用户输入，并保存到文件中。

    Args:
        key_name (str): 要加载的 API 密钥的名称（例如 "OPENAI_API_KEY"）。

    Returns:
        str: 对应的 API 密钥值。
    """
    file_name = "keys.json"
    keys_data = {}

    # 检查 keys.json 文件是否存在
    if os.path.exists(file_name):
        try:
            with open(file_name, "r", encoding="utf-8") as file:
                keys_data = json.load(file)
        except json.JSONDecodeError:
            print(f"Warning: {file_name} is not a valid JSON file. It will be overwritten if keys are updated.")
            keys_data = {} # 如果JSON解析失败，则清空数据，重新开始

    # 检查所需的 key 是否存在且有值
    if key_name in keys_data and keys_data[key_name]:
        print(f"Successfully loaded {key_name} from {file_name}.")
        return keys_data[key_name]
    else:
        # 提示用户输入密钥
        key_value = getpass.getpass(f"'{key_name}' not found in {file_name} or is empty. Please enter your {key_name}: ").strip()
        
        if not key_value:
            raise ValueError(f"'{key_name}' cannot be empty. Please provide a valid key.")

        # 将新的密钥保存到字典中
        keys_data[key_name] = key_value

        # 将更新后的字典写入 keys.json 文件
        try:
            with open(file_name, "w", encoding="utf-8") as file:
                json.dump(keys_data, file, indent=4) # indent=4 使JSON文件格式化，易于阅读
            print(f"'{key_name}' saved to {file_name}.")
        except IOError as e:
            print(f"Error saving {key_name} to {file_name}: {e}")
            # 如果保存失败，仍返回当前输入的key
            
        return key_value

# 示例使用 (如果你直接运行这个文件，可以取消注释下面部分进行测试)
if __name__ == "__main__":
    # Test loading an OpenAI API key
    try:
        openai_key = load_key("OPENAI_API_KEY")
        print(f"Loaded OpenAI API Key (first 5 chars): {openai_key[:5]}...")
        
        # Test loading a different key
        # google_key = load_key("GOOGLE_API_KEY")
        # print(f"Loaded Google API Key (first 5 chars): {google_key[:5]}...")
    except ValueError as e:
        print(f"Error: {e}")