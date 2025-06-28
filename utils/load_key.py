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
    # 获取当前文件 (load_key.py) 的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建到项目根目录的路径，假设 load_key.py 在 utils/ 目录下
    # 所以需要向上两级目录才能到达项目根目录
    project_root = os.path.abspath(os.path.join(current_dir, '..')) # utils -> project_root
    
    # 构造 keys.json 在项目根目录的完整路径
    keys_file_path = os.path.join(project_root, "keys.json")
    
    keys_data = {}

    # 检查 keys.json 文件是否存在于指定的根目录路径
    if os.path.exists(keys_file_path):
        try:
            with open(keys_file_path, "r", encoding="utf-8") as file:
                keys_data = json.load(file)
        except json.JSONDecodeError:
            print(f"Warning: {keys_file_path} is not a valid JSON file. It will be overwritten if keys are updated.")
            keys_data = {} # 如果JSON解析失败，则清空数据，重新开始

    # 检查所需的 key 是否存在且有值
    if key_name in keys_data and keys_data[key_name]:
        print(f"Successfully loaded {key_name} from {keys_file_path}.")
        return keys_data[key_name]
    else:
        # 提示用户输入密钥
        # 注意：getpass.getpass 不适用于 Streamlit Web UI，仅适用于命令行环境
        key_value = getpass.getpass(f"'{key_name}' not found in {keys_file_path} or is empty. Please enter your {key_name}: ").strip()
        
        if not key_value:
            raise ValueError(f"'{key_name}' cannot be empty. Please provide a valid key.")

        # 将新的密钥保存到字典中
        keys_data[key_name] = key_value

        # 将更新后的字典写入 keys.json 文件
        try:
            with open(keys_file_path, "w", encoding="utf-8") as file:
                json.dump(keys_data, file, indent=4) # indent=4 使JSON文件格式化，易于阅读
            print(f"'{key_name}' saved to {keys_file_path}.")
        except IOError as e:
            print(f"Error saving {key_name} to {keys_file_path}: {e}")
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