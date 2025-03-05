import json
import os
import munch  # 用于将字典转换为对象形式
import random
import numpy as np

# 从JSON文件中获取配置
def get_config_from_json(json_file):
    """
    从JSON文件中获取配置，并将其解析为字典格式。

    参数:
    json_file: JSON配置文件的路径

    返回:
    config_dict: 包含配置的字典
    """
    # 从提供的JSON文件中解析配置
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)  # 将JSON文件加载为字典格式

    return config_dict  # 返回解析后的字典

# 处理JSON文件并将其转换为可以通过属性访问的配置对象
def process_config(jsonfile):
    """
    将JSON配置文件转换为可以通过属性访问的Munch对象。

    参数:
    jsonfile: JSON配置文件的路径

    返回:
    config: 包含配置的Munch对象
    """
    config_dict = get_config_from_json(jsonfile)  # 调用get_config_from_json获取字典格式的配置
    config = munch.Munch(config_dict)  # 使用Munch将字典转换为可通过属性访问的对象
    return config  # 返回Munch对象
