import os
import shutil
import pickle
import collections

# 从.pickle文件中查找RNStralign 数据集中的rna .ct文件


# 定义RNA_SS_data namedtuple
RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')


def read_pickle_file(filename):
    """读取pickle文件并返回数据"""
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def copy_ct_files(pickle_file, source_dir, target_dir):
    """
    复制pickle_file中name字段对应的.ct文件到target_dir文件夹中
    :param pickle_file: pickle文件路径
    :param source_dir: RNAStrAlign文件夹路径
    :param target_dir: 目标文件夹路径
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 读取pickle文件中的数据
    data = read_pickle_file(pickle_file)
    count = 0
    for item in data:
        relative_path = item.name.replace('./RNAStrAlign/', '')
        ct_file_path = os.path.join(source_dir, relative_path)
        if os.path.exists(ct_file_path):
            # 获取.ct文件的文件名
            ct_file_name = os.path.basename(ct_file_path)
            # 构建目标文件路径
            target_file = os.path.join(target_dir, ct_file_name)
            # 复制文件
            count = count + 1
            shutil.copy2(ct_file_path, target_file)
            print(f"Copied {ct_file_path} to {target_file}")
        else:
            print(f"File {ct_file_path} not found.")

    print(count)


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在文件夹的绝对路径
    pickle_file = os.path.join(current_dir, '../data/test_no_redundant_600.pickle')  # pickle文件路径
    source_dir = os.path.join(current_dir, 'RNAStrAlign')  # RNAStrAlign文件夹路径
    target_dir = os.path.join(current_dir, 'test_no_redundant_600')  # 目标文件夹路径

    copy_ct_files(pickle_file, source_dir, target_dir)
