import os
import shutil

## 使用ct文件查找包含假结结构的所有RNA到指定文件夹
def contains_pseudoknot(ct_file_path):
    """
    判断 .ct 文件是否包含假结结构
    :param ct_file_path: .ct 文件的路径
    :return: 如果包含假结结构，返回 True，否则返回 False
    """
    try:
        with open(ct_file_path, 'r') as file:
            lines = file.readlines()
            base_pairs = []

            # 跳过文件头部的描述行
            for line in lines[1:]:
                fields = line.split()
                if len(fields) >= 6:
                    try:
                        index = int(fields[0])
                        pair = int(fields[4])
                        if pair > 0:
                            base_pairs.append((index, pair))
                    except ValueError:
                        continue

            # 检查假结结构
            for i, (index1, pair1) in enumerate(base_pairs):
                for index2, pair2 in base_pairs[i + 1:]:
                    if index1 < index2 < pair1 < pair2:
                        return True
    except Exception as e:
        print(f"Error processing file {ct_file_path}: {e}")
    return False


def copy_files_with_pseudoknots(src_folder, dst_folder):
    """
    将包含假结结构的 .ct 文件从 src_folder 复制到 dst_folder
    :param src_folder: 源文件夹路径
    :param dst_folder: 目标文件夹路径
    """
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    for filename in os.listdir(src_folder):
        if filename.endswith('.ct'):
            file_path = os.path.join(src_folder, filename)
            if contains_pseudoknot(file_path):
                shutil.copy(file_path, os.path.join(dst_folder, filename))

## 使用bpseq文件查找包含假结结构的所有RNA到指定文件夹
# def contains_pseudoknot_bpseq(bpseq_file_path):
#     """
#     判断 .bpseq 文件是否包含假结结构
#     :param bpseq_file_path: .bpseq 文件的路径
#     :return: 如果包含假结结构，返回 True，否则返回 False
#     """
#     try:
#         with open(bpseq_file_path, 'r') as file:
#             lines = file.readlines()
#             base_pairs = []
#
#             # 读取文件中的碱基对
#             for line in lines:
#                 fields = line.split()
#                 if len(fields) == 3:
#                     try:
#                         index = int(fields[0])
#                         pair = int(fields[2])
#                         if pair > 0:
#                             base_pairs.append((index, pair))
#                     except ValueError:
#                         continue
#
#             # 检查假结结构
#             for i, (index1, pair1) in enumerate(base_pairs):
#                 for index2, pair2 in base_pairs[i + 1:]:
#                     if (index1 < index2 < pair1 < pair2) or (index2 < index1 < pair2 < pair1):
#                         return True
#     except Exception as e:
#         print(f"Error processing file {bpseq_file_path}: {e}")
#     return False
#
# def copy_files_with_pseudoknots_bpseq(src_folder, dst_folder):
#     """
#     将包含假结结构的 .bpseq 文件从 src_folder 复制到 dst_folder
#     :param src_folder: 源文件夹路径
#     :param dst_folder: 目标文件夹路径
#     """
#     if not os.path.exists(dst_folder):
#         os.makedirs(dst_folder)
#
#     for filename in os.listdir(src_folder):
#         if filename.endswith('.bpseq'):
#             file_path = os.path.join(src_folder, filename)
#             if contains_pseudoknot_bpseq(file_path):
#                 shutil.copy(file_path, os.path.join(dst_folder, filename))


# 定义源文件夹和目标文件夹路径
src_folder = 'test_no_redundant_600'
dst_folder = 'test_psud'

# 复制包含假结结构的 .ct 文件
copy_files_with_pseudoknots(src_folder, dst_folder)
# 复制包含假结结构的 .bpseq 文件
# copy_files_with_pseudoknots_bpseq(src_folder, dst_folder)

def count_ct_files(folder):
    """
    计算文件夹中 .ct 文件的数量
    :param folder: 文件夹路径
    :return: .ct 文件的数量
    """
    count = 0
    for filename in os.listdir(folder):
        if filename.endswith('.ct'):
            count += 1
    return count

# 定义文件夹路径
folder_path = 'test_psud'

# 计算 .ct 文件的数量
ct_file_count = count_ct_files(folder_path)

# 打印结果
print(f"The folder '{folder_path}' contains {ct_file_count} .ct files.")