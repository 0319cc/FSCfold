import numpy as np
import os
import collections
import pickle as cPickle
import random
import sys

# Use bpseq to cPickle文件
# def one_hot(seq):
#     BASES = 'AUCG'
#     base_dict = {base: i for i, base in enumerate(BASES)}
#     feat = np.zeros((len(seq), len(BASES)), dtype=int)
#     for i, base in enumerate(seq.upper()):
#         if base in base_dict:
#             feat[i, base_dict[base]] = 1
#     return feat
#
#
# def clean_pair(pair_list, seq):
#     valid_pairs = {'A': 'U', 'U': 'A', 'C': 'G', 'G': 'C'}
#     return [item for item in pair_list if seq[item[0]] in valid_pairs and valid_pairs[seq[item[0]]] == seq[item[1]]]
#
#
# def process_file(file_dir, item_file):
#     seq = []
#     t1 = []
#     t2 = []
#
#     with open(file_dir + item_file) as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) == 3:
#                 try:
#                     idx, base, pair = parts
#                     seq.append(base)
#                     t1.append(int(idx) - 1)
#                     t2.append(int(pair) - 1 if pair != '0' else -1)
#                 except ValueError as e:
#                     print(f"Error processing line in file {item_file}: {line} - {e}")
#
#     seq = ''.join(seq)
#     one_hot_matrix = one_hot(seq)
#
#     pair_dict_all_list = [[t1[i], t2[i]] for i in range(len(t1)) if t2[i] != -1]
#     pair_dict_all_list = clean_pair(pair_dict_all_list, seq)
#
#     seq_name = item_file
#     seq_len = len(seq)
#     pair_dict_all = {item[0]: item[1] for item in pair_dict_all_list if item[0] < item[1]}
#
#     if seq_len > 0 and seq_len <= 600:
#         ss_label = np.zeros((seq_len, 3), dtype=int)
#         ss_label[list(pair_dict_all.keys()),] = [0, 1, 0]
#         ss_label[list(pair_dict_all.values()),] = [0, 0, 1]
#         ss_label[np.where(np.sum(ss_label, axis=1) == 0)[0],] = [1, 0, 0]
#
#         one_hot_matrix_600 = np.zeros((600, 4))
#         one_hot_matrix_600[:seq_len, ] = one_hot_matrix
#         ss_label_600 = np.zeros((600, 3), dtype=int)
#         ss_label_600[:seq_len, ] = ss_label
#         ss_label_600[np.where(np.sum(ss_label_600, axis=1) == 0)[0],] = [1, 0, 0]
#
#         return RNA_SS_data(seq=one_hot_matrix_600, ss_label=ss_label_600, length=seq_len, name=seq_name,
#                            pairs=pair_dict_all_list)
#     return None
#
#
# if __name__ == '__main__':
#     file_dir = 'data/bpnew/'
#     all_files = os.listdir(file_dir)
#     random.seed(4)
#     random.shuffle(all_files)
#
#     RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')
#     all_files_list = []
#
#     for index, item_file in enumerate(all_files):
#         try:
#             sample = process_file(file_dir, item_file)
#             if sample:
#                 all_files_list.append(sample)
#         except Exception as e:
#             print(f"Error processing file {item_file}: {e}")
#
#         if index % 1000 == 0:
#             print('current processing %d/%d' % (index + 1, len(all_files)))
#
#     print(len(all_files_list))
#     cPickle.dump(all_files_list, open("data/bpnew.cPickle", "wb"))


# Use ct to cPickle file
def one_hot(seq):
    BASES = 'AUCG'
    base_dict = {base: i for i, base in enumerate(BASES)}
    feat = np.zeros((len(seq), len(BASES)), dtype=int)
    for i, base in enumerate(seq.upper()):
        if base in base_dict:
            feat[i, base_dict[base]] = 1
    return feat

def clean_pair(pair_list, seq):
    valid_pairs = {'A': 'U', 'U': 'A', 'C': 'G', 'G': 'C'}
    return [item for item in pair_list if seq[item[0]] in valid_pairs and valid_pairs[seq[item[0]]] == seq[item[1]]]

def process_file(file_dir, item_file):
    seq = []
    t1 = []
    t2 = []

    with open(file_dir + item_file) as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip header line if present
            parts = line.strip().split()
            if len(parts) >= 6:
                try:
                    idx, base, _, _, pair, _ = parts
                    seq.append(base)
                    t1.append(int(idx) - 1)
                    t2.append(int(pair) - 1 if pair != '0' else -1)
                except ValueError as e:
                    print(f"Error processing line in file {item_file}: {line} - {e}")

    seq = ''.join(seq)
    one_hot_matrix = one_hot(seq)

    # Create pairs from the parsed data
    pair_dict_all_list = [[t1[i], t2[i]] for i in range(len(t1)) if t2[i] != -1]
    pair_dict_all_list = clean_pair(pair_dict_all_list, seq)

    seq_name = item_file
    seq_len = len(seq)
    pair_dict_all = {item[0]: item[1] for item in pair_dict_all_list if item[0] < item[1]}

    if seq_len > 0 and seq_len <= 600:
        ss_label = np.zeros((seq_len, 3), dtype=int)
        ss_label[list(pair_dict_all.keys()),] = [0, 1, 0]
        ss_label[list(pair_dict_all.values()),] = [0, 0, 1]
        ss_label[np.where(np.sum(ss_label, axis=1) == 0)[0],] = [1, 0, 0]

        one_hot_matrix_600 = np.zeros((600, 4))
        one_hot_matrix_600[:seq_len, ] = one_hot_matrix
        ss_label_600 = np.zeros((600, 3), dtype=int)
        ss_label_600[:seq_len, ] = ss_label
        ss_label_600[np.where(np.sum(ss_label_600, axis=1) == 0)[0],] = [1, 0, 0]

        return RNA_SS_data(seq=one_hot_matrix_600, ss_label=ss_label_600, length=seq_len, name=seq_name,
                           pairs=pair_dict_all_list)
    return None

if __name__ == '__main__':
    file_dir = 'data/predict/'
    all_files = os.listdir(file_dir)
    random.seed(4)
    random.shuffle(all_files)

    RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')
    all_files_list = []

    for index, item_file in enumerate(all_files):
        try:
            sample = process_file(file_dir, item_file)
            if sample:
                all_files_list.append(sample)
        except Exception as e:
            print(f"Error processing file {item_file}: {e}")

        if index % 1000 == 0:
            print(f'current processing {index + 1}/{len(all_files)}')

    print(len(all_files_list))
    cPickle.dump(all_files_list, open("data/predict.cPickle", "wb"))