import numpy as np
import os
import _pickle as cPickle
import collections
from Tool.utils import *
from multiprocessing import Pool
from torch.utils import data
from collections import Counter
from random import shuffle
import torch
from itertools import permutations, product
import pdb
from collections import defaultdict

import math

# 定义排列组合，用于后续生成特征矩阵
perm = list(product(np.arange(4), np.arange(4)))
perm2 = [[1,3],[3,1]]
perm_nc = [[0, 0], [0, 2], [0, 3], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 3]]

# RNA数据生成器类，用于处理RNA数据
class RNASSDataGenerator(object):
    def __init__(self, data_dir, split, upsampling=False):
        """
          data_dir: 数据所在的文件夹路径
          split: 数据集划分文件名称
          upsampling: 是否进行上采样
        """
        self.data_dir = data_dir
        self.split = split
        self.upsampling = upsampling
        # Load vocab explicitly when needed
        self.load_data()
        # Reset batch pointer to zero
        self.batch_pointer = 0

    def load_data(self):
        p = Pool()  # 使用多进程加速数据处理
        data_dir = self.data_dir
        # 定义RNA数据结构
        RNA_SS_data = collections.namedtuple('RNA_SS_data',
            'seq ss_label length name pairs')
        # 从Pickle文件加载数据
        with open(os.path.join(data_dir, '%s' % self.split), 'rb') as f:
            self.data = cPickle.load(f,encoding='iso-8859-1')
        # 分别提取序列、标签、配对信息等
        self.data_x = np.array([instance[0] for instance in self.data])
        self.data_y = np.array([instance[1] for instance in self.data])
        # self.pairs = np.array([instance[-1] for instance in self.data])
        self.pairs = np.array([instance[-1] for instance in self.data], dtype=object)

        #pdb.set_trace()
        self.seq_length = np.array([instance[2] for instance in self.data])  # 获取序列长度
        self.len = len(self.data)       # 数据长度
        self.seq = list(p.map(encoding2seq, self.data_x))   # 对序列进行编码
        self.seq_max_len = len(self.data_x[0])          # 序列的最大长度
        self.data_name = np.array([instance[3] for instance in self.data])          # 获取数据的名称
        # self.matrix_rep = np.array(list(p.map(creatmat, self.seq)))
        # self.matrix_rep = np.zeros([self.len, len(self.data_x[0]), len(self.data_x[0])])


    # 将RNA的配对关系转换为接触矩阵
    def pairs2map(self, pairs):
        seq_len = self.seq_max_len
        contact = np.zeros([seq_len, seq_len])
        for pair in pairs:
            contact[pair[0], pair[1]] = 1           # 根据配对信息生成接触矩阵
        return contact

    # 获取单个样本的数据
    def get_one_sample(self, index):
        data_y = self.data_y[index]      # 获取对应的标签
        data_seq = self.data_x[index]       # 获取序列数据
	#data_len = np.nonzero(self.data_x[index].sum(axis=2))[0].max()
        data_len = self.seq_length[index]        # 获取序列长度
        data_pair = self.pairs[index]           # 获取配对信息
        data_name = self.data_name[index]       # 获取样本名称

        contact= self.pairs2map(data_pair)      # 生成接触矩阵
        matrix_rep = np.zeros(contact.shape)         # 初始化矩阵表示
        return contact, data_seq, matrix_rep, data_len, data_name

# 预测时的数据加载类
class RNASSDataGenerator_input(object):
    def __init__(self,data_dir, split):
        """
        data_dir: 数据所在的文件夹路径
        split: 数据集划分文件名称
        """
        self.data_dir = data_dir
        self.split = split
        self.load_data()

    def load_data(self):
        """
        从txt文件加载预测数据，并进行处理
        """
        p = Pool()
        data_dir = self.data_dir
        RNA_SS_data = collections.namedtuple('RNA_SS_data',
                    'seq ss_label length name pairs')
        input_file = open(os.path.join(data_dir, '%s.txt' % self.split),'r').readlines()     # 读取输入文件
        self.data_name = np.array([itm.strip()[1:] for itm in input_file if itm.startswith('>')])   # 解析序列名称
        self.seq = [itm.strip().upper().replace('T','U') for itm in input_file if itm.upper().startswith(('A','U','C','G','T'))]        # 解析序列
        self.len = len(self.seq)             # 序列长度
        self.seq_length = np.array([len(item) for item in self.seq])                 # 每个序列的长度
        self.data_x = np.array([self.one_hot_600(item) for item in self.seq])       # 对序列进行One-Hot编码
        self.seq_max_len = 600      # 设置最大序列长度为600
        self.data_y = self.data_x        # 标签与数据相同

    # 将序列进行One-Hot编码，确保编码后的矩阵大小为600
    def one_hot_600(self,seq_item):
        RNN_seq = seq_item
        BASES = 'AUCG'
        bases = np.array([base for base in BASES])
        feat = np.concatenate(
                [[(bases == base.upper()).astype(int)] if str(base).upper() in BASES else np.array([[-1] * len(BASES)]) for base
                in RNN_seq])
        if len(seq_item) <= 600:
            one_hot_matrix_600 = np.zeros((600,4))
        else:
            one_hot_matrix_600 = np.zeros((600,4))
            # one_hot_matrix_600 = np.zeros((len(seq_item),4))
        one_hot_matrix_600[:len(seq_item),] = feat      # 将编码矩阵填充到600长度
        return one_hot_matrix_600

    # 获取一个样本的数据
    def get_one_sample(self, index):

        # This will return a smaller size if not sufficient
        # The user must pad the batch in an external API
        # Or write a TF module with variable batch size
        #data_y = self.data_y[index]
        data_seq = self.data_x[index]       # 获取One-Hot编码后的序列
        data_len = self.seq_length[index]   # 获取序列长度
        #data_pair = self.pairs[index]
        data_name = self.data_name[index]   # 获取样本名称

        #contact= self.pairs2map(data_pair)
        #matrix_rep = np.zeros(contact.shape)
        #return contact, data_seq, matrix_rep, data_len, data_name
        return data_seq, data_len, data_name

# PyTorch数据集类，用于数据的批量加载（预测）
class Dataset_new(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data):
        'Initialization'
        self.data = data

  def __len__(self):
        '返回样本数量'
        return self.data.len

  def __getitem__(self, index):
        '生成一个样本的数据'
        # Select sample
        data_seq, data_len, data_name = self.data.get_one_sample(index)      # 获取样本数据
        l = get_cut_len(data_len,80)        # 获取适合的序列长度
        data_fcn = np.zeros((16, l, l))             # 初始化特征矩阵
        feature = np.zeros((8,l,l))                 # 初始化其他特征矩阵
        if l >= 500:
            # 如果长度大于等于500，对序列进行调整
            contact_adj = np.zeros((l, l))
            #contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            #contact = contact_adj
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj
        # 生成特征矩阵
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1))
        data_fcn_1 = np.zeros((1,l,l))
        data_fcn_1[0,:data_len,:data_len] = creatmat(data_seq[:data_len,])       # 创建编码矩
        data_fcn_2 = np.concatenate((data_fcn,data_fcn_1),axis=0)           # 拼接特征矩阵
        return data_fcn_2, data_len, data_seq[:l], data_name


# 用于训练数据的加载器
class Dataset_new_merge_multi(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data_list):
        'Initialization'
        self.data2 = data_list[0]
        if len(data_list) > 1:
            self.data = self.merge_data(data_list)
        else:
            self.data = self.data2

    def __len__(self):
        'Denotes the total number of samples'
        return self.data.len

    def merge_data(self, data_list):

        self.data2.data_x = np.concatenate((data_list[0].data_x, data_list[1].data_x), axis=0)
        self.data2.data_y = np.concatenate((data_list[0].data_y, data_list[1].data_y), axis=0)
        self.data2.seq_length = np.concatenate((data_list[0].seq_length, data_list[1].seq_length), axis=0)
        self.data2.pairs = np.concatenate((data_list[0].pairs, data_list[1].pairs), axis=0)
        self.data2.data_name = np.concatenate((data_list[0].data_name, data_list[1].data_name), axis=0)
        for item in data_list[2:]:
            self.data2.data_x = np.concatenate((self.data2.data_x, item.data_x), axis=0)
            self.data2.data_y = np.concatenate((self.data2.data_y, item.data_y), axis=0)
            self.data2.seq_length = np.concatenate((self.data2.seq_length, item.seq_length), axis=0)
            self.data2.pairs = np.concatenate((self.data2.pairs, item.pairs), axis=0)
            self.data2.data_name = np.concatenate((self.data2.data_name, item.data_name), axis=0)

        self.data2.len = len(self.data2.data_name)
        return self.data2

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample(index)
        l = get_cut_len(data_len, 80)  # 获取长度
        data_fcn = np.zeros((16, l, l))
        feature = np.zeros((8, l, l))
        if l >= 500:
            contact_adj = np.zeros((l, l))
            contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            contact = contact_adj
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(data_seq[:data_len, i].reshape(-1, 1),
                                                          data_seq[:data_len, j].reshape(1, -1))
        data_fcn_1 = np.zeros((1, l, l))
        data_fcn_1[0, :data_len, :data_len] = creatmat(data_seq[:data_len, ])
        zero_mask = z_mask(data_len)[None, :, :, None]
        label_mask = l_mask(data_seq, data_len)
        temp = data_seq[None, :data_len, :data_len]
        temp = np.tile(temp, (temp.shape[1], 1, 1))
        feature[:, :data_len, :data_len] = np.concatenate([temp, np.transpose(temp, [1, 0, 2])], 2).reshape(
            (-1, data_len, data_len))
        feature = np.concatenate((data_fcn, feature), axis=0)
        # return contact[:l, :l], data_fcn, feature, matrix_rep, data_len, data_seq[:l], data_name
        # return contact[:l, :l], data_fcn, data_fcn, matrix_rep, data_len, data_seq[:l], data_name
        data_fcn_2 = np.concatenate((data_fcn, data_fcn_1), axis=0)
        return contact[:l, :l], data_fcn_2, matrix_rep, data_len, data_seq[:l], data_name
        # return contact[:l, :l], data_fcn_2, data_fcn_1, matrix_rep, data_len, data_seq[:l], data_name

# 用于测试的训练数据加载器
class Dataset_new_canonicle(data.Dataset):
    def __init__(self, data):
        'Initialization'
        self.data = data  # 初始化数据

    def __len__(self):
        '返回样本数量'
        return self.data.len

    def __getitem__(self, index):
        '获取一个样本数据'
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample(index)  # 获取样本数据
        l = get_cut_len(data_len, 80)  # 获取合适的序列长度
        data_fcn = np.zeros((16, l, l))  # 初始化特征矩阵
        data_nc = np.zeros((10, l, l))  # 初始化非标准接触矩阵

        # 对长度超过500的样本进行调整
        if l >= 500:
            contact_adj = np.zeros((l, l))  # 调整接触矩阵
            contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            contact = contact_adj
            seq_adj = np.zeros((l, 4))  # 调整序列
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj

        # 生成16通道的特征矩阵   16*L*L
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1))

        # 生成非标准接触矩阵  10*L*L
        for n, cord in enumerate(perm_nc):
            i, j = cord
            data_nc[n, :data_len, :data_len] = np.matmul(data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1))

        data_nc = data_nc.sum(axis=0).astype(np.bool_)  # 将矩阵元素转换为布尔类型
        data_fcn_1 = np.zeros((1, l, l))
        data_fcn_1[0, :data_len, :data_len] = creatmat(data_seq[:data_len, ])  # 创建编码矩阵
        data_fcn_2 = np.concatenate((data_fcn, data_fcn_1), axis=0)  # 拼接特征矩阵 17

        return contact[:l, :l], data_fcn_2, matrix_rep, data_len, data_seq[:l], data_name, data_nc, l  # 返回样本数据


# 用于预测的数据集类
class Dataset_Cut_input(data.Dataset):
    def __init__(self, data):
        'Initialization'
        self.data = data  # 初始化数据

    def __len__(self):
        '返回样本数量'
        return self.data.len

    def __getitem__(self, index):
        '获取一个样本数据'
        data_seq, data_len, data_name = self.data.get_one_sample(index)  # 获取样本数据
        l = get_cut_len(data_len, 160)  # 获取合适的序列长度
        data_fcn = np.zeros((16, l, l))  # 初始化特征矩阵

        # 对长度超过500的样本进行调整
        if l >= 500:
            seq_adj = np.zeros((l, 4))  # 调整序列
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj

        # 生成特征矩阵
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1))

        return data_fcn, data_len, data_seq[:l], data_name  # 返回样本数据


# 获取裁剪后的序列长度，确保为16的倍数
def get_cut_len(data_len, set_len):
    l = data_len
    if l <= set_len:
        l = set_len
    else:
        l = (((l - 1) // 16) + 1) * 16  # 将长度调整为16的倍数
    return l


# 创建上三角掩码矩阵
def z_mask(seq_len):
    mask = np.ones((seq_len, seq_len))
    return np.triu(mask, 2)  # 只保留上三角部分的掩码


# 创建基于输入序列的掩码矩阵
def l_mask(inp, seq_len):
    temp = []
    mask = np.ones((seq_len, seq_len))
    for k, K in enumerate(inp):
        if np.any(K == -1):
            temp.append(k)
    mask[temp, :] = 0  # 对非标准碱基的位置进行掩码处理
    mask[:, temp] = 0
    return np.triu(mask, 2)  # 只保留上三角部分


# 创建编码矩阵
def creatmat(data, device=None):
    """
    根据RNA序列生成配对能量矩阵，通过计算相邻碱基对的能量和距离，生成最终的特征矩阵。

    参数：
    data: RNA序列的One-Hot编码表示，长度为n，每个碱基使用4位向量表示。
    device: 计算设备（默认自动选择GPU或CPU）。

    返回：
    一个形状为(n, n)的矩阵，表示RNA序列中的碱基对之间的相互作用。
    """

    # 如果未指定device，则自动选择GPU或CPU
    if device is None:
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # 关闭梯度计算以节省内存和加速计算
    with torch.no_grad():
        # 将One-Hot编码转换回RNA序列的字符表示（A, U, C, G），如果碱基不存在，标记为'N'
        data = ''.join(['AUCG'[list(d).index(1)] if 1 in d else 'N' for d in data])

        # 定义碱基对的能量值，AU/UA为2，GC/CG为3，UG/GU为0.8，其他为0
        paired = defaultdict(int, {'AU': 2, 'UA': 2, 'GC': 3, 'CG': 3, 'UG': 0.8, 'GU': 0.8})

        # 生成配对能量矩阵，mat[i, j] 表示碱基 data[i] 和 data[j] 的配对能量
        mat = torch.tensor([[paired[x + y] for y in data] for x in data]).to(device)
        n = len(data)  # 序列长度

        # 生成用于偏移计算的网格索引
        i, j = torch.meshgrid(torch.arange(n).to(device), torch.arange(n).to(device), indexing=None)

        # 定义偏移范围 t，从0到30
        t = torch.arange(30).to(device)

        # 计算正向偏移的能量矩阵，i减去t，j加上t，对应的碱基配对能量，并乘以高斯衰减因子
        m1 = torch.where((i[:, :, None] - t >= 0) & (j[:, :, None] + t < n),
                         mat[torch.clamp(i[:, :, None] - t, 0, n - 1), torch.clamp(j[:, :, None] + t, 0, n - 1)], 0.0)
        m1 *= torch.exp(-0.5 * t * t)  # 以高斯分布权重对偏移值进行衰减

        # 用0填充矩阵 m1 并查找第一个值为0的位置
        m1_0pad = torch.nn.functional.pad(m1, (0, 1))
        first0 = torch.argmax((m1_0pad == 0).to(int), dim=2)

        # 找到填充0的位置并将该位置之后的值设为0
        to0indices = t[None, None, :] > first0[:, :, None]
        m1[to0indices] = 0.0
        m1 = m1.sum(dim=2)  # 将所有的偏移能量累加

        # 计算反向偏移的能量矩阵，i加上t，j减去t，并乘以高斯衰减因子
        t = torch.arange(1, 30).to(device)
        m2 = torch.where((i[:, :, None] + t < n) & (j[:, :, None] - t >= 0),
                         mat[torch.clamp(i[:, :, None] + t, 0, n - 1), torch.clamp(j[:, :, None] - t, 0, n - 1)], 0.0)
        m2 *= torch.exp(-0.5 * t * t)  # 以高斯分布权重对偏移值进行衰减

        # 用0填充矩阵 m2 并查找第一个值为0的位置
        m2_0pad = torch.nn.functional.pad(m2, (0, 1))
        first0 = torch.argmax((m2_0pad == 0).to(int), dim=2)

        # 找到填充0的位置并将该位置之后的值设为0
        to0indices = torch.arange(29).to(device)[None, None, :] > first0[:, :, None]
        m2[to0indices] = 0.0
        m2 = m2.sum(dim=2)  # 将所有的偏移能量累加

        # 如果正向能量m1的某个位置为0，则将对应的反向能量m2也设为0
        m2[m1 == 0] = 0.0

        # 返回累加后的能量矩阵（m1 + m2），并将其移到CPU上
        return (m1 + m2).to(torch.device('cpu'))

