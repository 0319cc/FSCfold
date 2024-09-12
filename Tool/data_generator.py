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

perm = list(product(np.arange(4), np.arange(4)))
perm2 = [[1,3],[3,1]]
perm_nc = [[0, 0], [0, 2], [0, 3], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 3]]


class RNASSDataGenerator(object):
    def __init__(self, data_dir, split, upsampling=False):
        self.data_dir = data_dir
        self.split = split
        self.upsampling = upsampling
        # Load vocab explicitly when needed
        self.load_data()
        # Reset batch pointer to zero
        self.batch_pointer = 0

    def load_data(self):
        p = Pool()
        data_dir = self.data_dir
        # Load the current split
        RNA_SS_data = collections.namedtuple('RNA_SS_data', 
            'seq ss_label length name pairs')
        with open(os.path.join(data_dir, '%s' % self.split), 'rb') as f:
            self.data = cPickle.load(f,encoding='iso-8859-1')
        self.data_x = np.array([instance[0] for instance in self.data])
        self.data_y = np.array([instance[1] for instance in self.data])
        # self.pairs = np.array([instance[-1] for instance in self.data])
        self.pairs = np.array([instance[-1] for instance in self.data], dtype=object)

        #pdb.set_trace()
        self.seq_length = np.array([instance[2] for instance in self.data])
        self.len = len(self.data)
        self.seq = list(p.map(encoding2seq, self.data_x))
        self.seq_max_len = len(self.data_x[0])
        self.data_name = np.array([instance[3] for instance in self.data])
        # self.matrix_rep = np.array(list(p.map(creatmat, self.seq)))
        # self.matrix_rep = np.zeros([self.len, len(self.data_x[0]), len(self.data_x[0])])


    # train的获取配对矩阵
    def pairs2map(self, pairs):
        seq_len = self.seq_max_len
        contact = np.zeros([seq_len, seq_len])
        for pair in pairs:
            contact[pair[0], pair[1]] = 1
        return contact

    #train的获取数据的各种信息
    def get_one_sample(self, index):
        data_y = self.data_y[index]
        data_seq = self.data_x[index]
	#data_len = np.nonzero(self.data_x[index].sum(axis=2))[0].max()
        data_len = self.seq_length[index]
        data_pair = self.pairs[index]
        data_name = self.data_name[index]

        contact= self.pairs2map(data_pair)
        matrix_rep = np.zeros(contact.shape)
        return contact, data_seq, matrix_rep, data_len, data_name

# 预测的数据加载类
class RNASSDataGenerator_input(object):
    def __init__(self,data_dir, split):
        self.data_dir = data_dir
        self.split = split
        self.load_data()

    def load_data(self):
        p = Pool()
        data_dir = self.data_dir
        RNA_SS_data = collections.namedtuple('RNA_SS_data',
                    'seq ss_label length name pairs')
        input_file = open(os.path.join(data_dir, '%s.txt' % self.split),'r').readlines()
        self.data_name = np.array([itm.strip()[1:] for itm in input_file if itm.startswith('>')])
        self.seq = [itm.strip().upper().replace('T','U') for itm in input_file if itm.upper().startswith(('A','U','C','G','T'))]
        self.len = len(self.seq)
        self.seq_length = np.array([len(item) for item in self.seq])
        self.data_x = np.array([self.one_hot_600(item) for item in self.seq])
        self.seq_max_len = 600
        self.data_y = self.data_x

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
        one_hot_matrix_600[:len(seq_item),] = feat
        return one_hot_matrix_600

    def get_one_sample(self, index):

        # This will return a smaller size if not sufficient
        # The user must pad the batch in an external API
        # Or write a TF module with variable batch size
        #data_y = self.data_y[index]
        data_seq = self.data_x[index]
        data_len = self.seq_length[index]
        #data_pair = self.pairs[index]
        data_name = self.data_name[index]

        #contact= self.pairs2map(data_pair)
        #matrix_rep = np.zeros(contact.shape)
        #return contact, data_seq, matrix_rep, data_len, data_name
        return data_seq, data_len, data_name

# predict的数据加载器
class Dataset_new(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data):
        'Initialization'
        self.data = data

  def __len__(self):
        'Denotes the total number of samples'
        return self.data.len

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        data_seq, data_len, data_name = self.data.get_one_sample(index)
        l = get_cut_len(data_len,80)
        data_fcn = np.zeros((16, l, l))
        feature = np.zeros((8,l,l))
        if l >= 500:
            contact_adj = np.zeros((l, l))
            #contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            #contact = contact_adj
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1))
        data_fcn_1 = np.zeros((1,l,l))
        data_fcn_1[0,:data_len,:data_len] = creatmat(data_seq[:data_len,])
        data_fcn_2 = np.concatenate((data_fcn,data_fcn_1),axis=0)
        return data_fcn_2, data_len, data_seq[:l], data_name


# Train 的数据加载器
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

  def merge_data(self,data_list):
	
        self.data2.data_x = np.concatenate((data_list[0].data_x,data_list[1].data_x),axis=0)
        self.data2.data_y = np.concatenate((data_list[0].data_y,data_list[1].data_y),axis=0)
        self.data2.seq_length = np.concatenate((data_list[0].seq_length,data_list[1].seq_length),axis=0)
        self.data2.pairs = np.concatenate((data_list[0].pairs,data_list[1].pairs),axis=0)
        self.data2.data_name = np.concatenate((data_list[0].data_name,data_list[1].data_name),axis=0)
        for item in data_list[2:]:
            self.data2.data_x = np.concatenate((self.data2.data_x,item.data_x),axis=0) 
            self.data2.data_y = np.concatenate((self.data2.data_y,item.data_y),axis=0) 
            self.data2.seq_length = np.concatenate((self.data2.seq_length,item.seq_length),axis=0) 
            self.data2.pairs = np.concatenate((self.data2.pairs,item.pairs),axis=0) 
            self.data2.data_name = np.concatenate((self.data2.data_name,item.data_name),axis=0) 

        self.data2.len = len(self.data2.data_name)
        return self.data2

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample(index)
        l = get_cut_len(data_len,80)        # 获取长度
        data_fcn = np.zeros((16, l, l))
        feature = np.zeros((8,l,l))
        if l >= 500:
            contact_adj = np.zeros((l, l))
            contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            contact = contact_adj
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1))
        data_fcn_1 = np.zeros((1,l,l))
        data_fcn_1[0,:data_len,:data_len] = creatmat(data_seq[:data_len,])
        zero_mask = z_mask(data_len)[None, :, :, None]
        label_mask = l_mask(data_seq, data_len)
        temp = data_seq[None, :data_len, :data_len]
        temp = np.tile(temp, (temp.shape[1], 1, 1))
        feature[:,:data_len,:data_len] = np.concatenate([temp, np.transpose(temp, [1, 0, 2])], 2).reshape((-1,data_len,data_len))
        feature = np.concatenate((data_fcn,feature),axis=0)
        #return contact[:l, :l], data_fcn, feature, matrix_rep, data_len, data_seq[:l], data_name
        #return contact[:l, :l], data_fcn, data_fcn, matrix_rep, data_len, data_seq[:l], data_name
        data_fcn_2 = np.concatenate((data_fcn,data_fcn_1),axis=0) 
        return contact[:l, :l], data_fcn_2, matrix_rep, data_len, data_seq[:l], data_name
        #return contact[:l, :l], data_fcn_2, data_fcn_1, matrix_rep, data_len, data_seq[:l], data_name

# test的训练加载器
class Dataset_new_canonicle(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data):
        'Initialization'
        self.data = data

  def __len__(self):
        'Denotes the total number of samples'
        return self.data.len

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample(index)
        #contact, data_seq, matrix_rep, data_len, data_name, data_pair = self.data.get_one_sample_addpairs(index)
        l = get_cut_len(data_len,80)
        data_fcn = np.zeros((16, l, l))
        #data_nc = np.zeros((2, l, l))
        data_nc = np.zeros((10, l, l))
        feature = np.zeros((8,l,l))
        if l >= 500:
            contact_adj = np.zeros((l, l))
            contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            contact = contact_adj
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1))
        for n, cord in enumerate(perm_nc):
            i, j = cord
            data_nc[n, :data_len, :data_len] = np.matmul(data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1))
        data_nc = data_nc.sum(axis=0).astype(np.bool_)
        data_fcn_1 = np.zeros((1,l,l))
        data_fcn_1[0,:data_len,:data_len] = creatmat(data_seq[:data_len,])
        data_fcn_2 = np.concatenate((data_fcn,data_fcn_1),axis=0)
        #return contact[:l, :l], data_fcn_2, matrix_rep, data_len, data_seq[:l], data_name, data_nc, data_pair
        return contact[:l, :l], data_fcn_2, matrix_rep, data_len, data_seq[:l], data_name, data_nc,l

class Dataset_Cut_input(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data):
        'Initialization'
        self.data = data

  def __len__(self):
        'Denotes the total number of samples'
        return self.data.len

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        data_seq, data_len, data_name = self.data.get_one_sample(index)
        l = get_cut_len(data_len,160)
        data_fcn = np.zeros((16, l, l))
        if l >= 500:
            contact_adj = np.zeros((l, l))
            #contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            #contact = contact_adj
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1))
        return  data_fcn, data_len, data_seq[:l], data_name
        #return contact[:l, :l], data_fcn, matrix_rep, data_len, data_seq[:l], data_name

def get_cut_len(data_len,set_len):
    l = data_len
    if l <= set_len:
        l = set_len
    else:
        l = (((l - 1) // 16) + 1) * 16
    return l

def z_mask(seq_len):
    mask = np.ones((seq_len, seq_len))
    return np.triu(mask, 2)

def l_mask(inp, seq_len):
    temp = []
    mask = np.ones((seq_len, seq_len))
    for k, K in enumerate(inp):
        if np.any(K == -1) == True:
            temp.append(k)
    mask[temp, :] = 0
    mask[:, temp] = 0
    return np.triu(mask, 2)

# 创建编码矩阵
def creatmat(data, device=None):                                                            # 计算编码矩阵
    if device==None:
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    with torch.no_grad():
        # data = ''.join(['AUCG'[list(d).index(1)] for d in data])
        # data = ''.join(['AUCG'[list(d).index(1)] for d in data])
        data = ''.join(['AUCG'[list(d).index(1)] if 1 in d else 'N' for d in data])
        paired = defaultdict(int, {'AU':2, 'UA':2, 'GC':3, 'CG':3, 'UG':0.8, 'GU':0.8})

        mat = torch.tensor([[paired[x+y] for y in data] for x in data]).to(device)
        n = len(data)

        i, j = torch.meshgrid(torch.arange(n).to(device), torch.arange(n).to(device), indexing=None)
        t = torch.arange(30).to(device)
        m1 = torch.where((i[:, :, None] - t >= 0) & (j[:, :, None] + t < n),
                         mat[torch.clamp(i[:, :, None] - t, 0, n - 1), torch.clamp(j[:, :, None] + t, 0, n - 1)], 0.0)
        m1 *= torch.exp(-0.5 * t * t)

        m1_0pad = torch.nn.functional.pad(m1, (0, 1))
        first0 = torch.argmax((m1_0pad == 0).to(int), dim=2)
        to0indices = t[None, None, :] > first0[:, :, None]
        m1[to0indices] = 0.0
        m1 = m1.sum(dim=2)

        t = torch.arange(1, 30).to(device)
        m2 = torch.where((i[:, :, None] + t < n) & (j[:, :, None] - t >= 0),
                         mat[torch.clamp(i[:, :, None] + t, 0, n - 1), torch.clamp(j[:, :, None] - t, 0, n - 1)], 0.0)
        m2 *= torch.exp(-0.5 * t * t)

        m2_0pad = torch.nn.functional.pad(m2, (0, 1))
        first0 = torch.argmax((m2_0pad == 0).to(int), dim=2)
        to0indices = torch.arange(29).to(device)[None, None, :] > first0[:, :, None]
        m2[to0indices] = 0.0
        m2 = m2.sum(dim=2)
        m2[m1 == 0] = 0.0

        return (m1+m2).to(torch.device('cpu'))
