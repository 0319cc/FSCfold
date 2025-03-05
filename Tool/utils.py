import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score  # 导入性能指标
import math
import torch
import torch.nn as nn
from scipy import signal  # 用于处理信号处理相关的操作
from multiprocessing import Pool  # 多进程处理
from functools import partial
import torch.nn.functional as F  # PyTorch中的函数库
import argparse  # 用于解析命令行参数
import pandas as pd  # 数据处理库
from scipy.sparse import diags  # 稀疏矩阵工具
import random  # 随机数生成
import os  # 操作系统接口
import pdb  # 调试工具

# 碱基结构字典：'.' 表示未配对，'(' 和 ')' 分别表示左括号和右括号，表示配对的碱基
label_dict = {
    '.': np.array([1, 0, 0]),
    '(': np.array([0, 1, 0]),
    ')': np.array([0, 0, 1])
}

# 碱基序列字典，将碱基转为One-Hot编码，包括标准碱基A, U, C, G和一些IUPAC符号
seq_dict = {
    'A': np.array([1, 0, 0, 0]),
    'U': np.array([0, 1, 0, 0]),
    'C': np.array([0, 0, 1, 0]),
    'G': np.array([0, 0, 0, 1]),
    'N': np.array([0, 0, 0, 0]),  # N表示不确定碱基
    # 以下为IUPAC符号，用于表示多种可能的碱基
    'M': np.array([1, 0, 1, 0]),  # A或C
    'Y': np.array([0, 1, 1, 0]),  # C或U
    'W': np.array([1, 0, 0, 0]),  # A或U
    'V': np.array([1, 0, 1, 1]),  # A、C或G
    'K': np.array([0, 1, 0, 1]),  # U或G
    'R': np.array([1, 0, 0, 1]),  # A或G
    'I': np.array([0, 0, 0, 0]),  # I表示Inosine，通常作为不配对碱基
    'X': np.array([0, 0, 0, 0]),  # X表示不确定碱基
    'S': np.array([0, 0, 1, 1]),  # C或G
    'D': np.array([1, 1, 0, 1]),  # A、U或G
    'P': np.array([0, 0, 0, 0]),  # P表示不常见的碱基
    'B': np.array([0, 1, 1, 1]),  # C、U或G
    'H': np.array([1, 1, 1, 0])  # A、C或U
}

# 用于将编码转换回碱基的字典
char_dict = {
    0: 'A',
    1: 'U',
    2: 'C',
    3: 'G'
}


# 获取命令行参数
def get_args():
    """
    解析命令行参数，返回配置文件路径和训练/测试文件名。
    """
    argparser = argparse.ArgumentParser(description="diff through pp")
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='Tool/config.json',  # 配置文件路径
        help='The Configuration file'
    )
    argparser.add_argument('--test', type=bool, default=False,
                           help='是否跳过训练直接进行测试。')
    argparser.add_argument('--nc', type=bool, default=False,
                           help='是否预测非标准的碱基对。')
    argparser.add_argument('--train_files', type=str, required=False, nargs='+', default=[
        'RNAStralign', 'ArchiveII', 'TR0_with_data_augmentation', 'TS0', 'bpnew'],
                           help='训练文件名称列表。')
    argparser.add_argument('--test_files', required=False, nargs='?', default='ArchiveII', choices=[
        'ArchiveII', 'TS0', 'bpnew', 'RNAStralign', 'RNAStrAlign_test_pseu', 'all_600', 'RNAStrAlign_test_ct'],
                           help='测试文件名称。')
    args = argparser.parse_args()  # 返回解析后的参数
    return args


# 软符号函数，用于平滑处理输入
def soft_sign(x, k):
    return torch.sigmoid(k * x)  # logistic函数


# 序列编码函数，将碱基序列转换为One-Hot编码
def seq_encoding(string):
    """
    将RNA序列中的碱基字符转化为One-Hot编码。
    """
    str_list = list(string)
    encoding = list(map(lambda x: seq_dict[x], str_list))  # 利用seq_dict将碱基字符转为One-Hot编码
    return np.stack(encoding, axis=0)  # 将编码结果堆叠成一个矩阵


# 高斯函数
def Gaussian(x):
    return math.exp(-0.5 * (x * x))


# 碱基配对评分函数
def paired(x, y):
    """
    判断两个碱基是否可以配对，并返回配对分数。
    """
    if x == 'A' and y == 'U':
        return 2
    elif x == 'G' and y == 'C':
        return 3
    elif x == 'G' and y == 'U':
        return 0.8
    elif x == 'U' and y == 'A':
        return 2
    elif x == 'C' and y == 'G':
        return 3
    elif x == 'U' and y == 'G':
        return 0.8
    else:
        return 0  # 不能配对的碱基返回0


# 创建配对能量矩阵
def creatmat(data):
    """
    根据RNA碱基序列生成一个配对能量矩阵。

    输入:
    data: RNA碱基序列的字符表示。

    输出:
    配对能量矩阵，表示不同位置的碱基对之间的配对强度。
    """
    mat = np.zeros([len(data), len(data)])
    for i in range(len(data)):
        for j in range(len(data)):
            coefficient = 0
            # 正向扫描，计算配对强度
            for add in range(30):
                if i - add >= 0 and j + add < len(data):
                    score = paired(data[i - add], data[j + add])
                    if score == 0:  # 如果不能配对，停止当前配对
                        break
                    else:
                        coefficient += score * Gaussian(add)  # 根据距离和高斯函数调整配对强度
                else:
                    break
            # 反向扫描，计算配对强度
            if coefficient > 0:
                for add in range(1, 30):
                    if i + add < len(data) and j - add >= 0:
                        score = paired(data[i + add], data[j - add])
                        if score == 0:
                            break
                        else:
                            coefficient += score * Gaussian(add)
                    else:
                        break
            mat[i, j] = coefficient  # 更新能量矩阵
    return mat


# 创建零矩阵
def createzeromat(data):
    return np.zeros([len(data), len(data)])  # 返回全零矩阵


# 将CT格式的RNA二级结构转化为配对列表
def ct2struct(ct):
    stack = list()  # 用栈存储未闭合的碱基
    struct = list()
    for i in range(len(ct)):
        if ct[i] == '(':  # 如果是左括号，说明有一个未闭合的配对
            stack.append(i)
        if ct[i] == ')':  # 如果是右括号，弹出栈顶元素，得到配对
            left = stack.pop()
            struct.append([left, i])  # 记录配对关系
    return struct


def prob2map(prob):
    """
    将概率矩阵转换为RNA的接触矩阵。

    参数:
    prob: 一个形状为 [seq_length, 3] 的概率矩阵，其中第二列和第三列分别表示左括号和右括号的配对概率。

    返回:
    一个对称的接触矩阵，表示RNA二级结构中的配对关系，值为1表示碱基配对，-1表示不可配对区域。
    """
    contact = np.zeros([len(prob), len(prob)])  # 初始化接触矩阵
    left_index = np.where(prob[:, 1])[0]  # 查找左括号的位置索引
    right_index = np.where(prob[:, 2])[0][::-1]  # 查找右括号的位置索引，逆序处理
    ct = np.array(['.'] * len(prob))  # 初始化字符表示的结构，用"."表示未配对
    ct[left_index] = '('  # 将左括号位置标记为 '('
    ct[right_index] = ')'  # 将右括号位置标记为 ')'
    struct = ct2struct(ct)  # 调用 ct2struct 函数，将结构转换为配对列表

    # 根据结构生成接触矩阵
    for index in struct:
        index_1 = max(index[0], index[1])  # 确保第一个索引为较大值
        index_2 = min(index[0], index[1])  # 确保第二个索引为较小值
        contact[index_1, index_2] = 1  # 将接触矩阵中的对应位置设为1，表示配对

    # 将上三角矩阵的值设为 -1 表示不可配对区域
    triu_index = np.triu_indices(len(contact), k=0)
    contact[triu_index] = -1
    return contact  # 返回最终的接触矩阵


# 将接触矩阵变为对称矩阵
def contact2sym(contact):
    """
    将接触矩阵转换为对称矩阵，仅保留下三角区域并将其复制到上三角区域。

    参数:
    contact: 输入的接触矩阵。

    返回:
    对称接触矩阵。
    """
    triu_index = np.triu_indices(contact.shape[-1], k=0)  # 获取上三角区域的索引
    contact[triu_index] = 0  # 将上三角区域的值设为0
    return contact + np.transpose(contact)  # 将下三角区域的值复制到上三角区域，并返回对称矩阵


# 将概率矩阵转换为配对结构（不处理假结）
def prob2struct(prob):
    """
    将概率矩阵转换为配对结构，不处理假结。

    参数:
    prob: 一个形状为 [seq_length, 3] 的概率矩阵。

    返回:
    RNA二级结构的配对列表。
    """
    left_index = np.where(prob[:, 1])[0]  # 查找左括号的位置
    right_index = np.where(prob[:, 2])[0][::-1]  # 查找右括号的位置，逆序处理
    ct = np.array(['.'] * len(prob))  # 初始化结构字符串，未配对区域为'.'
    ct[left_index] = '('  # 左括号位置标记为 '('
    ct[right_index] = ')'  # 右括号位置标记为 ')'
    struct = ct2struct(ct)  # 调用 ct2struct 将其转换为配对列表
    return struct  # 返回配对列表


# 将One-Hot编码转为碱基序列
def encoding2seq(arr):
    """
    将One-Hot编码表示的碱基序列转换为字符表示。

    参数:
    arr: RNA序列的One-Hot编码数组。

    返回:
    RNA碱基序列的字符串表示。
    """
    seq = list()
    for arr_row in list(arr):
        if sum(arr_row) == 0:
            seq.append('.')  # 如果没有匹配到任何碱基，标记为 '.'
        else:
            seq.append(char_dict[np.argmax(arr_row)])  # 查找One-Hot编码中值为1的索引，并转换为碱基字符
    return ''.join(seq)  # 返回转换后的RNA序列字符串


# 将接触矩阵转换为CT格式的RNA二级结构表示
def contact2ct(contact, sequence_encoding, seq_len):
    """
    将接触矩阵转换为CT格式的RNA二级结构文件。

    参数:
    contact: 接触矩阵，表示配对关系。
    sequence_encoding: RNA序列的One-Hot编码。
    seq_len: RNA序列长度。

    返回:
    一个Pandas DataFrame表示的CT格式结构，包含碱基序列和其配对关系。
    """
    seq = encoding2seq(sequence_encoding)[:seq_len].replace('.', 'N')  # 将One-Hot编码转换为碱基序列，并替换未配对的碱基
    contact = contact[:seq_len, :seq_len]  # 只保留与序列长度匹配的接触矩阵
    structure = np.where(contact)  # 查找配对的位置
    pair_dict = dict()  # 用于存储配对信息的字典
    for i in range(seq_len):
        pair_dict[i] = -1  # 初始化每个碱基的配对信息为-1（未配对）
    for i in range(len(structure[0])):
        pair_dict[structure[0][i]] = structure[1][i]  # 更新配对信息

    # 构建CT格式的各列数据
    first_col = list(range(1, seq_len + 1))  # 第一列是索引
    second_col = list(seq)  # 第二列是碱基序列
    third_col = list(range(seq_len))  # 第三列是上一个碱基的索引
    fourth_col = list(range(2, seq_len + 2))  # 第四列是下一个碱基的索引
    fifth_col = [pair_dict[i] + 1 for i in range(seq_len)]  # 第五列是配对的碱基索引
    last_col = list(range(1, seq_len + 1))  # 第六列是序列位置

    # 将数据存入Pandas DataFrame中，形成CT格式
    df = pd.DataFrame()
    df['index'] = first_col
    df['base'] = second_col
    df['index-1'] = third_col
    df['index+1'] = fourth_col
    df['pair_index'] = fifth_col
    df['n'] = last_col
    return df  # 返回CT格式的DataFrame


def padding(data_array, maxlen):
    a, b = data_array.shape
    return np.pad(data_array, ((0,maxlen-a),(0,0)), 'constant')

def F1_low_tri(opt_state, true_a):
	tril_index = np.tril_indices(len(opt_state),k=-1)
	return f1_score(true_a[tril_index], opt_state[tril_index])

def acc_low_tri(opt_state, true_a):
	tril_index = np.tril_indices(len(opt_state),k=-1)
	return accuracy_score(true_a[tril_index], opt_state[tril_index])



def logit2binary(pred_contacts):
    sigmoid_results = torch.sigmoid(pred_contacts)
    binary = torch.where(sigmoid_results > 0.5, 
        torch.ones(pred_contacts.shape), 
        torch.zeros(pred_contacts.shape))
    return binary


def unravel2d_torch(ind, ncols):
    x = ind / ncols 
    y = ind % ncols
    return (int(x),int(y))


def postprocess_sort(contact):
    ncols = contact.shape[-1]
    contact = torch.sigmoid(contact)
    contact_flat = contact.reshape(-1)
    final_contact = torch.zeros(contact.shape)
    contact_sorted, sorted_ind = torch.sort(contact_flat,descending=True)
    ind_one = sorted_ind[contact_sorted>0.9]
    length = len(ind_one)
    use = min(length, 10000)
    ind_list = list(map(lambda x: unravel2d_torch(x, ncols), ind_one[:use]))
    row_list = list()
    col_list = list()
    for ind_x, ind_y in ind_list:
        if (ind_x not in row_list) and (ind_y not in col_list):
            row_list.append(ind_x)
            col_list.append(ind_y)
            final_contact[ind_x, ind_y] = 1
        else:
            ind_list.remove((ind_x, ind_y))
    return final_contact

def conflict_sort(contacts):
    processed = list(map(postprocess_sort, 
        list(contacts)))
    return processed


def check_thredhold(pred_contacts, contacts):
    a = contacts[0]
    b = pred_contacts[0]
    b = torch.sigmoid(b)
    b = b.cpu().numpy()
    print(len(np.where(a>0)[0]))
    print(len(np.where(b>0.5)[0]))
    print(min(b[np.where(a>0)]))
    print(len(np.where(b> min(b[np.where(a>0)]))[0]))

def postprocess_sampling(contact):
    from scipy.special import softmax
    ncols = contact.shape[-1]
    contact = torch.sigmoid(contact)
    contact_flat = contact.reshape(-1)
    final_contact = torch.zeros(contact.shape)
    contact_sorted, sorted_ind = torch.sort(contact_flat,descending=True)
    ind_one = sorted_ind[contact_sorted>0.5]
    used_values = contact_sorted[contact_sorted>0.5]
    ind_list = list(map(lambda x: unravel2d_torch(x, ncols), ind_one))
    row_list = list()
    col_list = list()
    prob = used_values.cpu().numpy()
    # for each step, sample one from the list
    # then remove that value and the index, add it into the row and col list
    # before add check the row and col list first, to avoid conflict
    for i in range(len(used_values)):
        prob = softmax(prob)
        ind = int(np.random.choice(len(prob), 1, p=prob))
        ind_x, ind_y = ind_list[ind]
        if (ind_x not in row_list) and (ind_y not in col_list):
            row_list.append(ind_x)
            col_list.append(ind_y)
            final_contact[ind_x, ind_y] = 1
        ind_list.remove((ind_x, ind_y))
        prob = np.delete(prob, ind)
    return final_contact

def conflict_sampling(contacts):
    processed = list(map(postprocess_sampling, 
        list(contacts)))
    return processed


# we first apply a kernel to the ground truth a
# then we multiple the kernel with the prediction, to get the TP allows shift
# then we compute f1
# we unify the input all as the symmetric matrix with 0 and 1, 1 represents pair
def evaluate_shifted(pred_a, true_a):
    kernel = np.array([[0.0,1.0,0.0],
                        [1.0,1.0,1.0],
                        [0.0,1.0,0.0]])
    pred_a_filtered = signal.convolve2d(pred_a, kernel, 'same')
    fn = len(torch.where((true_a - torch.Tensor(pred_a_filtered))==1)[0])
    pred_p = torch.sign(torch.Tensor(pred_a)).sum()
    true_p = true_a.sum()
    tp = true_p - fn
    fp = pred_p - tp
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    if np.isnan(precision):
        #pdb.set_trace()
        precision = 0
    f1_score = 2*tp/(2*tp + fp + fn)
    return precision, recall, f1_score
# 计算准确性，召回率，f1性能参数
def evaluate_exact_new(pred_a, true_a, eps=1e-11):
    tp_map = torch.sign(torch.Tensor(pred_a)*torch.Tensor(true_a))
    tp = tp_map.sum()
    pred_p = torch.sign(torch.Tensor(pred_a)).sum()
    true_p = true_a.sum()
    fp = pred_p - tp
    fn = true_p - tp
    # recall = tp/(tp+fn)
    # precision = tp/(tp+fp)
    # f1_score = 2*tp/(2*tp + fp + fn)
    recall = (tp + eps)/(tp+fn+eps)
    precision = (tp + eps)/(tp+fp+eps)
    f1_score = (2*tp + eps)/(2*tp + fp + fn + eps)
    return precision, recall, f1_score

def evaluate_exact(pred_a, true_a):
    tp_map = torch.sign(torch.Tensor(pred_a)*torch.Tensor(true_a))
    tp = tp_map.sum()
    pred_p = torch.sign(torch.Tensor(pred_a)).sum()
    true_p = true_a.sum()
    fp = pred_p - tp
    fn = true_p - tp
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    if np.isnan(precision):
        #pdb.set_trace()
        precision = 0
    f1_score = 2*tp/(2*tp + fp + fn)
    return precision, recall, f1_score

def test_evaluation():
    pred_a = np.zeros([4,5])
    true_a = np.zeros([4,5])
    true_a[0,1]=1;true_a[1,1]=1;true_a[2,2]=1;true_a[3,3]=1
    pred_a[0,2]=1;pred_a[1,2]=1;pred_a[2,0]=1;pred_a[3,3]=1;pred_a[3,1]=1
    print(evaluate_shifted(pred_a, true_a))
    print(evaluate_exact(pred_a, true_a))

def constraint_matrix(x):
    base_a = x[:, 0]
    base_u = x[:, 1]
    base_c = x[:, 2]
    base_g = x[:, 3]
    au = torch.matmul(base_a.view(-1, 1), base_u.view(1, -1))
    au_ua = au + au.t()
    cg = torch.matmul(base_c.view(-1, 1), base_g.view(1, -1))
    cg_gc = cg + cg.t()
    ug = torch.matmul(base_u.view(-1, 1), base_g.view(1, -1))
    ug_gu = ug + ug.t()
    return au_ua + cg_gc + ug_gu

def constraint_matrix_batch(x):
    base_a = x[:, :, 0]
    base_u = x[:, :, 1]
    base_c = x[:, :, 2]
    base_g = x[:, :, 3]
    batch = base_a.shape[0]
    length = base_a.shape[1]
    au = torch.matmul(base_a.view(batch, length, 1), base_u.view(batch, 1, length))
    au_ua = au + torch.transpose(au, -1, -2)
    cg = torch.matmul(base_c.view(batch, length, 1), base_g.view(batch, 1, length))
    cg_gc = cg + torch.transpose(cg, -1, -2)
    ug = torch.matmul(base_u.view(batch, length, 1), base_g.view(batch, 1, length))
    ug_gu = ug + torch.transpose(ug, -1, -2)
    return au_ua + cg_gc + ug_gu


def constraint_matrix_batch_diag(x, offset=3):
    base_a = x[:, :, 0]
    base_u = x[:, :, 1]
    base_c = x[:, :, 2]
    base_g = x[:, :, 3]
    batch = base_a.shape[0]
    length = base_a.shape[1]
    au = torch.matmul(base_a.view(batch, length, 1), base_u.view(batch, 1, length))
    au_ua = au + torch.transpose(au, -1, -2)
    cg = torch.matmul(base_c.view(batch, length, 1), base_g.view(batch, 1, length))
    cg_gc = cg + torch.transpose(cg, -1, -2)
    ug = torch.matmul(base_u.view(batch, length, 1), base_g.view(batch, 1, length))
    ug_gu = ug + torch.transpose(ug, -1, -2)
    m = au_ua + cg_gc + ug_gu

    # create the band mask
    mask = diags([1]*7, [-3, -2, -1, 0, 1, 2, 3], 
        shape=(m.shape[-2], m.shape[-1])).toarray()
    m = m.masked_fill(torch.Tensor(mask).bool(), 0)
    return m

def contact_map_masks(seq_lens, max_len):
    n_seq = len(seq_lens)
    masks = np.zeros([n_seq, max_len, max_len])
    for i in range(n_seq):
        l = int(seq_lens[i].cpu().numpy())
        masks[i, :l, :l]=1
    return masks

# for test the f1 loss filter
# true_a = torch.Tensor(np.arange(25)).view(5,5).unsqueeze(0)

def f1_loss(pred_a, true_a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_a  = -(F.relu(-pred_a+1)-1)

    true_a = true_a.unsqueeze(1)
    unfold = nn.Unfold(kernel_size=(3, 3), padding=1)
    true_a_tmp = unfold(true_a)
    w = torch.Tensor([0, 0.0, 0, 0.0, 1, 0.0, 0, 0.0, 0]).to(device)
    true_a_tmp = true_a_tmp.transpose(1, 2).matmul(w.view(w.size(0), -1)).transpose(1, 2)
    true_a = true_a_tmp.view(true_a.shape)
    true_a = true_a.squeeze(1)

    tp = pred_a*true_a
    tp = torch.sum(tp, (1,2))

    fp = pred_a*(1-true_a)
    fp = torch.sum(fp, (1,2))

    fn = (1-pred_a)*true_a
    fn = torch.sum(fn, (1,2))

    f1 = torch.div(2*tp, (2*tp + fp + fn))
    return 1-f1.mean()    


def find_pseudoknot(data):
    rnadata1 = data.loc[:,0]
    rnadata2 = data.loc[:,4]
    flag = False
    for i in range(len(rnadata2)):
        for j in range(len(rnadata2)):
            if (rnadata1[i] < rnadata1[j] < rnadata2[i] < rnadata2[j]):
                flag = True
                break
    return flag

# return index of contact pairing, index start from 0
def get_pairings(data):
    rnadata1 = list(data.loc[:,0].values)
    rnadata2 = list(data.loc[:,4].values)
    rna_pairs = list(zip(rnadata1, rnadata2))
    rna_pairs = list(filter(lambda x: x[1]>0, rna_pairs))
    rna_pairs = (np.array(rna_pairs)-1).tolist()
    return rna_pairs


def generate_label_dot_bracket(data):
    rnadata1 = data.loc[:,0]
    rnadata2 = data.loc[:,4]
    rnastructure = []
    for i in range(len(rnadata2)):
        if rnadata2[i] <= 0:
            rnastructure.append(".")
        else:
            if rnadata1[i] > rnadata2[i]:
                rnastructure.append(")")
            else:
                rnastructure.append("(")
    return ''.join(rnastructure)


# extract the pseudoknot index given the data
def extract_pseudoknot(data):
    rnadata1 = data.loc[:,0]
    rnadata2 = data.loc[:,4]
    for i in range(len(rnadata2)):
        for j in range(len(rnadata2)):
            if (rnadata1[i] < rnadata1[j] < rnadata2[i] < rnadata2[j]):
                print(i,j)
                break

def get_pe(seq_lens, max_len):
    num_seq = seq_lens.shape[0]
    pos_i_abs = torch.Tensor(np.arange(1,max_len+1)).view(1, 
        -1, 1).expand(num_seq, -1, -1).double()
    pos_i_rel = torch.Tensor(np.arange(1,max_len+1)).view(1, -1).expand(num_seq, -1)
    pos_i_rel = pos_i_rel.double()/seq_lens.view(-1, 1).double()
    pos_i_rel = pos_i_rel.unsqueeze(-1)
    pos = torch.cat([pos_i_abs, pos_i_rel], -1)

    PE_element_list = list()
    # 1/x, 1/x^2
    PE_element_list.append(pos)
    PE_element_list.append(1.0/pos_i_abs)
    PE_element_list.append(1.0/torch.pow(pos_i_abs, 2))

    # sin(nx)
    for n in range(1, 50):
        PE_element_list.append(torch.sin(n*pos))

    # poly
    for i in range(2, 5):
        PE_element_list.append(torch.pow(pos_i_rel, i))

    for i in range(3):
        gaussian_base = torch.exp(-torch.pow(pos, 
            2))*math.sqrt(math.pow(2,i)/math.factorial(i))*torch.pow(pos, i)
        PE_element_list.append(gaussian_base)

    PE = torch.cat(PE_element_list, -1)
    for i in range(num_seq):
        PE[i, seq_lens[i]:, :] = 0
    return PE

# Random seed
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True






