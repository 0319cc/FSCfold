import time
import torch
from torch.utils import data
import collections
import warnings
import multiprocessing as mp
from Network.Net import FSCfold  # 导入模型结构
from Tool.utils import *  # 导入工具函数
from Tool.data_generator import RNASSDataGenerator  # 数据生成器
from Tool.data_generator import Dataset_new_canonicle as Dataset_FSC  # 数据集定义
from Tool.postprocess import postprocess  # 后处理函数
import numpy as np

warnings.filterwarnings("ignore")  # 忽略不必要的警告

args = get_args()  # 获取命令行参数
RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')  # 定义一个命名元组用于存储RNA序列数据


# 模型测试与评估函数 test
def model_eval_all_test(contact_net, test_generator):
    """
    该函数用于对输入的模型和测试数据生成器进行测试和评估。
    contact_net: 训练好的模型
    test_generator: 测试数据生成器
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设置设备为GPU或CPU
    contact_net.train()  # 设置模型为训练模式（某些情况下影响BatchNorm和Dropout层）

    # 初始化存储结果的列表
    result_no_train = []
    result_nc = []
    seq_names = []
    nc_name_list = []
    seq_lens_list = []
    run_time = []

    # 遍历测试数据集
    for batch_n, (contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name, nc_map, l_len) in enumerate(
            test_generator):
        if batch_n % 100 == 0:  # 每100个批次打印一次批次编号
            print(f'Batch number: {batch_n}')

        # 将数据转移到设备（GPU/CPU）
        contacts_batch = contacts.float().to(device)
        seq_embedding_batch = seq_embeddings.float().to(device)
        seq_ori = seq_ori.float().to(device)   #AUCG

        # 记录序列名和长度
        seq_names.append(seq_name[0])
        seq_lens_list.append(seq_lens.item())

        tik = time.time()  # 记录开始时间

        with torch.no_grad():  # 关闭梯度计算以节省内存和加速
            pred_contacts = contact_net(seq_embedding_batch)  # 通过模型预测接触矩阵

        # 仅后处理（不进行训练）
        u_no_train = postprocess(pred_contacts, seq_ori, 0.01, 0.1, 100, 1.6, True, 1.5)  # 应用后处理函数
        nc_no_train = nc_map.float().to(device) * u_no_train  # 对非标准接触图应用后处理
        map_no_train = (u_no_train > 0.5).float()  # 二值化预测结果
        map_no_train_nc = (nc_no_train > 0.5).float()  # 二值化非标准接触图预测结果

        tok = time.time()  # 记录结束时间
        t0 = tok - tik  # 计算运行时间
        run_time.append(t0)

        # 对每个样本计算准确性指标
        result_no_train += [
            evaluate_exact_new(map_no_train.cpu()[i], contacts_batch.cpu()[i])
            for i in range(contacts_batch.shape[0])
        ]

        # 如果存在非标准接触图，则计算其准确性指标
        if nc_map.float().sum() != 0:
            result_nc += [
                evaluate_exact_new(map_no_train_nc.cpu()[i], nc_map.float().cpu()[i])
                for i in range(contacts_batch.shape[0])
            ]
            nc_name_list.append(seq_name[0])

    print(f'Spend time per batch: {np.mean(run_time):.3f} seconds')  # 打印每个批次的平均运行时间

    # 打印测试集的评估结果
    nt_exact_p, nt_exact_r, nt_exact_f1 = zip(*result_no_train)
    print('Average testing F1 score with pure post-processing: ', np.average(nt_exact_f1))
    print('Average testing precision with pure post-processing: ', np.average(nt_exact_p))
    print('Average testing recall with pure post-processing: ', np.average(nt_exact_r))


# 主函数，执行模型加载和测试
def main():
    torch.multiprocessing.set_sharing_strategy('file_system')  # 设置多进程共享策略为文件系统

    test_file = args.test_files  # 获取测试文件路径

    # 根据测试文件选择相应的模型权重文件
    if test_file == 'TS0':
        MODEL_SAVED = 'Models/TS0.pt'
    elif test_file == 'bpnew':
        MODEL_SAVED = 'Models/bpnew.pt'
    else:
        MODEL_SAVED = 'Models/RNAStralign.pt'

    print(MODEL_SAVED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设置设备为GPU或CPU
    seed_torch()  # 设置随机种子以确保结果可复现

    print(f'Loading test file: {test_file}')  # 打印加载的测试文件名
    if test_file in ['all_600', 'ArchiveII', 'RNAStralign']:
        test_data = RNASSDataGenerator('Dataset/', f'{test_file}.pickle')  # 加载pickle格式的测试数据
    else:
        test_data = RNASSDataGenerator('Dataset/', f'{test_file}.cPickle')  # 加载cPickle格式的测试数据

    seq_len = test_data.data_y.shape[-2]  # 获取序列最大长度
    print(f'Max seq length: {seq_len}')

    # 定义测试数据生成器的参数
    params = {
        'batch_size': 1,
        'shuffle': True,
        'num_workers': 12,
        'drop_last': True
    }

    test_set = Dataset_FSC(test_data)  # 初始化测试数据集
    test_generator = data.DataLoader(test_set, **params)  # 使用Dataloader生成批次数据

    contact_net = FSCfold()  # 初始化模型
    contact_net.load_state_dict(torch.load(MODEL_SAVED, map_location=device))  # 加载预训练模型权重
    print('==========Finish Loading==========')
    contact_net.to(device)  # 将模型转移到设备（GPU/CPU）

    model_eval_all_test(contact_net, test_generator)  # 调用模型测试函数


# 设置多进程启动方式并执行主函数
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
