import time
import torch
import torch.optim as optim
from torch.utils import data
from Network.Net import FSCfold
from Tool.utils import get_args, seed_torch
from Tool.config import process_config
from Tool.data_generator import RNASSDataGenerator, Dataset_new_merge_multi as Dataset_FCN_merge
import collections

import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

args = get_args()
RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')

# 查看模型参数
def view_model_param():
    model = FSCfold()
    total_param = sum(np.prod(list(param.data.size())) for param in model.parameters())
    print("MODEL DETAILS:\n")
    print(model)
    print('MODEL/Total parameters:', total_param)

# 训练
def train(contact_net, train_merge_generator):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_weight = torch.Tensor([300]).to(device)
    criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    u_optimizer = optim.Adam(contact_net.parameters())
    print('start training...')
    start_epoch = 0
    epoch_last = 100
    for epoch in range(start_epoch, epoch_last):
        torch.cuda.empty_cache()  # 在每个epoch开始前释放缓存
        contact_net.train()
        steps_done = 0
        start = time.time()

        for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name in train_merge_generator:
            contacts_batch = contacts.float().to(device)
            seq_embedding_batch = seq_embeddings.float().to(device)
            pred_contacts = contact_net(seq_embedding_batch)

            contact_masks = torch.zeros_like(pred_contacts)
            contact_masks[:, :seq_lens, :seq_lens] = 1

            # Compute loss
            loss_u = criterion_bce_weighted(pred_contacts * contact_masks, contacts_batch)

            # Optimize the model
            u_optimizer.zero_grad()
            loss_u.backward()
            u_optimizer.step()
            steps_done += 1

        print(f'Epoch: {epoch}, Steps: {steps_done}, Time: {time.time() - start:.2f}s, Loss: {loss_u.item()}')
        torch.save(contact_net.state_dict(), f'Models/RNAStralign_{epoch}.pt')

def main():

    config_file = args.config
    config = process_config(config_file)
    print(config)

    view_model_param()
    # os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    train_files = args.train_files

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch()

    train_data_list = []
    for file_item in train_files:
        print(f'Loading dataset: {file_item}')
        if file_item in ['RNAStralign', 'ArchiveII', 'train']:
            train_data_list.append(RNASSDataGenerator('Dataset/', f'{file_item}.pickle'))
        else:
            train_data_list.append(RNASSDataGenerator('Dataset/', f'{file_item}.cPickle'))
    print('Data Loading Done!!!')

    params = {
        'batch_size': 1,
        'shuffle': True,
        'num_workers': 16,
        'drop_last': True
    }
    train_merge = Dataset_FCN_merge(train_data_list)
    train_merge_generator = data.DataLoader(train_merge, **params, multiprocessing_context='spawn')

    contact_net = FSCfold()
    contact_net.to(device)

    # checkpoint_path = 'Models/RNAStralign.pt'  # 修改为最后的保存点文件
    # contact_net.load_state_dict(torch.load(checkpoint_path))

    train(contact_net, train_merge_generator)

if __name__ == '__main__':
    main()
