import time
import torch
from torch.utils import data
import collections
import warnings
import multiprocessing as mp
from Network.Net import FSCfold
from Tool.utils import *
from Tool.data_generator import RNASSDataGenerator
from Tool.data_generator import Dataset_new_canonicle as Dataset_FSC
from Tool.postprocess import postprocess
import numpy as np


warnings.filterwarnings("ignore")

args = get_args()
RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')

# Model testing and evaluation
def model_eval_all_test(contact_net, test_generator):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")     # use
    contact_net.train()
    result_no_train = []
    result_nc = []
    seq_names = []
    nc_name_list = []
    seq_lens_list = []
    run_time = []

    for batch_n, (contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name, nc_map, l_len) in enumerate(
            test_generator):

        if batch_n % 100 == 0:
            print(f'Batch number: {batch_n}')

        contacts_batch = contacts.float().to(device)
        seq_embedding_batch = seq_embeddings.float().to(device)
        seq_ori = seq_ori.float().to(device)
        seq_names.append(seq_name[0])
        seq_lens_list.append(seq_lens.item())
        tik = time.time()

        with torch.no_grad():
            pred_contacts = contact_net(seq_embedding_batch)

        # Only post-processing without learning
        u_no_train = postprocess(pred_contacts, seq_ori, 0.01, 0.1, 100, 1.6, True, 1.5)
        nc_no_train = nc_map.float().to(device) * u_no_train
        map_no_train = (u_no_train > 0.5).float()
        map_no_train_nc = (nc_no_train > 0.5).float()

        tok = time.time()
        t0 = tok - tik
        run_time.append(t0)

        result_no_train += [
            evaluate_exact_new(map_no_train.cpu()[i], contacts_batch.cpu()[i])
            for i in range(contacts_batch.shape[0])
        ]

        if nc_map.float().sum() != 0:
            result_nc += [
                evaluate_exact_new(map_no_train_nc.cpu()[i], nc_map.float().cpu()[i])
                for i in range(contacts_batch.shape[0])
            ]
            nc_name_list.append(seq_name[0])

    print(f'Spend time per batch: {np.mean(run_time):.3f} seconds')

    nt_exact_p, nt_exact_r, nt_exact_f1 = zip(*result_no_train)
    print('Average testing F1 score with pure post-processing: ', np.average(nt_exact_f1))
    print('Average testing precision with pure post-processing: ', np.average(nt_exact_p))
    print('Average testing recall with pure post-processing: ', np.average(nt_exact_r))


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')

    test_file = args.test_files

    if test_file == 'TS0':
        MODEL_SAVED = 'Models/TS0.pt'
    elif test_file == 'bpnew':
        MODEL_SAVED = 'Models/bpnew.pt'
    else:
        MODEL_SAVED = 'Models/RNAStralign.pt'

    print(MODEL_SAVED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_torch()

    print(f'Loading test file: {test_file}')
    if test_file in ['all_600', 'ArchiveII', 'RNAStralign']:
        test_data = RNASSDataGenerator('Dataset/', f'{test_file}.pickle')
    else:
        test_data = RNASSDataGenerator('Dataset/', f'{test_file}.cPickle')

    seq_len = test_data.data_y.shape[-2]
    print(f'Max seq length: {seq_len}')

    params = {
        'batch_size': 1,
        'shuffle': True,
        'num_workers': 12,
        'drop_last': True
    }

    test_set = Dataset_FSC(test_data)
    test_generator = data.DataLoader(test_set, **params)

    contact_net = FSCfold()
    contact_net.load_state_dict(torch.load(MODEL_SAVED, map_location=device))
    print('==========Finish Loading==========')
    contact_net.to(device)
    model_eval_all_test(contact_net, test_generator)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()





