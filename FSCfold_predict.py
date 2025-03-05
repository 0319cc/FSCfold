import torch
from torch.utils import data
from Network.Net import FSCfold
from Tool.utils import seed_torch
from Tool.data_generator import RNASSDataGenerator_input
from Tool.data_generator import Dataset_new as Dataset_FCN
from Tool.postprocess import postprocess
import collections
import warnings
warnings.filterwarnings("ignore")
RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')
def get_seq(seq_ori: torch.Tensor) -> str:
    seq_map = {0: 'A', 1: 'U', 2: 'C', 3: 'G'}
    seq = ''
    for i in seq_ori:
        if i.sum() == 0:
            seq += 'N'
        else:
            for j, k in enumerate(i):
                if k == 1:
                    seq += seq_map.get(j, 'N')
    return seq

def getbpseq_predict(mapping: dict, true_seq: str, seq_name: str, seq_len: int) -> None:
    file_path = f'./Results/save_ct_file/{seq_name}.ct'
    with open(file_path, 'w') as f:
        for i in range(seq_len):
            f.write(f"{i + 1}   {true_seq[i]}   {mapping.get(i + 1, 0)}\n")

def getmatch(map_no_train: torch.Tensor, seq_len: int) -> dict:
    mapping = {i + 1: 0 for i in range(seq_len)}
    for i in range(seq_len):
        if map_no_train[i].sum() == 1:
            for j in range(seq_len):
                if map_no_train[i][j] == 1:
                    mapping[i + 1] = j + 1
    return mapping

def model_eval_all_test(contact_net: torch.nn.Module, test_generator: data.DataLoader) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    contact_net.train()
    batch_n = 0
    seq_names = []
    errors = []

    for seq_embeddings, seq_len, seq_ori, seq_name in test_generator:
        if batch_n % 10 == 0:
            print(f'Sequencing number: {batch_n}')
        batch_n += 1
        seq_embedding_batch = seq_embeddings.float().to(device)
        seq_ori = seq_ori.float().to(device)
        seq_names.append(seq_name[0])

        with torch.no_grad():
            pred_contacts = contact_net(seq_embedding_batch)

        u_no_train = postprocess(pred_contacts, seq_ori, 0.01, 0.1, 100, 1.6, True, 1.5)
        map_no_train = (u_no_train > 0.5).float()

        mapping = getmatch(map_no_train[0], seq_len)
        true_seq = get_seq(seq_ori[0])

        try:
            getbpseq_predict(mapping, true_seq, seq_name[0], seq_len.item())
        except Exception as e:
            errors.append(seq_name[0])
            print(f"Error processing sequence {seq_name[0]}: {e}")

def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.cuda.set_device(0)

    MODEL_SAVED = 'Models/RNAStralign.pt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed_torch()

    params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 16,
        'drop_last': True
    }

    predict_data = RNASSDataGenerator_input('Dataset/', 'input')
    test_set = Dataset_FCN(predict_data)
    test_generator = data.DataLoader(test_set, **params, multiprocessing_context='spawn')
    contact_net = FSCfold()
    print('==========Start Loading Pretrained Model==========')
    contact_net.load_state_dict(torch.load(MODEL_SAVED, map_location=device))
    contact_net.to(device)
    print('==========Finish Loading Pretrained Model==========')
    model_eval_all_test(contact_net, test_generator)
    print('==========Done!!! Please check results folder for the predictions!==========')

if __name__ == '__main__':
    print('Welcome using FSCFold to predict!')
    main()
