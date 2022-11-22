import os
import torch
import torch.nn as nn
import numpy as np
import json
from tqdm import tqdm
import argparse


def search_knn(query_tensor, datastore, search_knn_k=32):
    # cosine_similarity
    #cosine_similarity = nn.functional.cosine_similarity(query_tensor, datastore, dim=-1)
    #top_k_value, top_k_index = torch.topk(cosine_similarity, k=search_knn_k, dim=-1) # search_k
    # dot_similarity
    dot_similarity = torch.mul(query_tensor, datastore).sum(dim=-1)
    top_k_value, top_k_index = torch.topk(dot_similarity, k=search_knn_k, dim=-1) # search_k
    return top_k_index, top_k_value

#def search_knn(query_tensor, datastore, token_labels, search_knn_k=32, dividen_k=2, is_train=False):
#    # cosine_similarity
#    #cosine_similarity = nn.functional.cosine_similarity(query_tensor, datastore, dim=-1)
#    #top_k_value, top_k_index = torch.topk(cosine_similarity, k=search_knn_k, dim=-1) # search_k
#    # dot_similarity
#    # real_batch_size * token_num
#    dot_similarity = torch.mul(query_tensor, datastore).sum(dim=-1)
#    values_, indices_ = torch.topk(dot_similarity, k=search_knn_k*search_knn_k, dim=-1) # search_k
#    batch_size, token_num = indices_.shape
#    top_k_index = torch.zeros_like(indices_[:, :search_knn_k])
#    top_k_value = torch.zeros_like(values_[:, :search_knn_k])
#    max_len = torch.max(token_labels) + 1
#    for batch_idx in range(batch_size):
#        count_ = torch.zeros(max_len, dtype=torch.int32).cuda()
#        count_[token_labels[indices_[batch_idx][0]]] += 1
#        search_cnt = 1
#        top_k_index[batch_idx][0] = indices_[batch_idx][0]
#        top_k_value[batch_idx][0] = values_[batch_idx][0]
#        if is_train:
#            top_k_index[batch_idx][1] = indices_[batch_idx][0]
#            top_k_value[batch_idx][1] = values_[batch_idx][0]
#            search_cnt += 1
#        for idx_ in range(1, token_num):
#            if count_[token_labels[indices_[batch_idx][idx_]]] < dividen_k:
#                top_k_index[batch_idx][search_cnt] = indices_[batch_idx][idx_]
#                top_k_value[batch_idx][search_cnt] = values_[batch_idx][idx_]
#                count_[token_labels[indices_[batch_idx][idx_]]] += 1
#                search_cnt += 1
#                if search_cnt >= search_knn_k:
#                    break
#            
#    return top_k_index, top_k_value

def offline_knn_saerch(feature_info_path, feature_path, label_path, datastore_dir, save_path, prefix="train", search_knn_k=32, is_gpu=False, batch_size=1):
    feature_info = json.load(open(os.path.join(feature_info_path, f"{prefix}_feature_info.json"), "r"))
    seq_num = feature_info["seq_num"]
    max_seq_len = feature_info["max_seq_len"]
    hidden_size = feature_info["hidden_size"]
    features = np.memmap(os.path.join(feature_path, f"{prefix}_features.npy"), 
                         dtype=np.float32,
                         mode="r",
                         shape=(seq_num, max_seq_len, hidden_size))
    labels = np.memmap(os.path.join(label_path, f"{prefix}_labels.npy"), 
                       dtype=np.int32,
                       mode="r",
                       shape=(seq_num, max_seq_len))
    
    datastore_info = json.load(open(os.path.join(datastore_dir, "datastore_info.json"), "r"))
    token_num = datastore_info["token_sum"]
    l_size = datastore_info["l_size"]
    r_size = datastore_info["r_size"]
    hidden_size = datastore_info["hidden_size"]
    mid_size = l_size + r_size + 1
    keys = np.memmap(os.path.join(datastore_dir, "datastore_features.npy"), 
                     dtype=np.float32,
                     mode="r",
                     shape=(token_num, mid_size, hidden_size))
    mid_keys = keys[:, mid_size//2, :]
    values = np.memmap(os.path.join(datastore_dir, "datastore_vals.npy"), 
                       dtype=np.int32,
                       mode="r",
                       shape=(token_num, mid_size))
    mid_values = values[:, mid_size//2]
    
    search_reuslt_in_memory = np.zeros((seq_num, max_seq_len, search_knn_k), dtype=np.int32)
    search_reuslt = np.memmap(save_path, 
                              dtype=np.int32,
                              mode="w+",
                              shape=(seq_num, max_seq_len, search_knn_k))
    
    search_reuslt_value_in_memory = np.zeros((seq_num, max_seq_len, search_knn_k), dtype=np.float32)
    search_reuslt_value = np.memmap(save_path+'.values', 
                              dtype=np.float32,
                              mode="w+",
                              shape=(seq_num, max_seq_len, search_knn_k))
    '''
    tensor_features = torch.from_numpy(features)
    tensor_labels = torch.from_numpy(labels)
    tensor_datastore = torch.from_numpy(mid_keys)
    if is_gpu:
        tensor_features = tensor_features.cuda()
        tensor_labels = tensor_labels.cuda()
        tensor_datastore = tensor_datastore.cuda()
    
    real_seq_lens = (tensor_labels != -1).long().sum(dim=-1)
    '''
    tensor_datastore = torch.from_numpy(mid_keys)
    tensor_datastore_values = torch.from_numpy(mid_values)
    if is_gpu:
        tensor_datastore = tensor_datastore.cuda()
        tensor_datastore_values = tensor_datastore_values.cuda().view(-1) # token_num

    for seq_idx in tqdm(range(seq_num)):
        tensor_features = torch.from_numpy(features[seq_idx])
        tensor_labels = torch.from_numpy(labels[seq_idx])
        if is_gpu:
            tensor_features = tensor_features.cuda()
            tensor_labels = tensor_labels.cuda()
        now_real_seq_len = (tensor_labels != -1).long().sum(dim=-1).item()
        start_ = 0
        while (start_ < now_real_seq_len):
            flag = False
            real_search_knn_k = search_knn_k
            if prefix == 'train' and start_ != 0 and start_ != now_real_seq_len - 1:
                flag = True
            if flag:
                real_search_knn_k += 1
            end_ = min(start_+batch_size, now_real_seq_len)
            real_batch_size = end_ - start_
            # real_batch_size * token_num * hidden_size --> real_batch_size * search_knn_k
            top_k_index, top_k_value = search_knn(tensor_features[start_:end_, ].unsqueeze(dim=1).expand(-1, token_num, hidden_size), tensor_datastore.unsqueeze(dim=0).expand(real_batch_size, token_num, hidden_size), search_knn_k=real_search_knn_k)
            #top_k_index, top_k_value = search_knn(tensor_features[start_:end_, ].unsqueeze(dim=1).expand(-1, token_num, hidden_size), tensor_datastore.unsqueeze(dim=0).expand(real_batch_size, token_num, hidden_size), token_labels=tensor_datastore_values, search_knn_k=real_search_knn_k)
            #top_k_index, top_k_value = search_knn(tensor_features[start_:end_, ].unsqueeze(dim=1).expand(-1, token_num, hidden_size), tensor_datastore.unsqueeze(dim=0).expand(real_batch_size, token_num, hidden_size), token_labels=tensor_datastore_values, search_knn_k=real_search_knn_k)
            if flag:
                search_reuslt_in_memory[seq_idx, start_:end_, :] = top_k_index.cpu().numpy()[:, 1:]
                search_reuslt_value_in_memory[seq_idx, start_:end_, :] = top_k_value.cpu().numpy()[:, 1:]
            else:
                search_reuslt_in_memory[seq_idx, start_:end_, :] = top_k_index.cpu().numpy()[:]
                search_reuslt_value_in_memory[seq_idx, start_:end_, :] = top_k_value.cpu().numpy()[:]       
            start_ = end_
        '''
        for real_seq_len_idx in range(now_real_seq_len):
            top_k_index, top_k_value = search_knn(tensor_features[seq_idx][real_seq_len_idx], tensor_datastore, search_knn_k=search_knn_k)
            search_reuslt_in_memory[seq_idx][real_seq_len_idx] = top_k_index.cpu().numpy()
        '''
    
    search_reuslt[:] = search_reuslt_in_memory[:]
    search_reuslt_value[:] = search_reuslt_value_in_memory[:]

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, help="train data path")
    parser.add_argument("--datastore_dir", type=str, help="dir for datastore.")
    parser.add_argument("--save_path", type=str, help="train data path")
    parser.add_argument("--prefix", type=str, help="train/dev/test")
    parser.add_argument("--is_gpu", action="store_true", help="use gpu for knn saerch")
    parser.add_argument("--gnn_k", default=32, type=int, help="used for knn search in the process of gnn")
    parser.add_argument("--batch_size", default=1, type=int, help="batch_size used for knn search in the process of gnn")

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    offline_knn_saerch(feature_info_path=args.data_dir, 
                       feature_path=args.data_dir, 
                       label_path=args.data_dir, 
                       datastore_dir=args.datastore_dir, 
                       save_path=args.save_path, prefix=args.prefix, 
                       search_knn_k=args.gnn_k, is_gpu=args.is_gpu,
                       batch_size=args.batch_size)

if __name__ == '__main__':
    main()


'''
Dutch S-MISC 
forward O 
Reggie B-PER 
Blinker E-PER 
had O 
his O 
indefinite O 
suspension O 
lifted O 
by O 
FIFA S-ORG 
on O 
Friday O 
and O 
was O 
set O 
to O 
make O 
his O 
Sheffield B-ORG 
Wednesday E-ORG 
comeback O 
against O 
Liverpool S-ORG 
on O 
Saturday O 
. O
'''