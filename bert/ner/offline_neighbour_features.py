import os
import numpy as np
import json
import torch
from tqdm import tqdm
import argparse


def get_features(feature_info_path, label_path, nei_path, prefix, datastore_dir, search_knn_k, batch_size):
    feature_info = json.load(open(os.path.join(feature_info_path, f"{prefix}_feature_info.json"), "r"))
    seq_num = feature_info["seq_num"]
    max_seq_len = feature_info["max_seq_len"]
    hidden_size = feature_info["hidden_size"]
    labels = np.memmap(os.path.join(label_path, f"{prefix}_labels.npy"), 
                       dtype=np.int32,
                       mode="r",
                       shape=(seq_num, max_seq_len))
    neighbours = np.memmap(os.path.join(nei_path, f"{prefix}_neighbour_idx.npy"), 
                           dtype=np.int32,
                           mode="r",
                           shape=(seq_num, max_seq_len, search_knn_k))
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
    values = np.memmap(os.path.join(datastore_dir, "datastore_vals.npy"), 
                       dtype=np.int32,
                       mode="r",
                       shape=(token_num, mid_size))

    keys = torch.from_numpy(keys).cuda()
    values = torch.from_numpy(values).cuda()
    
    mmap_ntgt_features = np.memmap(os.path.join(nei_path, f"{prefix}_ntgt_features.npy"), 
                                   dtype=np.float32,
                                   mode="w+",
                                   shape=(seq_num, max_seq_len*search_knn_k*mid_size, hidden_size))
    mmap_ntgt_labels = np.memmap(os.path.join(nei_path, f"{prefix}_ntgt_labels.npy"), 
                                   dtype=np.int32,
                                   mode="w+",
                                   shape=(seq_num, max_seq_len*search_knn_k*mid_size))

    pbar = tqdm(total=seq_num)
    start_ = 0
    while start_ < seq_num:
        end_ = min(start_+batch_size, seq_num)
        batch_labels = torch.from_numpy(labels[start_:end_]).cuda()
        batch_neighbours = torch.from_numpy(neighbours[start_:end_]).cuda()
        real_seq_len = (batch_labels != -1).long().sum(dim=-1).cpu()
        for idx_ in range(end_-start_):
            now_seq_len = real_seq_len[idx_]
            ntgt_features = []
            ntgt_labels = []
            for seq_len_idx in range(now_seq_len):
                ntgt_features.append(keys[batch_neighbours[idx_][seq_len_idx]].view(search_knn_k*mid_size, -1))
                ntgt_labels.append(values[batch_neighbours[idx_][seq_len_idx]].view(-1))
            ntgt_features = torch.cat(ntgt_features, dim=0)
            ntgt_labels = torch.cat(ntgt_labels, dim=0)
            mmap_ntgt_features[start_+idx_, :ntgt_features.shape[0], :] = ntgt_features.cpu()[:]
            mmap_ntgt_labels[start_+idx_, :ntgt_labels.shape[0]] = ntgt_labels.cpu()[:]
        pbar.update(end_-start_)
        start_ = end_
    


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--feature_info_path", type=str, help="")
    parser.add_argument("--label_path", type=str, help="")
    parser.add_argument("--nei_path", type=str, help="")
    parser.add_argument("--prefix", type=str, help="")
    parser.add_argument("--datastore_dir", type=str, help="")
    parser.add_argument("--search_knn_k", default=32, type=int, help="use gpu for knn saerch")
    parser.add_argument("--batch_size", default=32, type=int, help="used for knn search in the process of gnn")

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    get_features(feature_info_path=args.feature_info_path,
                 label_path=args.label_path,
                 nei_path=args.nei_path,
                 prefix=args.prefix,
                 datastore_dir=args.datastore_dir,
                 search_knn_k=args.search_knn_k,
                 batch_size=args.batch_size)

if __name__ == '__main__':
    main()