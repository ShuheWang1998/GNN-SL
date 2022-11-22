import torch
import numpy as np
from torch.utils.data import Dataset
import json


class GraphDataset(Dataset):
    def __init__(self, feature_info_path, feature_path, label_path, mask_path, nei_path, search_knn_k, datastore, add_knn=False):
        self.feature_info = json.load(open(feature_info_path, "r"))
        seq_num = self.feature_info["seq_num"]
        max_seq_len = self.feature_info["max_seq_len"]
        hidden_size = self.feature_info["hidden_size"]
        self.features = np.memmap(feature_path, 
                                  dtype=np.float32,
                                  mode="r",
                                  shape=(seq_num, max_seq_len, hidden_size))
        self.labels = np.memmap(label_path, 
                                dtype=np.int32,
                                mode="r",
                                shape=(seq_num, max_seq_len))
        self.sequence_mask = np.memmap(mask_path, 
                                dtype=np.int32,
                                mode="r",
                                shape=(seq_num, max_seq_len))
        self.neighbours = np.memmap(nei_path, 
                                    dtype=np.int32,
                                    mode="r",
                                    shape=(seq_num, max_seq_len, search_knn_k))
        self.neighbours_values = np.memmap(nei_path+".values", 
                                    dtype=np.float32,
                                    mode="r",
                                    shape=(seq_num, max_seq_len, search_knn_k))
        
        self.add_knn = add_knn
        self.datastore = datastore

    def __len__(self):
        return self.feature_info["seq_num"]

    def __getitem__(self, index):
        #print("=============")
        #print(self.labels[index])
        #print(self.sequence_mask[index])
        #print("=============")
        if not self.add_knn:
            return torch.Tensor(self.features[index]), torch.LongTensor(self.neighbours[index]), torch.LongTensor(self.labels[index]), torch.BoolTensor(self.sequence_mask[index])
        else:
            return torch.Tensor(self.features[index]), torch.LongTensor(self.neighbours[index]), torch.LongTensor(self.labels[index]), torch.BoolTensor(self.sequence_mask[index]), torch.FloatTensor(self.neighbours_values[index])