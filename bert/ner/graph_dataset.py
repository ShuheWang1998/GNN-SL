import torch
import numpy as np
from torch.utils.data import Dataset
import json
import os


class GraphDataset(Dataset):
    def __init__(self, feature_info_path, feature_path, label_path, nei_path, search_knn_k, datastore, add_knn=False):
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
        self.neighbours = np.memmap(nei_path, 
                                    dtype=np.int32,
                                    mode="r",
                                    shape=(seq_num, max_seq_len, search_knn_k))
        self.neighbours_values = np.memmap(nei_path+".values", 
                                    dtype=np.float32,
                                    mode="r",
                                    shape=(seq_num, max_seq_len, search_knn_k))
        
        #self.features = torch.from_numpy(self.features)
        #self.labels = torch.from_numpy(self.labels)
        #self.neighbours = torch.from_numpy(self.neighbours)
        #self.neighbours_values = torch.from_numpy(self.neighbours_values)

        self.add_knn = add_knn
        self.datastore = datastore

    def __len__(self):
        return self.feature_info["seq_num"]

    def __getitem__(self, index):
        if not self.add_knn:
            return torch.Tensor(self.features[index]), torch.LongTensor(self.neighbours[index]), torch.LongTensor(self.labels[index])
            #return self.features[index], self.neighbours[index].long(), self.labels[index].long()
        else:
            return torch.Tensor(self.features[index]), torch.LongTensor(self.neighbours[index]), torch.LongTensor(self.labels[index]), torch.FloatTensor(self.neighbours_values[index])