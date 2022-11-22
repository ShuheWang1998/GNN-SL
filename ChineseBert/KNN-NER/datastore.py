import os
from time import sleep
import numpy as np
import json
import torch
import torch.nn as nn
import dgl
import dgl.function as fn

class Datastore():

    def __init__(self, datastore_dir, is_gpu=True, add_label=False):
        self.is_gpu = is_gpu
        self.add_label = add_label
        self.datastore_info = json.load(open(os.path.join(datastore_dir, "datastore_info.json"), "r"))
        token_num = self.datastore_info["token_sum"]
        self.l_size = self.datastore_info["l_size"]
        self.r_size = self.datastore_info["r_size"]
        hidden_size = self.datastore_info["hidden_size"]
        mid_size = self.l_size + self.r_size + 1
        self.keys = np.memmap(os.path.join(datastore_dir, "datastore_features.npy"), 
                              dtype=np.float32,
                              mode="r",
                              shape=(token_num, mid_size, hidden_size))
        self.mid_keys = self.keys[:, mid_size//2, :]
        self.values = np.memmap(os.path.join(datastore_dir, "datastore_vals.npy"), 
                                dtype=np.int32,
                                mode="r",
                                shape=(token_num, mid_size))
        self.keys = torch.from_numpy(self.keys)
        #self.mid_keys = torch.from_numpy(self.mid_keys)
        #if self.add_label:
        #    self.values = torch.from_numpy(self.values)
        self.values = torch.from_numpy(self.values)

        if self.is_gpu:
            self.keys = self.keys.cuda() # token_num * (l_size + r_size + 1) * hidden_size
            #self.mid_keys = self.mid_keys.cuda() # token_num * hidden_size
            if self.add_label:
               self.values = self.values.cuda() # token_num * (l_size + r_size + 1)
            # self.values = self.values.cuda() # token_num * (l_size + r_size + 1)

    #def find_knn(self, query_tensor, knn_seach_k):
    #    # batch_size * seq_len * token_num
    #    cosine_similarity = nn.functional.cosine_similarity(query_tensor.unsqueeze(dim=2), self.mid_keys.unsqueeze(dim=0).unsqueeze(dim=0))
    #    top_k_value, top_k_index = torch.topk(cosine_similarity, k=knn_seach_k, dim=-1) # batch_size * seq_len * search_k
    #    return top_k_index, top_k_value

    def build_graph(self, source: torch.Tensor, neighbours: torch.Tensor, real_seq_len: torch.Tensor, knn_search_k):
        """
        build dgl graph
        Args:
            source: source tensor
            offsets: np.array
            neighbor_idxs: np.array of shape [len(source), self.k]
            target: target tensor shifted left by 1
        """
        '''
        def get_inter_edge(source, nums):
            l_size = self.datastore_info["l_size"]
            r_size = self.datastore_info["r_size"]
            mid_size = l_size + r_size + 1
            half_mid_size = mid_size // 2
            return [[idx_*(mid_size)+half_mid_size for idx_ in range(nums)], [source]*nums]
        '''
        def get_inter_edge(now_real_seq_len, nums):
            tmp_inter_edge = [[], []]
            now_cnt = 0
            for source in range(now_real_seq_len):
                for _ in range(nums):
                    tmp_inter_edge[0].append(now_cnt*3+1)
                    tmp_inter_edge[1].append(source)
                    now_cnt += 1
            return tmp_inter_edge

        def get_ntgt_intra_edge(nums):
            tmp_intra_edge = [[], []]
            for idx_ in range(nums):
                tmp_intra_edge[0].append(idx_*3)
                tmp_intra_edge[1].append(idx_*3+1)
                tmp_intra_edge[0].append(idx_*3+1)
                tmp_intra_edge[1].append(idx_*3)
                tmp_intra_edge[0].append(idx_*3+2)
                tmp_intra_edge[1].append(idx_*3+1)
                tmp_intra_edge[0].append(idx_*3+1)
                tmp_intra_edge[1].append(idx_*3+2)
            return tmp_intra_edge
        
        def get_tgt_intra_edge(nums):
            tmp_intra_edge = [[], []]
            for i in range(nums):
                for j in range(i + 1, nums):
                    tmp_intra_edge[0].append(i)
                    tmp_intra_edge[1].append(j)
                    tmp_intra_edge[0].append(j)
                    tmp_intra_edge[1].append(i)
            return tmp_intra_edge
        
        def get_label_edge(nums):
            tmp_label_edge = [[], []]
            for idx_ in range(nums):
                for span_ in range(3):
                    tmp_label_edge[0].append(idx_*3+span_)
                    tmp_label_edge[1].append(idx_*3+span_)
            return tmp_label_edge

        graphs = []
        # batch_size * seq_len * search_k
        batch_size = neighbours.shape[0]

        for batch_size_idx in range(batch_size):
            now_real_seq_len = real_seq_len[batch_size_idx].item()
            tgt_features = source[batch_size_idx, :now_real_seq_len,]
            tgt2tgt = get_tgt_intra_edge(now_real_seq_len)
            ntgt2tgt = get_inter_edge(now_real_seq_len, knn_search_k)
            ntgt2ntgt = get_ntgt_intra_edge(knn_search_k*now_real_seq_len)
            
            if self.add_label:
                label2ntgt = get_label_edge(knn_search_k*now_real_seq_len)

            ntgt_features = []
            graph_labels = []
            for seq_len_idx in range(now_real_seq_len):
                ntgt_features.append(self.keys[neighbours[batch_size_idx][seq_len_idx]].view(knn_search_k*3, -1))
                # if self.add_label:
                #    graph_labels.append(self.values[neighbours[batch_size_idx][seq_len_idx]].view(-1))
                graph_labels.append(self.values[neighbours[batch_size_idx][seq_len_idx]].view(-1))
            ntgt_features = torch.cat(ntgt_features, dim=0)
            if self.add_label:
                graph_labels = torch.cat(graph_labels, dim=0)
                graph = dgl.heterograph({
                        ('tgt', 'intra', 'tgt'): (torch.LongTensor(tgt2tgt[0]), torch.LongTensor(tgt2tgt[1])),
                        ('ntgt', 'inter', 'tgt'): (torch.LongTensor(ntgt2tgt[0]), torch.LongTensor(ntgt2tgt[1])),
                        ('tgt', 'inter', 'ntgt'): (torch.LongTensor(ntgt2tgt[1]), torch.LongTensor(ntgt2tgt[0])),
                        ('ntgt', 'intra', 'ntgt'): (torch.LongTensor(ntgt2ntgt[0]), torch.LongTensor(ntgt2ntgt[1])),
                        ('label', 'label_edge', 'ntgt'): (torch.LongTensor(label2ntgt[0]), torch.LongTensor(label2ntgt[1])),
                        ('ntgt', 'label_edge', 'label'): (torch.LongTensor(label2ntgt[1]), torch.LongTensor(label2ntgt[0])),
                        #('ntgt', 'label_edge', 'label'): (torch.LongTensor(label2ntgt[0]), torch.LongTensor(label2ntgt[1])),
                        ('label', 'intra', 'label'): (torch.LongTensor(ntgt2ntgt[0]), torch.LongTensor(ntgt2ntgt[1])),
                    }).to(tgt_features.device)
                #graph.nodes["label"].data["h"] = graph_labels
            else:
                graph_labels = torch.cat(graph_labels, dim=0)
                graph = dgl.heterograph({
                            ('tgt', 'intra', 'tgt'): (torch.LongTensor(tgt2tgt[0]), torch.LongTensor(tgt2tgt[1])),
                            ('ntgt', 'inter', 'tgt'): (torch.LongTensor(ntgt2tgt[0]), torch.LongTensor(ntgt2tgt[1])),
                            ('tgt', 'inter', 'ntgt'): (torch.LongTensor(ntgt2tgt[1]), torch.LongTensor(ntgt2tgt[0])),
                            ('ntgt', 'intra', 'ntgt'): (torch.LongTensor(ntgt2ntgt[0]), torch.LongTensor(ntgt2ntgt[1])),
                        }).to(tgt_features.device)
            graph.nodes["tgt"].data["h"] = tgt_features
            graph.nodes["ntgt"].data["h"] = ntgt_features
            if not self.add_label:
                graphs.append((batch_size_idx, now_real_seq_len, graph, graph_labels))
            else:
                graphs.append((batch_size_idx, now_real_seq_len, graph, graph_labels))              
        return graphs