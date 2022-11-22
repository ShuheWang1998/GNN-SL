#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file  : pos_dataset.py
@author: shuhe wang
@contact : shuhe_wang@shannonai.com
@date  : 2021/07/08 20:09
@version: 1.0
@desc  :
"""

import os

import torch
from torch.utils.data import Dataset
from tokenizers import BertWordPieceTokenizer


class CWSDataset(Dataset):

    __ner_labels = None

    """the Dataset Class for NER task."""
    def __init__(self, directory, prefix, vocab_file, max_length=512, file_name="char.bmes"):
        """
        Args:
            directory: str, path to data directory.
            prefix: str, one of [train/dev/test]
            vocab_file: str, path to the vocab file for model pre-training.
            max_length: int,
        """
        super().__init__()
        self.max_length = max_length
        data_file_path = os.path.join(directory, "{}.{}".format(prefix, file_name))
        self.data_items = CWSDataset._read_cws(data_file_path)
        self.tokenizer = BertWordPieceTokenizer(vocab_file)
        self.label_to_idx = {label_item: label_idx for label_idx, label_item in enumerate(CWSDataset.get_labels(os.path.join(directory, "ner_labels.txt")))}
        self.idx_to_label = {}
        for key, value in self.label_to_idx.items():
            self.idx_to_label[int(value)] = key

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        data_item = self.data_items[idx]
        token_sequence, label_sequence = data_item[0], data_item[1]
        label_sequence = [self.label_to_idx[label_item] for label_item in label_sequence]
        
        sentence_sequence = "".join(token_sequence[: min(self.max_length - 2, len(token_sequence))])
        
        #token_sequence = token_sequence[:self.max_length - 2]
        label_sequence = label_sequence[:min(self.max_length - 2, len(label_sequence))]
        # convert string to ids
        tokenizer_output = self.tokenizer.encode(sentence_sequence)
        
        bert_tokens = tokenizer_output.ids
        bert_sub_token_start, label_sequence = self._bert_sub_token_satrt_zh(token_sequence[: min(self.max_length - 2, len(token_sequence))], tokenizer_output, label_sequence)

        assert len(bert_tokens) <= self.max_length
        assert len(bert_tokens) == len(bert_sub_token_start)
        assert len(bert_tokens) == len(label_sequence)
        input_ids = torch.LongTensor(bert_tokens)
        label = torch.LongTensor(label_sequence)
        input_start = torch.LongTensor(bert_sub_token_start)
        return input_ids, label, input_start
    
    def _bert_sub_token_satrt_zh(self, token_sequence, token_data, label_sequence):
        char2idx = {}
        token_idx = 1
        ch_idx = 0
        for word in token_sequence:
            for _ in word:
                char2idx[ch_idx] = token_idx
                ch_idx += 1
            token_idx += 1
        
        start_sequence = [0]
        update_label_sequence = [self.label_to_idx["O"]]
        for idx_ in range(1, len(token_data.offsets)-1):
            start_sequence.append(char2idx[token_data.offsets[idx_][0]])
            update_label_sequence.append(label_sequence[char2idx[token_data.offsets[idx_][0]]-1])
        start_sequence.append(token_idx)
        update_label_sequence.append(self.label_to_idx["O"])
        return start_sequence, update_label_sequence

    @classmethod
    def get_labels(cls, file_path):
        """gets the list of labels for this data set."""
        
        if (cls.__ner_labels is None):
            cls.__ner_labels = []
            file = open(file_path, "r")
            for line in file:
                if (len(line.strip()) != 0):
                    cls.__ner_labels.append(line.strip())

        return cls.__ner_labels

    @staticmethod
    def _read_cws(input_file):
        data_list = []
        file = open(input_file, "r", encoding='utf-8')
        for line in file:
            line = line.strip()
            if line == "":
                continue
            line = line.split('\t')
            word_ = line[0].strip().split(" ")
            cws_tag = line[1].strip().split(" ")
            data_list.append([word_, cws_tag])
        return data_list