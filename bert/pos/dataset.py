#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file  : ner_dataset.py
@author: shuhe wang
@contact : shuhe_wang@shannonai.com
@date  : 2021/07/08 20:09
@version: 1.0
@desc  :
"""

import os
from importlib_metadata import SelectableGroups

import torch
from torch.utils.data import Dataset
from tokenizers import BertWordPieceTokenizer
from transformers import RobertaTokenizer


class NERDataset(Dataset):

    __ner_labels = None

    """the Dataset Class for NER task."""
    def __init__(self, directory, prefix, vocab_file, config_path, max_length=512, file_name="char.bmes", lower_case=False, language="en", en_roberta=False, pos_=False):
        """
        Args:
            directory: str, path to data directory.
            prefix: str, one of [train/dev/test]
            vocab_file: str, path to the vocab file for model pre-training.
            config_path: str, config_path must contain [pinyin_map.json, id2pinyin.json, pinyin2tensor.json]
            max_length: int,
        """
        super().__init__()
        self.max_length = max_length
        data_file_path = os.path.join(directory, "{}.{}".format(prefix, file_name))
        if not pos_:
            self.data_items = NERDataset._read_conll(data_file_path)
        else:
            self.data_items = NERDataset._read_pos(data_file_path)
        self.en_roberta = en_roberta
        if not en_roberta:
            self.tokenizer = BertWordPieceTokenizer(vocab_file, lowercase=lower_case)
        else:
            self.tokenizer = RobertaTokenizer.from_pretrained(vocab_file)
        self.label_to_idx = {label_item: label_idx for label_idx, label_item in enumerate(NERDataset.get_labels(os.path.join(directory, "ner_labels.txt")))}
        self.idx_to_label = {}
        for key, value in self.label_to_idx.items():
            self.idx_to_label[int(value)] = key
        self.language = language

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        data_item = self.data_items[idx]
        token_sequence, label_sequence = data_item[0], data_item[1]
        label_sequence = [self.label_to_idx[label_item] for label_item in label_sequence]
        
        concate_word = ""
        if (self.language == "en"):
            concate_word = " "
        sentence_sequence = concate_word.join(token_sequence[: min(self.max_length - 2, len(token_sequence))])
        
        #token_sequence = token_sequence[:self.max_length - 2]
        label_sequence = label_sequence[:min(self.max_length - 2, len(label_sequence))]
        # convert string to ids
        tokenizer_output = self.tokenizer.encode(sentence_sequence)
        
        '''
        if not self.en_roberta:
            bert_tokens = tokenizer_output.ids
            if (self.language == "zh"):
                label_sequence = self._update_labels_using_tokenize_offsets(tokenizer_output.offsets, label_sequence)
            else:
                label_sequence = self._update_labels_using_tokenize_offsets_english(tokenizer_output, label_sequence, token_sequence)
        else:
            bert_tokens = tokenizer_output
            label_sequence = self._roberta_update_labels_using_tokenize_offsets_english(token_sequence.strip().split(), label_sequence)
        '''
        if not self.en_roberta:
            bert_tokens = tokenizer_output.ids
            if self.language == "zh":
                bert_sub_token_start, label_sequence = self._bert_sub_token_satrt_zh(token_sequence[: min(self.max_length - 2, len(token_sequence))], tokenizer_output, label_sequence)
            else:
                bert_sub_token_start, label_sequence = self._bert_sub_token_satrt(sentence_sequence, tokenizer_output, label_sequence)

        assert len(bert_tokens) <= self.max_length
        assert len(bert_tokens) == len(bert_sub_token_start)
        assert len(bert_tokens) == len(label_sequence)
        input_ids = torch.LongTensor(bert_tokens)
        label = torch.LongTensor(label_sequence)
        input_start = torch.LongTensor(bert_sub_token_start)
        return input_ids, label, input_start

    def _update_labels_using_tokenize_offsets(self, offsets, original_sequence_labels):
        """part of offset sequence [(51, 52), (52, 54)] -> (token index after tokenized, original token index)"""
        update_sequence_labels = []
        for offset_idx, offset_item in enumerate(offsets):
            if offset_idx == 0 or offset_idx == (len(offsets) - 1):
                continue
            update_index, origin_index = offset_item
            current_label = original_sequence_labels[origin_index-1]
            update_sequence_labels.append(current_label)
        update_sequence_labels = [self.label_to_idx["O"]] + update_sequence_labels + [self.label_to_idx["O"]]
        return update_sequence_labels
    
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

    def _bert_sub_token_satrt(self, token_sequence, token_data, label_sequence):
        char2idx = {}
        token_idx = 1
        for idx_, ch in enumerate(token_sequence):
            if ch == ' ':
                token_idx += 1
            else:
                char2idx[idx_] = token_idx
        
        start_sequence = [0]
        update_label_sequence = [self.label_to_idx["O"]]
        for idx_ in range(1, len(token_data.offsets)-1):
            start_sequence.append(char2idx[token_data.offsets[idx_][0]])
            update_label_sequence.append(label_sequence[char2idx[token_data.offsets[idx_][0]]-1])
        start_sequence.append(token_idx+1)
        update_label_sequence.append(self.label_to_idx["O"])
        return start_sequence, update_label_sequence
    
    def _roberta_update_labels_using_tokenize_offsets_english(self, token_sequence, original_sequence_labels):
        updated_sequence_labels = [self.label_to_idx["O"]]
        for idx_, token_word in enumerate(token_sequence):
            add_prefix_space = True
            if idx_ == 0:
                add_prefix_space = False
            token_idx = self.tokenizer.encode(token_word, add_special_tokens=False, add_prefix_space=add_prefix_space)
            for sub_idx in range(len(token_idx)):
                if sub_idx > 0 and self.idx_to_label[original_sequence_labels[idx_]][0] == 'B':
                    updated_sequence_labels.append(self.label_to_idx['M'+self.idx_to_label[original_sequence_labels[idx_]][1:]])
                elif sub_idx != len(token_idx)-1 and len(token_idx) > 1 and self.idx_to_label[original_sequence_labels[idx_]][0] == 'E':
                    updated_sequence_labels.append(self.label_to_idx['M'+self.idx_to_label[original_sequence_labels[idx_]][1:]])
                else:
                    updated_sequence_labels.append(original_sequence_labels[idx_])
        updated_sequence_labels.append(self.label_to_idx["O"])
        return updated_sequence_labels

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
    def _read_conll(input_file, delimiter=" "):
        """load ner dataset from CoNLL-format files."""
        dataset_item_lst = []
        with open(input_file, "r", encoding="utf-8") as r_f:
            datalines = r_f.readlines()

        cached_token, cached_label = [], []
        for idx, data_line in enumerate(datalines):
            data_line = data_line.strip()
            if len(data_line) == 0:
                if (len(cached_token) != 0 and len(cached_label) != 0):
                    dataset_item_lst.append([cached_token, cached_label])
                cached_token, cached_label = [], []
            else:
                token_label = data_line.split(delimiter)
                token_data_line, label_data_line = token_label[0], token_label[1]
                cached_token.append(token_data_line)
                cached_label.append(label_data_line)
        return dataset_item_lst

    @staticmethod
    def _read_pos(input_file):
        dataset_item_lst = []
        file = open(input_file, "r", encoding='utf-8')
        for line in file:
            if line == 0:
                continue
            line = line.strip().split('\t')
            dataset_item_lst.append([line[0].strip().split(" "), line[1].strip().split(" ")])
        file.close()
        return dataset_item_lst
