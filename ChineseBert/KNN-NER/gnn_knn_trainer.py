#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file  : ner_trainer.py
@author: shuhe wang
@contact : shuhe_wang@shannonai.com
@date  : 2021/07/07 16:40
@version: 1.0
@desc  :
"""

import os
import json
import argparse
import logging
from collections import namedtuple

from subutils import set_random_seed
from metrics.ner import SpanF1ForNER

# enable reproducibility
# https://pytorch-lightning.readthedocs.io/en/latest/trainer.html
set_random_seed(2333)

import torch
from torch.nn import functional as F
from torch.nn.modules import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from datastore import Datastore
from graph_dataset import GraphDataset
from hgt import GNNNERModel
import dgl

class GNNNER(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        """Initialize a model, tokenizer and config"""
        super().__init__()
        self.args = args
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
        else:
            TmpArgs = namedtuple("tmp_args", field_names=list(args.keys()))
            self.args = args = TmpArgs(**args)
        
        self.add_label = args.add_label

        self.entity_labels = self.read_labels(self.args.ner_vocab_path)
        self.num_labels = len(self.entity_labels)
        is_gpu = ~(len([x for x in str(self.args.gpus).split(",") if x.strip()]) == 0)
        self.datastore = Datastore(datastore_dir=self.args.datastore_dir, is_gpu=is_gpu, add_label=self.add_label)
        ntypes = {"tgt": 0, "ntgt": 1}
        etypes = {"intra": 0, "inter": 1}
        #if self.add_label:
        #    ntypes["label"] = 2
        #    etypes["label_edge"] = 2
        self.model = GNNNERModel(
                        ntype2idx=ntypes,
                        etype2idx=etypes,
                        in_dim=self.args.gnn_in_dim,
                        hidden_dim=self.args.gnn_hidden_size,
                        out_dim=self.args.gnn_in_dim,
                        n_layers=self.args.gnn_layer, n_heads=self.args.gnn_head,
                        add_label=self.add_label, dropout=self.args.gcc_dropout,
                        attn_drop=self.args.gcc_attention_dropout,
                        num_labels=self.num_labels
                    )

        if is_gpu:
            self.model.to('cuda:0')

        self.ner_evaluation_metric = SpanF1ForNER(entity_labels=self.entity_labels, save_prediction=self.args.save_ner_prediction)

        format = '%(asctime)s - %(name)s - %(message)s'
        logging.basicConfig(format=format, filename=os.path.join(self.args.save_path, "eval_result_log.txt"), level=logging.INFO)
        self.result_logger = logging.getLogger(__name__)
        self.result_logger.setLevel(logging.INFO)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if self.args.optimizer == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters, betas=(0.9, 0.999), lr=self.args.lr, eps=self.args.adam_epsilon, )
        elif self.args.optimizer == "torch.adam":
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                          lr=self.args.lr,
                                          eps=self.args.adam_epsilon,
                                          weight_decay=self.args.weight_decay)
        else:
            raise ValueError("Please import the Optimizer first. ")
        num_gpus = len([x for x in str(self.args.gpus).split(",") if x.strip()])
        t_total = (len(self.train_dataloader()) // (self.args.accumulate_grad_batches * num_gpus) + 1) * self.args.max_epochs
        warmup_steps = int(self.args.warmup_proportion * t_total)
        if self.args.no_lr_scheduler:
            return [optimizer]
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    def inference(self, batch):
        # batch_size * max_seq_len * hidden_size
        # batch_size * max_seq_len * gnn_k
        # batch_size * max_seq_len
        sequence_features, neighbours, sequence_labels, sequence_mask, knn_values = batch
        batch_szie, max_seq_len, hidden_size = sequence_features.shape
        real_seq_len = (sequence_labels != -1).long().sum(dim=-1)
        max_seq_len = (sequence_mask != False).long().sum(dim=-1).max(dim=-1).values.item()
        graphs = self.datastore.build_graph(sequence_features, neighbours, real_seq_len, knn_search_k=self.args.gnn_k)
        logits = []
        labels = []
        batch_graph = dgl.batch([sub_graph[-2] for sub_graph in graphs])
        batch_labels = None
        if self.add_label:
            batch_labels = torch.cat([sub_graph[-1] for sub_graph in graphs], dim=0)
        features, features_tgt = self.model(batch_graph, labels=batch_labels)
        last_index = 0
        new_knn_values = []
        knn_labels = []
        for batch_graph in graphs:
            batch_idx, seq_len_idx, _, sub_knn_label = batch_graph
            sub_knn_label = sub_knn_label.view(-1, self.args.gnn_k, self.datastore.l_size+self.datastore.r_size+1)[:, :, self.datastore.l_size]
            logit_feature = features[last_index:last_index+seq_len_idx]
            logit_label = torch.masked_select(sequence_labels[batch_idx], sequence_mask[batch_idx]).view(1, -1)
            if seq_len_idx < max_seq_len:
                logit_feature = torch.cat((logit_feature, torch.zeros_like(features[0]).cuda().view(1, -1).expand(max_seq_len-seq_len_idx, self.num_labels)))
                logit_label = torch.cat((logit_label.view(-1), torch.zeros_like(logit_label[0][0]).cuda().expand(max_seq_len-seq_len_idx).fill_(-1)))
                sub_knn_label = torch.cat((sub_knn_label, sub_knn_label[0].view(1, -1).expand(max_seq_len-seq_len_idx, self.args.gnn_k)))

            logits.append(logit_feature.unsqueeze(dim=0))
            labels.append(logit_label.view(1, -1))

            new_knn_values.append(knn_values[batch_idx, :max_seq_len, ])
            knn_labels.append(sub_knn_label.view(1, max_seq_len, self.args.gnn_k))

            last_index += seq_len_idx
        logits = torch.cat(logits, dim=0).view(batch_szie, max_seq_len, self.num_labels)
        labels = torch.cat(labels, dim=0).view(batch_szie, max_seq_len)
        new_knn_values = torch.cat(new_knn_values, dim=0).view(batch_szie, max_seq_len, -1)
        knn_labels = torch.cat(knn_labels, dim=0).view(batch_szie, max_seq_len, self.args.gnn_k)
        loss_mask = (labels != -1).bool()

        return logits, labels, loss_mask, new_knn_values, knn_labels

    def compute_loss(self, logits, labels, loss_mask=None):
        """
        Desc:
            compute cross entropy loss
        Args:
            logits: FloatTensor, shape of [batch_size, sequence_len, num_labels]
            labels: LongTensor, shape of [batch_size, sequence_len, num_labels]
            loss_mask: Optional[LongTensor], shape of [batch_size, sequence_len].
                1 for non-PAD tokens, 0 for PAD tokens.
        """
        loss_fct = CrossEntropyLoss()
        if loss_mask is not None:
            active_loss = loss_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss
    
    def test_step(self, batch, batch_idx):

        #logits, labels, sequence_mask = self.forward(batch)
        #loss = self.compute_loss(logits, labels, loss_mask=sequence_mask)
        logits, labels, sequence_mask, knn_values, knn_labels = self.inference(batch)

        probabilities, argmax_labels = self.postprocess_logits_to_labels(logits, knn_values, knn_labels)
        confusion_matrix = self.ner_evaluation_metric(argmax_labels, labels, sequence_mask=sequence_mask)
        return {"confusion_matrix": confusion_matrix}

    def test_epoch_end(self, outputs):
        confusion_matrix = torch.stack([x["confusion_matrix"] for x in outputs]).sum(0)
        all_true_positive, all_false_positive, all_false_negative = confusion_matrix
        if self.args.save_ner_prediction:
            precision, recall, f1, entity_tuple = self.ner_evaluation_metric.compute_f1_using_confusion_matrix(all_true_positive, all_false_positive, all_false_negative, prefix="test")
            gold_entity_lst, pred_entity_lst = entity_tuple
            self.save_predictions_to_file(gold_entity_lst, pred_entity_lst)
        else:
            precision, recall, f1 = self.ner_evaluation_metric.compute_f1_using_confusion_matrix(all_true_positive, all_false_positive, all_false_negative)

        tensorboard_logs = {"test_f1": f1}
        self.result_logger.info(f"TEST RESULT -> TEST F1: {f1}, Precision: {precision}, Recall: {recall} , link: {self.link_temperature}, ratio: {self.link_ratio}")
        return {"test_log": tensorboard_logs, "test_f1": f1, "test_precision": precision, "test_recall": recall}

    def postprocess_logits_to_labels(self, logits, knn_values, knn_labels):
        """input logits should in the shape [batch_size, seq_len, num_labels]"""
        probabilities = F.softmax(logits, dim=2) # shape of [batch_size, seq_len, num_labels]

        sim_probs = torch.softmax((2 * knn_values / (knn_values.max()-knn_values.min()) - 1) / self.link_temperature, dim=-1) # [bsz, max_seq_len, gnn_k]

        knn_probabilities = torch.zeros_like(sim_probs[:, :, 0]).unsqueeze(-1).repeat([1, 1, self.num_labels])  # [bsz, max_seq_len, num_labels]
        knn_probabilities = knn_probabilities.scatter_add(dim=2, index=knn_labels.long(), src=sim_probs) # [bsz, max_seq_len, num_labels]

        probabilities = self.link_ratio*knn_probabilities + (1-self.link_ratio)*probabilities

        argmax_labels = torch.argmax(probabilities, 2, keepdim=False) # shape of [batch_size, seq_len]
        return probabilities, argmax_labels

    def train_dataloader(self,) -> DataLoader:
        return self.get_dataloader("train")

    def val_dataloader(self, ) -> DataLoader:
        return self.get_dataloader("dev")
    
    def test_dataloader(self, ) -> DataLoader:
        return self.get_dataloader("test")

    def _load_dataset(self, prefix="test"):
        feature_info_file = os.path.join(self.args.data_dir, f"{prefix}_feature_info.json")
        feature_file = os.path.join(self.args.data_dir, f"{prefix}_features.npy")
        label_file = os.path.join(self.args.data_dir, f"{prefix}_labels.npy")
        sequence_mask_file = os.path.join(self.args.data_dir, f"{prefix}_mask.npy")
        neighbour_file = os.path.join(self.args.neighbour_dir, f"{prefix}_neighbour_idx.npy")
        dataset = GraphDataset(feature_info_path=feature_info_file,
                               feature_path=feature_file, 
                               label_path=label_file,
                               mask_path=sequence_mask_file,
                               nei_path=neighbour_file, 
                               datastore=self.datastore, 
                               search_knn_k=self.args.gnn_k,
                               add_knn=self.add_knn)

        return dataset

    def get_dataloader(self, prefix="train", limit=None) -> DataLoader:
        """return {train/dev/test} dataloader"""
        dataset = self._load_dataset(prefix=prefix)

        if prefix == "train":
            batch_size = self.args.train_batch_size
            # small dataset like weibo ner, define data_generator will help experiment reproducibility.
            data_generator = torch.Generator()
            data_generator.manual_seed(self.args.seed)
            data_sampler = RandomSampler(dataset, generator=data_generator)
        else:
            batch_size = self.args.eval_batch_size
            data_sampler = SequentialSampler(dataset)

        # sampler option is mutually exclusive with shuffle
        dataloader = DataLoader(dataset=dataset, sampler=data_sampler, batch_size=batch_size,
                                num_workers=self.args.workers, drop_last=False)

        return dataloader

    def save_predictions_to_file(self, gold_entity_lst, pred_entity_lst, prefix="test"):
        save_file_path = os.path.join(self.args.save_path, "test_predictions.txt")
        print(f"INFO -> write predictions to {save_file_path}")
        with open(save_file_path, "w") as f:
            for idx_ in range(len(gold_entity_lst)):
                gold_label_item = gold_entity_lst[idx_]
                pred_label_item = pred_entity_lst[idx_]
                f.write("=!" * 20+"\n")
                f.write(json.dumps(gold_label_item)+"\n")
                f.write(json.dumps(pred_label_item)+"\n")
    
    def read_labels(self, label_file_path):
        file = open(label_file_path, "r")
        labels = file.readlines()
        file.close()
        return labels


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--path_to_model_hparams_file", default="", type=str, help="use for evaluation")
    parser.add_argument("--checkpoint_path", default="", type=str, help="use for evaluation.")
    parser.add_argument("--add_knn", action="store_true", help="add knn with gnn model.")
    parser.add_argument("--link_temperature", default=1.0, type=float, help="temperature used by edge linking.")
    parser.add_argument("--link_ratio", default=0.0, type=float, help="ratio of vocab probs predicted by edge linking.")
    parser.add_argument("--gnn_k", default=32, type=int, help="used for knn search in the process of gnn")

    return parser

def add_parser(model, args):
    model.add_knn = args.add_knn
    model.link_temperature = args.link_temperature
    model.link_ratio = args.link_ratio
    model.gnn_k = args.gnn_k

def main():
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model = GNNNER.load_from_checkpoint(checkpoint_path=args.checkpoint_path,
                                        hparams_file=args.path_to_model_hparams_file,
                                        map_location=None,
                                        eval_batch_size=1)
    trainer = Trainer.from_argparse_args(args, deterministic=True)
    add_parser(model, args)

    trainer.test(model)

if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
