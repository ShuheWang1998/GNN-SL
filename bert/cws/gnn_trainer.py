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
import re
import json
import argparse
import logging
from collections import namedtuple

from utils import set_random_seed
from metrics import SpanF1ForNER

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
        if self.add_label:
            ntypes["label"] = 2
            etypes["label_edge"] = 2
        self.model = GNNNERModel(ntype2idx=ntypes,
                                 etype2idx=etypes,
                                 in_dim=self.args.gnn_in_dim,
                                 hidden_dim=self.args.gnn_hidden_size,
                                 out_dim=self.args.gnn_in_dim,
                                 n_layers=self.args.gnn_layer, n_heads=self.args.gnn_head,
                                 add_label=self.add_label, dropout=self.args.gcc_dropout,
                                 attn_drop=self.args.gcc_attention_dropout,
                                 num_labels=self.num_labels)

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

    def forward(self, input_ids):
        attention_mask = (input_ids != 0).long()
        return self.model(input_ids, attention_mask=attention_mask)

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

    def training_step(self, batch, batch_idx):
        # batch_size * max_seq_len * hidden_size
        # batch_size * max_seq_len * gnn_k
        # batch_size * max_seq_len
        sequence_features, neighbours, sequence_labels = batch
        batch_szie, max_seq_len, hidden_size = sequence_features.shape
        real_seq_len = (sequence_labels != -1).long().sum(dim=-1)
        sequence_mask = (sequence_labels != -1).bool()
        graphs = self.datastore.build_graph(sequence_features, neighbours, real_seq_len, knn_search_k=self.args.gnn_k)
        logits = []
        labels = []
        batch_graph = dgl.batch([sub_graph[-2] for sub_graph in graphs])
        batch_labels = None
        if self.add_label:
            batch_labels = torch.cat([sub_graph[-1] for sub_graph in graphs], dim=0)
        features = self.model(batch_graph, labels=batch_labels)
        last_index = 0
        for batch_graph in graphs:
            batch_idx, seq_len_idx, _, _ = batch_graph
            logit_feature = features[last_index:last_index+seq_len_idx]
            if seq_len_idx < max_seq_len:
                logit_feature = torch.cat((logit_feature, features[0].view(1, -1).expand(max_seq_len-seq_len_idx, self.num_labels)))
            logits.append(logit_feature.unsqueeze(dim=0))
            labels.append(sequence_labels[batch_idx].view(1, -1))
            last_index += seq_len_idx
        logits = torch.cat(logits, dim=0).view(batch_szie, max_seq_len, self.num_labels)
        labels = torch.cat(labels, dim=0).view(batch_szie, max_seq_len)
        loss = self.compute_loss(logits, labels, loss_mask=sequence_mask)

        tf_board_logs = {
            "train_loss": loss,
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"]
        }
        return {"loss": loss, "log": tf_board_logs}

    def validation_step(self, batch, batch_idx):
        # batch_size * max_seq_len * hidden_size, batch_size * max_seq_len
        sequence_features, neighbours, sequence_labels = batch
        batch_szie, max_seq_len, hidden_size = sequence_features.shape
        real_seq_len = (sequence_labels != -1).long().sum(dim=-1)
        sequence_mask = (sequence_labels != -1).bool()
        graphs = self.datastore.build_graph(sequence_features, neighbours, real_seq_len, knn_search_k=self.args.gnn_k)
        logits = []
        labels = []
        batch_graph = dgl.batch([sub_graph[-2] for sub_graph in graphs])
        batch_labels = None
        if self.add_label:
            batch_labels = torch.cat([sub_graph[-1] for sub_graph in graphs], dim=0)
        features = self.model(batch_graph, labels=batch_labels)
        last_index = 0
        for batch_graph in graphs:
            batch_idx, seq_len_idx, _, _ = batch_graph
            logit_feature = features[last_index:last_index+seq_len_idx]
            if seq_len_idx < max_seq_len:
                logit_feature = torch.cat((logit_feature, features[0].view(1, -1).expand(max_seq_len-seq_len_idx, self.num_labels)))
            logits.append(logit_feature.unsqueeze(dim=0))
            labels.append(sequence_labels[batch_idx].view(1, -1))
            last_index += seq_len_idx
        logits = torch.cat(logits, dim=0).view(batch_szie, max_seq_len, self.num_labels)
        labels = torch.cat(labels, dim=0).view(batch_szie, max_seq_len)
        loss = self.compute_loss(logits, labels, loss_mask=sequence_mask)

        probabilities, argmax_labels = self.postprocess_logits_to_labels(logits)
        confusion_matrix = self.ner_evaluation_metric(argmax_labels, labels, sequence_mask=sequence_mask)
        return {"val_loss": loss, "confusion_matrix": confusion_matrix}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        confusion_matrix = torch.stack([x["confusion_matrix"] for x in outputs]).sum(0)
        all_true_positive, all_false_positive, all_false_negative = confusion_matrix
        precision, recall, f1 = self.ner_evaluation_metric.compute_f1_using_confusion_matrix(all_true_positive, all_false_positive, all_false_negative)

        self.result_logger.info(f"EVAL INFO -> current_epoch is: {self.trainer.current_epoch}, current_global_step is: {self.trainer.global_step} ")
        self.result_logger.info(f"EVAL INFO -> valid_f1 is: {f1}")
        tensorboard_logs = {"val_loss": avg_loss, "val_f1": f1, }
        return {"val_loss": avg_loss, "val_log": tensorboard_logs, "val_f1": f1, "val_precision": precision, "val_recall": recall}
    
    def test_step(self, batch, batch_idx):
        # batch_size * max_seq_len * hidden_size, batch_size * max_seq_len
        sequence_features, neighbours, sequence_labels = batch
        batch_szie, max_seq_len, hidden_size = sequence_features.shape
        real_seq_len = (sequence_labels != -1).long().sum(dim=-1)
        sequence_mask = (sequence_labels != -1).bool()
        graphs = self.datastore.build_graph(sequence_features, neighbours, real_seq_len, knn_search_k=self.args.gnn_k)
        logits = []
        labels = []
        batch_graph = dgl.batch([sub_graph[-2] for sub_graph in graphs])
        batch_labels = None
        if self.add_label:
            batch_labels = torch.cat([sub_graph[-1] for sub_graph in graphs], dim=0)
        features = self.model(batch_graph, labels=batch_labels)
        last_index = 0
        for batch_graph in graphs:
            batch_idx, seq_len_idx, _, _ = batch_graph
            logit_feature = features[last_index:last_index+seq_len_idx]
            if seq_len_idx < max_seq_len:
                logit_feature = torch.cat((logit_feature, features[0].view(1, -1).expand(max_seq_len-seq_len_idx, self.num_labels)))
            logits.append(logit_feature.unsqueeze(dim=0))
            labels.append(sequence_labels[batch_idx].view(1, -1))
            last_index += seq_len_idx
        logits = torch.cat(logits, dim=0).view(batch_szie, max_seq_len, self.num_labels)
        labels = torch.cat(labels, dim=0).view(batch_szie, max_seq_len)

        probabilities, argmax_labels = self.postprocess_logits_to_labels(logits)
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
        self.result_logger.info(f"TEST RESULT -> TEST F1: {f1}, Precision: {precision}, Recall: {recall} ")
        return {"test_log": tensorboard_logs, "test_f1": f1, "test_precision": precision, "test_recall": recall}

    def postprocess_logits_to_labels(self, logits):
        """input logits should in the shape [batch_size, seq_len, num_labels]"""
        probabilities = F.softmax(logits, dim=2) # shape of [batch_size, seq_len, num_labels]
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
        neighbour_file = os.path.join(self.args.neighbour_dir, f"{prefix}_neighbour_idx.npy")
        dataset = GraphDataset(feature_info_path=feature_info_file,
                               feature_path=feature_file, 
                               label_path=label_file,
                               nei_path=neighbour_file, 
                               datastore=self.datastore, 
                               search_knn_k=self.args.gnn_k)

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

    parser.add_argument("--train_batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--use_memory", action="store_true", help="load dataset to memory to accelerate.")
    parser.add_argument("--data_dir", type=str, help="train data path")
    parser.add_argument("--datastore_dir", type=str, help="dir for datastore.")
    parser.add_argument("--neighbour_dir", type=str, help="dir for neighbours.")
    parser.add_argument("--save_path", type=str, help="train data path")
    parser.add_argument("--save_topk", default=1, type=int, help="save topk checkpoint")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--no_lr_scheduler", action="store_true")
    parser.add_argument("--path_to_model_hparams_file", default="", type=str, help="use for evaluation")
    parser.add_argument("--checkpoint_path", default="", type=str, help="use for evaluation.")
    parser.add_argument("--save_ner_prediction", action="store_true", help="only work for test.")
    parser.add_argument("--ner_vocab_path", type=str, help="path for ner vocabulary.")
    parser.add_argument("--gnn_k", default=32, type=int, help="used for knn search in the process of gnn")
    parser.add_argument("--gnn_in_dim", default=1024, type=int, help="input_dim for gnn model.")
    parser.add_argument("--gnn_hidden_size", default=1024, type=int, help="hidden_size for gnn model.")
    parser.add_argument("--gnn_layer", default=3, type=int, help="the number of layers for gnn model.")
    parser.add_argument("--gnn_head", default=8, type=int, help="the number of heads for gnn model.")
    parser.add_argument("--add_label", action="store_true", help="use label embedding for gnn.")
    parser.add_argument("--gcc_dropout", default=0.1, type=float, help="dropout used for gnn model.")
    parser.add_argument("--gcc_attention_dropout", default=0.1, type=float, help="attention dropout used for gnn model attention architecture.")

    return parser


def main():
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    model = GNNNER(args)

    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(args.save_path, "checkpoint", "{epoch}",),
                                          save_top_k=args.save_topk,
                                          save_last=False,
                                          monitor="val_f1",
                                          mode="max",
                                          verbose=True,
                                          period=-1,)

    logger = TensorBoardLogger(save_dir=args.save_path,
                               name='log')

    # save args
    with open(os.path.join(args.save_path, "checkpoint", "args.json"), "w") as f:
        args_dict = args.__dict__
        del args_dict["tpu_cores"]
        json.dump(args_dict, f, indent=4)

    trainer = Trainer.from_argparse_args(args,
                                         checkpoint_callback=checkpoint_callback,
                                         logger=logger,
                                         deterministic=True)
    trainer.fit(model)

    # after training, use the model checkpoint which achieves the best f1 score on dev set to compute the f1 on test set.
    best_f1_on_dev, path_to_best_checkpoint = find_best_checkpoint_on_dev(args.save_path)
    model.result_logger.info("=&"*20)
    model.result_logger.info(f"Best F1 on DEV is {best_f1_on_dev}")
    model.result_logger.info(f"Best checkpoint on DEV set is {path_to_best_checkpoint}")
    checkpoint = torch.load(path_to_best_checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    trainer.test(model)
    model.result_logger.info("=&"*20)


def find_best_checkpoint_on_dev(output_dir: str, log_file: str = "eval_result_log.txt"):
    with open(os.path.join(output_dir, log_file)) as f:
        log_lines = f.readlines()

    F1_PATTERN = re.compile(r"val_f1 reached \d+\.\d* \(best")
    # val_f1 reached 0.00000 (best 0.00000)
    CKPT_PATTERN = re.compile(r"saving model to \S+ as top")
    checkpoint_info_lines = []
    for log_line in log_lines:
        if "saving model to" in log_line:
            checkpoint_info_lines.append(log_line)
    # example of log line
    # Epoch 00000: val_f1 reached 0.00000 (best 0.00000), saving model to /data/xiaoya/outputs/glyce/0117/debug_5_12_2e-5_0.001_0.001_275_0.1_1_0.25/checkpoint/epoch=0.ckpt as top 20
    best_f1_on_dev = 0
    best_checkpoint_on_dev = 0
    for checkpoint_info_line in checkpoint_info_lines:
        current_f1 = float(re.findall(F1_PATTERN, checkpoint_info_line)[0].replace("val_f1 reached ", "").replace(" (best", ""))
        current_ckpt = re.findall(CKPT_PATTERN, checkpoint_info_line)[0].replace("saving model to ", "").replace(" as top", "")

        if current_f1 >= best_f1_on_dev:
            best_f1_on_dev = current_f1
            best_checkpoint_on_dev = current_ckpt

    return best_f1_on_dev, best_checkpoint_on_dev


def evaluate():
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model = GNNNER.load_from_checkpoint(checkpoint_path=args.checkpoint_path,
                                        hparams_file=args.path_to_model_hparams_file,
                                        map_location=None,
                                        batch_size=1)
    trainer = Trainer.from_argparse_args(args, deterministic=True)

    trainer.test(model)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()