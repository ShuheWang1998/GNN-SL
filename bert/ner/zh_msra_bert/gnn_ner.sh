export PYTHONPATH="$PWD"

DARA_DIR="/data2/wangshuhe/gnn_ner/data/zh_msra/feature_files"
SAVE_PATH="/data2/wangshuhe/gnn_ner/data/zh_msra/feature_files/msra_bert_base_gnn_32"
DATASTORE_DIR="/data2/wangshuhe/gnn_ner/data/zh_msra/feature_files"
NER_VOCAB_PATH="/data2/wangshuhe/gnn_ner/data/zh_msra/ner_labels.txt"
NEIGHBOUR_DIR="/data2/wangshuhe/gnn_ner/data/zh_msra/feature_files/32"

mkdir -p $SAVE_PATH

CUDA_VISIBLE_DEVICES=0 /opt/intel/oneapi/intelpython/latest/bin/python ./gnn_trainer.py \
--lr 3e-5 \
--max_epochs 20 \
--weight_decay 0.01 \
--warmup_proportion 0.001  \
--train_batch_size 4 \
--eval_batch_size 4 \
--accumulate_grad_batches 1 \
--save_topk 20 \
--val_check_interval 0.1 \
--save_path $SAVE_PATH \
--datastore_dir $DATASTORE_DIR \
--neighbour_dir $NEIGHBOUR_DIR \
--ner_vocab_path $NER_VOCAB_PATH \
--data_dir $DARA_DIR \
--gnn_k 32 \
--gnn_in_dim 768 \
--gnn_hidden_size 1024 \
--gnn_layer 3 \
--gnn_head 2 \
--gpus="1"
