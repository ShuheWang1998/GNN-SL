export PYTHONPATH="$PWD"

DARA_DIR="/data2/wangshuhe/gnn_ner/data/en_conll03/feature_files"
SAVE_PATH="/data2/wangshuhe/gnn_ner/results/conll_bert_large_gnn_32_1e-5"
DATASTORE_DIR="/data2/wangshuhe/gnn_ner/data/en_conll03/feature_files"
NER_VOCAB_PATH="/data2/wangshuhe/gnn_ner/data/en_conll03/ner_labels.txt"
NEIGHBOUR_DIR="/data2/wangshuhe/gnn_ner/data/en_conll03/feature_files/32"

mkdir -p $SAVE_PATH

CUDA_VISIBLE_DEVICES=7 /opt/intel/oneapi/intelpython/latest/bin/python ./gnn_trainer.py \
--lr 5e-5 \
--max_epochs 10 \
--weight_decay 0.01 \
--warmup_proportion 0.002  \
--train_batch_size 4 \
--eval_batch_size 4 \
--accumulate_grad_batches 1 \
--save_topk 20 \
--val_check_interval 0.25 \
--save_path $SAVE_PATH \
--datastore_dir $DATASTORE_DIR \
--neighbour_dir $NEIGHBOUR_DIR \
--ner_vocab_path $NER_VOCAB_PATH \
--data_dir $DARA_DIR \
--gnn_k 32 \
--gnn_in_dim 1024 \
--gnn_hidden_size 1024 \
--gnn_layer 3 \
--gnn_head 8 \
--gcc_dropout 0.1 \
--gcc_attention_dropout 0.1 \
--gpus="1"
