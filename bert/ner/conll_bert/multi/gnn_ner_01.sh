export PYTHONPATH="$PWD"

DARA_DIR="/data2/wangshuhe/gnn_ner/data/official_conll03/feature_files_context_7"
SAVE_PATH="/data2/wangshuhe/gnn_ner/results/conll_bert_large_gnn_32_label_context_7_9e-5_batch128__multi_edge"
DATASTORE_DIR="/data2/wangshuhe/gnn_ner/data/official_conll03/feature_files_context_7"
NER_VOCAB_PATH="/data2/wangshuhe/gnn_ner/data/official_conll03/ner_labels.txt"
NEIGHBOUR_DIR="/data2/wangshuhe/gnn_ner/data/official_conll03/feature_files_context_7/32"

rm -r $SAVE_PATH
mkdir -p $SAVE_PATH

CUDA_VISIBLE_DEVICES=4 python ./gnn_trainer.py \
--lr 9e-5 \
--max_epochs 20 \
--weight_decay 0.01 \
--warmup_proportion 0.002  \
--train_batch_size 1 \
--eval_batch_size 1 \
--accumulate_grad_batches 128 \
--save_topk 5 \
--val_check_interval 0.25 \
--save_path $SAVE_PATH \
--datastore_dir $DATASTORE_DIR \
--neighbour_dir $NEIGHBOUR_DIR \
--ner_vocab_path $NER_VOCAB_PATH \
--data_dir $DARA_DIR \
--gnn_k 32 \
--gnn_in_dim 1024 \
--gnn_hidden_size 1024 \
--gnn_layer 6 \
--gnn_head 8 \
--gcc_dropout 0.1 \
--gcc_attention_dropout 0.1 \
--add_label \
--gpus="1"
