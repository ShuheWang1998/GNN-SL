export PYTHONPATH="$PWD"

DARA_DIR="/data2/wangshuhe/gnn_ner/cws_data/cws/pku/feature_files"
SAVE_PATH="/data2/wangshuhe/gnn_ner/cws_results/pku_bert_base_gnn_32_label_7e-5_batch128"
DATASTORE_DIR="/data2/wangshuhe/gnn_ner/cws_data/cws/pku/feature_files"
NER_VOCAB_PATH="/data2/wangshuhe/gnn_ner/cws_data/cws/pku/ner_labels.txt"
NEIGHBOUR_DIR="/data2/wangshuhe/gnn_ner/cws_data/cws/pku/feature_files/32"

rm -r $SAVE_PATH
mkdir -p $SAVE_PATH

CUDA_VISIBLE_DEVICES=7 python ./gnn_trainer.py \
--lr 7e-5 \
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
--gnn_in_dim 768 \
--gnn_hidden_size 1024 \
--gnn_layer 4 \
--gnn_head 8 \
--gcc_dropout 0.1 \
--gcc_attention_dropout 0.1 \
--add_label \
--gpus="1"
