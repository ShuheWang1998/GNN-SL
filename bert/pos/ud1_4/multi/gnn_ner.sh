export PYTHONPATH="$PWD"

DARA_DIR="/data2/wangshuhe/gnn_ner/pos_data/pos/ud1/feature_files_bio_context_3"
SAVE_PATH="/data2/wangshuhe/gnn_ner/results_pos/ud1.4_bert_base_gnn_bio_1e-4_16_warm0.02"
DATASTORE_DIR="/data2/wangshuhe/gnn_ner/pos_data/pos/ud1/feature_files_bio_context_3"
NER_VOCAB_PATH="/data2/wangshuhe/gnn_ner/pos_data/pos/ud1/ner_labels.txt"
NEIGHBOUR_DIR="/data2/wangshuhe/gnn_ner/pos_data/pos/ud1/feature_files_bio_context_3/32"

rm -r $SAVE_PATH
mkdir -p $SAVE_PATH

CUDA_VISIBLE_DEVICES=2 python ./gnn_trainer.py \
--lr 1e-4 \
--max_epochs 30 \
--weight_decay 0.01 \
--warmup_proportion 0.02  \
--train_batch_size 1 \
--eval_batch_size 1 \
--accumulate_grad_batches 16 \
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
--gnn_layer 3 \
--gnn_head 8 \
--gcc_dropout 0.1 \
--gcc_attention_dropout 0.1 \
--gpus="1"
