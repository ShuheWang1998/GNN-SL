export PYTHONPATH="$PWD"

DARA_DIR="/nfs1/shuhe/gnn_ner/data/pos/ud1/feature_files_context5"
SAVE_PATH="/nfs1/shuhe/gnn_ner/results/ud1.4_gnn_32_label_context5"
DATASTORE_DIR="/nfs1/shuhe/gnn_ner/data/pos/ud1/feature_files_context5"
NER_VOCAB_PATH="/nfs1/shuhe/gnn_ner/data/pos/ud1/ner_labels.txt"
NEIGHBOUR_DIR="/nfs1/shuhe/gnn_ner/data/pos/ud1/feature_files_context5/32"

rm -r $SAVE_PATH
mkdir -p $SAVE_PATH

CUDA_VISIBLE_DEVICES=7 python ./KNN-NER/gnn_trainer.py \
--lr 3e-4 \
--max_epochs 30 \
--weight_decay 0.001 \
--warmup_proportion 0.1  \
--train_batch_size 2 \
--eval_batch_size 2 \
--accumulate_grad_batches 18 \
--save_topk 2 \
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
--gcc_dropout 0.2 \
--gcc_attention_dropout 0.2 \
--add_label \
--gpus="1"
