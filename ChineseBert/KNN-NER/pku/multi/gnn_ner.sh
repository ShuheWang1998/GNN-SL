export PYTHONPATH="$PWD"

DARA_DIR="/nfs1/shuhe/gnn_ner/data/pku/feature_files"
SAVE_PATH="/nfs1/shuhe/gnn_ner/results/pku_gnn_32"
DATASTORE_DIR="/nfs1/shuhe/gnn_ner/data/pku/feature_files"
NER_VOCAB_PATH="/nfs1/shuhe/gnn_ner/data/pku/ner_labels.txt"
NEIGHBOUR_DIR="/nfs1/shuhe/gnn_ner/data/pku/feature_files/32"

rm -r $SAVE_PATH
mkdir -p $SAVE_PATH

CUDA_VISIBLE_DEVICES=2 python ./KNN-NER/gnn_trainer.py \
--lr 5e-5 \
--max_epochs 20 \
--weight_decay 0.01 \
--warmup_proportion 0.002  \
--train_batch_size 2 \
--eval_batch_size 2 \
--accumulate_grad_batches 8 \
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
--gnn_layer 3 \
--gnn_head 8 \
--gcc_dropout 0.1 \
--gcc_attention_dropout 0.1 \
--gpus="1"
