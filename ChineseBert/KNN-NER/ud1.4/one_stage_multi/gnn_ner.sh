export PYTHONPATH="$PWD"

DARA_DIR="/nfs1/shuhe/gnn_ner/data/pos/ud1"
FILE_NAME="char.bmes"
BERT_PATH="/nfs1/shuhe/gnn_ner/models/ChineseBERT-large"
SAVE_PATH="/nfs1/shuhe/gnn_ner/results/ud1.4_gnn_32_one_stage"
DATASTORE_DIR="/nfs1/shuhe/gnn_ner/data/pos/ud1/feature_files"
NER_VOCAB_PATH="/nfs1/shuhe/gnn_ner/data/pos/ud1/ner_labels.txt"
NEIGHBOUR_DIR="/nfs1/shuhe/gnn_ner/data/pos/ud1/feature_files/32_real_bert"
LCONTEXT=1
RCONTEXT=1

rm -r $SAVE_PATH
mkdir -p $SAVE_PATH

CUDA_VISIBLE_DEVICES=7 python ./KNN-NER/one_stage_trainer.py \
--lr 1e-4 \
--max_epochs 40 \
--weight_decay 0.002 \
--warmup_proportion 0.1 \
--train_batch_size 8 \
--eval_batch_size 2 \
--accumulate_grad_batches 4 \
--save_topk 2 \
--val_check_interval 0.25 \
--save_path $SAVE_PATH \
--datastore_dir $DATASTORE_DIR \
--neighbour_dir $NEIGHBOUR_DIR \
--ner_vocab_path $NER_VOCAB_PATH \
--data_dir $DARA_DIR \
--bert_path $BERT_PATH \
--file_name $FILE_NAME \
--optimizer torch.adam \
--classifier multi \
--task_name "cws" \
--gnn_k 32 \
--gnn_in_dim 1024 \
--gnn_hidden_size 1024 \
--gnn_layer 3 \
--gnn_head 16 \
--gcc_dropout 0.1 \
--gcc_attention_dropout 0.1 \
--datastore_l_context $LCONTEXT \
--datastore_r_context $RCONTEXT \
--add_label \
--gpus="1"
