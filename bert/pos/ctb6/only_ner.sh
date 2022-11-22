export PYTHONPATH="$PWD"

DARA_DIR="/data2/wangshuhe/gnn_ner/pos_data/pos/ctb6"
FILE_NAME="char.bio"
SAVE_PATH="/data2/wangshuhe/gnn_ner/results_pos/ctb6_bert_base_bio"
BERT_PATH="/data2/wangshuhe/gnn_ner/models/chinese-bert-wwm-ext"

rm -r $SAVE_PATH
mkdir -p $SAVE_PATH

CUDA_VISIBLE_DEVICES=7 python ./ner_trainer.py \
--lr 7e-5 \
--max_epochs 20 \
--max_length 512 \
--weight_decay 0.01 \
--hidden_dropout_prob 0.1 \
--warmup_proportion 0.002  \
--train_batch_size 16 \
--accumulate_grad_batches 1 \
--save_topk 20 \
--val_check_interval 0.25 \
--save_path $SAVE_PATH \
--bert_path $BERT_PATH \
--data_dir $DARA_DIR \
--file_name $FILE_NAME \
--language zh \
--pos_task \
--gpus="1"
