export PYTHONPATH="$PWD"

DARA_DIR="/data2/wangshuhe/gnn_ner/data/official_conll03"
FILE_NAME="bio.txt"
SAVE_PATH="/data2/wangshuhe/gnn_ner/results/conll_bert_large_official_100epoch"
BERT_PATH="/data2/wangshuhe/gnn_ner/models/bert-large-cased-wwm"

rm -r $SAVE_PATH
mkdir -p $SAVE_PATH

CUDA_VISIBLE_DEVICES=0 python ./ner_trainer.py \
--lr 7e-5 \
--max_epochs 100 \
--max_length 512 \
--weight_decay 0.01 \
--hidden_dropout_prob 0.1 \
--warmup_proportion 0.001  \
--train_batch_size 8 \
--accumulate_grad_batches 16 \
--save_topk 20 \
--val_check_interval 0.25 \
--save_path $SAVE_PATH \
--bert_path $BERT_PATH \
--data_dir $DARA_DIR \
--file_name $FILE_NAME \
--gpus="1"
