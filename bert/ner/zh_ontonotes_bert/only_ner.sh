export PYTHONPATH="$PWD"

DARA_DIR="/data2/wangshuhe/gnn_ner/data/zh_ontonotes4"
FILE_NAME="char.bmes"
SAVE_PATH="/data2/wangshuhe/gnn_ner/results/zh_ontonotes4_bert_base"
BERT_PATH="/data2/wangshuhe/gnn_ner/models/chinese-bert-wwm-ext"

#rm -r $SAVE_PATH
#mkdir -p $SAVE_PATH

CUDA_VISIBLE_DEVICES=3 python ./ner_trainer.py \
--lr 2e-5 \
--max_epochs 10 \
--max_length 275 \
--weight_decay 0.001 \
--hidden_dropout_prob 0.1 \
--warmup_proportion 0.1  \
--train_batch_size 16 \
--accumulate_grad_batches 1 \
--save_topk 20 \
--val_check_interval 0.25 \
--save_path $SAVE_PATH \
--bert_path $BERT_PATH \
--data_dir $DARA_DIR \
--file_name $FILE_NAME \
--precision=16 \
--language zh \
--save_ner_prediction \
--gpus="1"