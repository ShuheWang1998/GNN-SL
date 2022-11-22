export PYTHONPATH="$PWD"

DARA_DIR="/data2/wangshuhe/gnn_ner/cws_data/cws/as"
FILE_NAME="char.bio"
SAVE_PATH="/data2/wangshuhe/gnn_ner/cws_results/as_bert_base_bio"
BERT_PATH="/data2/wangshuhe/gnn_ner/models/bert-base-chinese"

rm -r $SAVE_PATH
mkdir -p $SAVE_PATH

CUDA_VISIBLE_DEVICES=7 python ./ner_trainer.py \
--lr 3e-5 \
--max_epochs 15 \
--max_length 512 \
--weight_decay 0.01 \
--hidden_dropout_prob 0.1 \
--warmup_proportion 0.002  \
--train_batch_size 32 \
--accumulate_grad_batches 4 \
--save_topk 20 \
--val_check_interval 0.25 \
--save_path $SAVE_PATH \
--bert_path $BERT_PATH \
--data_dir $DARA_DIR \
--file_name $FILE_NAME \
--gpus="1"
