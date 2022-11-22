export PYTHONPATH="$PWD"

DARA_DIR="/nfs1/shuhe/gnn_ner/data/zh_ontonotes4"
FILE_NAME="char.bmes"
SAVE_PATH="/nfs1/shuhe/gnn_ner/results/ontonotes_large_true"
BERT_PATH="/nfs1/shuhe/gnn_ner/models/ChineseBERT-large"

rm -r $SAVE_PATH
mkdir -p $SAVE_PATH

CUDA_VISIBLE_DEVICES=0 python ./KNN-NER/ner_trainer.py \
--lr 3e-5 \
--max_epochs 15 \
--max_length 275 \
--weight_decay 0.002 \
--hidden_dropout_prob 0.2 \
--warmup_proportion 0.1  \
--train_batch_size 18 \
--accumulate_grad_batches 2 \
--save_topk 2 \
--val_check_interval 0.25 \
--save_path $SAVE_PATH \
--bert_path $BERT_PATH \
--data_dir $DARA_DIR \
--file_name $FILE_NAME \
--optimizer torch.adam \
--classifier multi \
--gpus="1"
