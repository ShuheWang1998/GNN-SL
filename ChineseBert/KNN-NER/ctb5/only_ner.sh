export PYTHONPATH="$PWD"

DARA_DIR="/nfs1/shuhe/gnn_ner/data/pos/ctb5"
FILE_NAME="char.bmes"
SAVE_PATH="/nfs1/shuhe/gnn_ner/results/ctb5_true_pre32"
BERT_PATH="/nfs1/shuhe/gnn_ner/models/ChineseBERT-large"

rm -r $SAVE_PATH
mkdir -p $SAVE_PATH

CUDA_VISIBLE_DEVICES=5 python ./KNN-NER/ner_trainer.py \
--lr 3e-5 \
--max_epochs 5 \
--max_length 512 \
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
--task_name "cws" \
--gpus="1"
