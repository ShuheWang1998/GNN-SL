export PYTHONPATH="$PWD"

DARA_DIR="/data2/wangshuhe/gnn_ner/data/en_conll03"
FILE_NAME="word.bmes"
SAVE_PATH="/data2/wangshuhe/gnn_ner/results/conll_bert_large"
BERT_PATH="/data2/wangshuhe/gnn_ner/models/bert-large-cased"
PARAMS_FILE="/data2/wangshuhe/gnn_ner/results/conll_bert_large/log/version_0/hparams.yaml"
CHECKPOINT_PATH="/data2/wangshuhe/gnn_ner/results/conll_bert_large/checkpoint/epoch=1.ckpt"
DATASTORE_PATH="/data2/wangshuhe/gnn_ner/data/en_conll03/feature_files"
LCONTEXT=1
RCONTEXT=1

for sub_name in "test" "dev" "train";do

CUDA_VISIBLE_DEVICES=2 python ./build_datastore.py \
--bert_path $BERT_PATH \
--batch_size 4 \
--workers 16 \
--max_length 512 \
--data_dir $DARA_DIR \
--file_name $FILE_NAME \
--path_to_model_hparams_file $PARAMS_FILE \
--checkpoint_path $CHECKPOINT_PATH \
--datastore_path $DATASTORE_PATH \
--datastore_sub_set $sub_name \
--datastore_l_context $LCONTEXT \
--datastore_r_context $RCONTEXT \
--gpus="1"

done

CUDA_VISIBLE_DEVICES=2 python ./build_datastore.py \
--bert_path $BERT_PATH \
--batch_size 4 \
--workers 16 \
--max_length 512 \
--data_dir $DARA_DIR \
--file_name $FILE_NAME \
--path_to_model_hparams_file $PARAMS_FILE \
--checkpoint_path $CHECKPOINT_PATH \
--datastore_path $DATASTORE_PATH \
--datastore_sub_set "train" \
--build_datastore \
--gpus="1"