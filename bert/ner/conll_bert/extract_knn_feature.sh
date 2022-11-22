export PYTHONPATH="$PWD"

DARA_DIR="/data2/wangshuhe/gnn_ner/data/official_conll03"
FILE_NAME="bio.txt"
SAVE_PATH="/data2/wangshuhe/gnn_ner/results/conll_bert_large_official_60epoch"
BERT_PATH="/data2/wangshuhe/gnn_ner/models/bert-large-cased-wwm"
PARAMS_FILE="/data2/wangshuhe/gnn_ner/results/conll_bert_large_official_60epoch/log/version_0/hparams.yaml"
CHECKPOINT_PATH="/data2/wangshuhe/gnn_ner/results/conll_bert_large_official_60epoch/checkpoint/epoch=28_v1.ckpt"
DATASTORE_PATH="/data2/wangshuhe/gnn_ner/data/official_conll03/feature_files_context_7"
LCONTEXT=3
RCONTEXT=3

for sub_name in "test" "dev" "train";do

CUDA_VISIBLE_DEVICES=6 python ./build_datastore.py \
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

CUDA_VISIBLE_DEVICES=6 python ./build_datastore.py \
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
--datastore_l_context $LCONTEXT \
--datastore_r_context $RCONTEXT \
--gpus="1"