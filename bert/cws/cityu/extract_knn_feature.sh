export PYTHONPATH="$PWD"

DARA_DIR="/data2/wangshuhe/gnn_ner/cws_data/cws/cityu"
FILE_NAME="char.bio"
SAVE_PATH="/data2/wangshuhe/gnn_ner/cws_results/cityu_bert_base_bio"
BERT_PATH="/data2/wangshuhe/gnn_ner/models/bert-base-chinese"
PARAMS_FILE="/data2/wangshuhe/gnn_ner/cws_results/cityu_bert_base_bio/log/version_0/hparams.yaml"
CHECKPOINT_PATH="/data2/wangshuhe/gnn_ner/cws_results/cityu_bert_base_bio/checkpoint/epoch=12.ckpt"
DATASTORE_PATH="/data2/wangshuhe/gnn_ner/cws_data/cws/cityu/feature_files"
LCONTEXT=1
RCONTEXT=1

for sub_name in "test" "dev" "train";do

CUDA_VISIBLE_DEVICES=5 python ./build_datastore.py \
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

CUDA_VISIBLE_DEVICES=5 python ./build_datastore.py \
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