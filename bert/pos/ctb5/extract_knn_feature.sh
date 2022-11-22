export PYTHONPATH="$PWD"

DARA_DIR="/data2/wangshuhe/gnn_ner/pos_data/pos/ctb5"
FILE_NAME="char.bio"
SAVE_PATH="/data2/wangshuhe/gnn_ner/results_pos/ctb5_bert_base_bio"
BERT_PATH="/data2/wangshuhe/gnn_ner/models/chinese-bert-wwm-ext"
PARAMS_FILE="/data2/wangshuhe/gnn_ner/results_pos/ctb5_bert_base_bio/log/version_0/hparams.yaml"
CHECKPOINT_PATH="/data2/wangshuhe/gnn_ner/results_pos/ctb5_bert_base_bio/checkpoint/epoch=12.ckpt"
DATASTORE_PATH="/data2/wangshuhe/gnn_ner/pos_data/pos/ctb5/feature_files_bio_context_3"
LCONTEXT=3
RCONTEXT=3

for sub_name in "test" "dev" "train";do

CUDA_VISIBLE_DEVICES=3 python ./build_datastore.py \
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
--pos_task \
--gpus="1"

done

CUDA_VISIBLE_DEVICES=3 python ./build_datastore.py \
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
--pos_task \
--gpus="1"