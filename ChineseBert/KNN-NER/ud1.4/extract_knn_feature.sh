export PYTHONPATH="$PWD"

DARA_DIR="/nfs1/shuhe/gnn_ner/data/pos/ud1"
FILE_NAME="char.bmes"
SAVE_PATH="/nfs1/shuhe/gnn_ner/results/ud1.4_true_data_pre32"
BERT_PATH="/nfs1/shuhe/gnn_ner/models/ChineseBERT-large"
PARAMS_FILE="/nfs1/shuhe/gnn_ner/results/ud1.4_true_data_pre32/log/version_0/hparams.yaml"
CHECKPOINT_PATH="/nfs1/shuhe/gnn_ner/results/ud1.4_true_data_pre32/checkpoint/epoch=9.ckpt"
DATASTORE_PATH="/nfs1/shuhe/gnn_ner/data/pos/ud1/feature_files_pre32_bert"
LCONTEXT=1
RCONTEXT=1

rm -r $DATASTORE_PATH
mkdir -p $DATASTORE_PATH

for sub_name in "test" "dev" "train";do

CUDA_VISIBLE_DEVICES=7 python ./KNN-NER/build_datastore.py \
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
--task_name "cws" \
--gpus="1"

done

CUDA_VISIBLE_DEVICES=7 python ./KNN-NER/build_datastore.py \
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
--task_name "cws" \
--gpus="1"
