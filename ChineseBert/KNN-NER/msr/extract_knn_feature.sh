export PYTHONPATH="$PWD"

DARA_DIR="/nfs1/shuhe/gnn_ner/data/msr"
FILE_NAME="char.bmes"
SAVE_PATH="/nfs1/shuhe/gnn_ner/results/msr"
BERT_PATH="/nfs1/shuhe/gnn_ner/models/ChineseBERT-large"
PARAMS_FILE="/nfs1/shuhe/gnn_ner/results/msr/log/version_0/hparams.yaml"
CHECKPOINT_PATH="/nfs1/shuhe/gnn_ner/results/msr/checkpoint/epoch=4_v0.ckpt"
DATASTORE_PATH="/nfs1/shuhe/gnn_ner/data/msr/feature_files"
LCONTEXT=1
RCONTEXT=1

rm -r $DATASTORE_PATH
mkdir -p $DATASTORE_PATH

for sub_name in "test" "dev" "train";do

CUDA_VISIBLE_DEVICES=4 python ./KNN-NER/build_datastore.py \
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

CUDA_VISIBLE_DEVICES=4 python ./KNN-NER/build_datastore.py \
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
