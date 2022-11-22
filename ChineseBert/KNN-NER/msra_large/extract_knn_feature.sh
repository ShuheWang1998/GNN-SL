export PYTHONPATH="$PWD"

DARA_DIR="/nfs1/shuhe/gnn_ner/data/zh_msra"
FILE_NAME="char.bmes"
SAVE_PATH="/nfs1/shuhe/gnn_ner/results/msra_large_true"
BERT_PATH="/nfs1/shuhe/gnn_ner/models/ChineseBERT-large"
PARAMS_FILE="/nfs1/shuhe/gnn_ner/results/msra_large_true/log/version_0/hparams.yaml"
CHECKPOINT_PATH="/nfs1/shuhe/gnn_ner/results/msra_large_true/checkpoint/epoch=10.ckpt"
DATASTORE_PATH="/nfs1/shuhe/gnn_ner/data/zh_msra/feature_files_bert"
LCONTEXT=1
RCONTEXT=1

rm -r $DATASTORE_PATH
mkdir -p $DATASTORE_PATH

for sub_name in "test" "dev" "train";do

CUDA_VISIBLE_DEVICES=1 python ./KNN-NER/build_datastore.py \
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

CUDA_VISIBLE_DEVICES=1 python ./KNN-NER/build_datastore.py \
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
