export PYTHONPATH="$PWD"

DARA_DIR="/nfs1/shuhe/gnn_ner/data/pos/ud1"
FILE_NAME="char.bmes"
SAVE_PATH="/nfs1/shuhe/gnn_ner/results/ud1.4"
BERT_PATH="/nfs1/shuhe/gnn_ner/models/ChineseBERT-large"
PARAMS_FILE="/nfs1/shuhe/gnn_ner/results/ud1.4_true_data/log/version_0/hparams.yaml"
CHECKPOINT_PATH="/nfs1/shuhe/gnn_ner/results/ud1.4_true_data/checkpoint/epoch=8_v0.ckpt"
DATASTORE_PATH="/nfs1/shuhe/gnn_ner/data/pos/ud1/feature_files_true"

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
--get_datastore_index \
--task_name "cws" \
--gpus="1"