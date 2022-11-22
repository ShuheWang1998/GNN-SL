export PYTHONPATH="$PWD"

DARA_DIR="/data2/wangshuhe/gnn_ner/data/ontonotes-release-5.0/bio_data"
FILE_NAME="txt.bio"
SAVE_PATH="/data2/wangshuhe/gnn_ner/results/ontonotes_bert_large_bio_5e-5"
BERT_PATH="/data2/wangshuhe/gnn_ner/models/bert-large-cased-wwm"
PARAMS_FILE="/data2/wangshuhe/gnn_ner/results/ontonotes_bert_large_bio_5e-5/log/version_0/hparams.yaml"
CHECKPOINT_PATH="/data2/wangshuhe/gnn_ner/results/ontonotes_bert_large_bio_5e-5/checkpoint/epoch=8_v1.ckpt"
DATASTORE_PATH="/data2/wangshuhe/gnn_ner/data/ontonotes-release-5.0/bio_data/feature_files_context_7"

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
--get_datastore_index \
--gpus="1"