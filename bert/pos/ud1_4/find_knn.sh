export PYTHONPATH="$PWD"

DARA_DIR="/data2/wangshuhe/gnn_ner/pos_data/pos/ud1/feature_files_bio_context_3"
SAVE_DIR="/data2/wangshuhe/gnn_ner/pos_data/pos/ud1/feature_files_bio_context_3/32"
DATASTORE_DIR="/data2/wangshuhe/gnn_ner/pos_data/pos/ud1/feature_files_bio_context_3"

mkdir -p $SAVE_DIR

for sub_name in "test" "dev" "train";do

SAVE_PATH=${SAVE_DIR}/${sub_name}_neighbour_idx.npy
CUDA_VISIBLE_DEVICES=5 python ./offline_knn_search.py \
--save_path $SAVE_PATH \
--datastore_dir $DATASTORE_DIR \
--data_dir $DARA_DIR \
--prefix $sub_name \
--gnn_k 32 \
--batch_size 1 \
--is_gpu

done