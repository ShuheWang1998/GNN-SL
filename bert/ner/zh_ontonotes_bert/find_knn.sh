export PYTHONPATH="$PWD"

DARA_DIR="/data2/wangshuhe/gnn_ner/data/zh_ontonotes4/feature_files"
SAVE_DIR="/data2/wangshuhe/gnn_ner/data/zh_ontonotes4/feature_files/64"
DATASTORE_DIR="/data2/wangshuhe/gnn_ner/data/zh_ontonotes4/feature_files"

mkdir -p $SAVE_DIR

for sub_name in "test" "dev" "train";do

SAVE_PATH=${SAVE_DIR}/${sub_name}_neighbour_idx.npy
CUDA_VISIBLE_DEVICES=3 python ./offline_knn_search.py \
--save_path $SAVE_PATH \
--datastore_dir $DATASTORE_DIR \
--data_dir $DARA_DIR \
--prefix $sub_name \
--gnn_k 64 \
--batch_size 1 \
--is_gpu

done