export PYTHONPATH="$PWD"

DARA_DIR="/nfs1/shuhe/gnn_ner/data/pos/ud1/feature_files_pre32_bert"
SAVE_DIR="/nfs1/shuhe/gnn_ner/data/pos/ud1/feature_files_pre32_bert/32"
DATASTORE_DIR="/nfs1/shuhe/gnn_ner/data/pos/ud1/feature_files_pre32_bert"

rm -r $SAVE_DIR
mkdir -p $SAVE_DIR

for sub_name in "test" "dev" "train";do

SAVE_PATH=${SAVE_DIR}/${sub_name}_neighbour_idx.npy
CUDA_VISIBLE_DEVICES=7 python ./KNN-NER/offline_knn_search.py \
--save_path $SAVE_PATH \
--datastore_dir $DATASTORE_DIR \
--data_dir $DARA_DIR \
--prefix $sub_name \
--gnn_k 32 \
--batch_size 2 \
--is_gpu

done