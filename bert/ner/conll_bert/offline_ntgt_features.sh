export PYTHONPATH="$PWD"

FEATURE_INFO="/data2/wangshuhe/gnn_ner/data/official_conll03/feature_files_context_7"
LABEL_PATH="/data2/wangshuhe/gnn_ner/data/official_conll03/feature_files_context_7"
NEI_PATH="/data2/wangshuhe/gnn_ner/data/official_conll03/feature_files_context_7/32"
DATASTORE_DIR="/data2/wangshuhe/gnn_ner/data/official_conll03/feature_files_context_7"
SEARCH_KNN=32
BATCH_SIZE=1

for sub_name in "test" "dev" "train";do

CUDA_VISIBLE_DEVICES=1 python ./offline_neighbour_features.py \
--feature_info_path $FEATURE_INFO \
--label_path $LABEL_PATH \
--nei_path $NEI_PATH \
--prefix $sub_name \
--datastore_dir $DATASTORE_DIR \
--search_knn_k $SEARCH_KNN \
--batch_size $BATCH_SIZE

done