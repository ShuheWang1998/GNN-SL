export PYTHONPATH="$PWD"

PREMODEL=/data2/wangshuhe/gnn_ner/cws_results/pku_bert_base_gnn_32/checkpoint/epoch=14_v0.ckpt
PREFILE=/data2/wangshuhe/gnn_ner/cws_results/pku_bert_base_gnn_32/log/version_0/hparams.yaml

link_temperature=0.1
link_ratio=1

for link_temperature in 0.01 0.11 0.21 0.31 0.41 0.51 0.61 0.71 0.81 0.91; do
for link_ratio in 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31; do

CUDA_VISIBLE_DEVICES=3 python ./gnn_knn_test.py \
--path_to_model_hparams_file $PREFILE \
--checkpoint_path $PREMODEL \
--add_knn \
--link_temperature $link_temperature \
--link_ratio $link_ratio \
--gnn_k 32 \
--gpus="1"

done
done