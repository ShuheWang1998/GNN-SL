export PYTHONPATH="$PWD"

PREMODEL=/data2/wangshuhe/gnn_ner/results/ontonotes_bert_large_bio_gnn_context7_9e-5_batch16_multi_edge_truemetric/checkpoint/epoch=2_v0.ckpt
PREFILE=/data2/wangshuhe/gnn_ner/results/ontonotes_bert_large_bio_gnn_context7_9e-5_batch16_multi_edge_truemetric/log/version_0/hparams.yaml

link_temperature=0.1
link_ratio=1

for link_temperature in 0.01 0.11 0.21 0.31 0.41 0.51 0.61 0.71 0.81 0.91; do
for link_ratio in 0.01 0.11 0.21 0.31 0.41 0.51 0.61 0.71 0.81 0.91; do

CUDA_VISIBLE_DEVICES=5 python ./gnn_knn_test.py \
--path_to_model_hparams_file $PREFILE \
--checkpoint_path $PREMODEL \
--add_knn \
--link_temperature $link_temperature \
--link_ratio $link_ratio \
--gnn_k 32 \
--gpus="1"

done
done