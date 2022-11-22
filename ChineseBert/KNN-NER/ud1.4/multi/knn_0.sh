export PYTHONPATH="$PWD"

PREMODEL=/nfs1/shuhe/gnn_ner/results/ud1.4_gnn_32_label_real_layer2/checkpoint/epoch=4.ckpt
PREFILE=/nfs1/shuhe/gnn_ner/results/ud1.4_gnn_32_label_real_layer2/log/version_0/hparams.yaml

link_temperature=1
link_ratio=1

#for link_temperature in 1.1 10.1 100.1; do
for link_ratio in 0.01 0.11 0.21 0.31 0.41 0.51 0.61 0.71 0.81 0.91; do

echo $link_temperature $link_ratio

CUDA_VISIBLE_DEVICES=7 python ./KNN-NER/gnn_knn_trainer.py \
--path_to_model_hparams_file $PREFILE \
--checkpoint_path $PREMODEL \
--add_knn \
--link_temperature $link_temperature \
--link_ratio $link_ratio \
--gnn_k 32 \
--gpus="1"

#done
done