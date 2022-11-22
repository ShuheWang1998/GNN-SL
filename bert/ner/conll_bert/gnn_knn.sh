export PYTHONPATH="$PWD"

PREMODEL=/data2/wangshuhe/gnn_ner/results/conll_bert_large_gnn_32_label/checkpoint/epoch=14_v1.ckpt
PREFILE=/data2/wangshuhe/gnn_ner/results/conll_bert_large_gnn_32_label/log/version_0/hparams.yaml

CUDA_VISIBLE_DEVICES=5 python ./gnn_knn_test.py \
--path_to_model_hparams_file $PREFILE \
--checkpoint_path $PREMODEL \
--gpus="1"
