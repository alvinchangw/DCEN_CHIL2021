#!/bin/sh

export TRAIN_DATA_DIR=dataset/main_inputs_train
export VALID_DATA_DIR=dataset/main_inputs_valid
export EVAL_DATA_DIR=dataset/main_inputs_test
export ACT_REP_TRAIN_FILE=dataset/aux_inputs/aux_RBP_RNAmod.csv
export ACT_REP_EVAL_FILE=dataset/aux_inputs/aux_RBP_RNAmod.csv
export GENE_DICT_FILE=dataset/gene_dict_alltraintest.jsonl
export CACHE_SPLICEAI_DIR=dataset/cache
export SPLICEAI_WEIGHTS=models/spliceai_clsreg/spliceai_train_checkpoint-130000/spliceai_pytorch_model.bin
export SITE_AUX_WEIGHTS=models/spliceai_clsreg/spliceai_train_checkpoint-130000/site_aux_pytorch_model.bin
export OUTPUT_DIR=models/dcen

python train_and_infer.py \
    --do_train \
    --train_data_dir=$TRAIN_DATA_DIR \
    --valid_data_dir=$VALID_DATA_DIR \
    --eval_data_dir=$EVAL_DATA_DIR \
    --act_rep_train_file=$ACT_REP_TRAIN_FILE \
    --act_rep_eval_file=$ACT_REP_EVAL_FILE \
    --output_dir=$OUTPUT_DIR \
    --gene_seq_dict_file=$GENE_DICT_FILE \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 16 \
    --overwrite_output_dir \
    --num_train_spliceai_epochs 0 \
    --num_train_only_spliceosome_model_epochs 1 \
    --num_train_full_spliceai_spliceosome_model_epochs 0 \
    --save_total_limit 500 \
    --save_steps 200 \
    --logging_steps 100 \
    --max_spliceai_forward_batch 100 \
    --logging_valid_steps 200 \
    --max_valid_step 100 \
    --num_sampled_sub_per_acc_don 0 \
    --load_pretrained_spliceai_path=$SPLICEAI_WEIGHTS \
    --load_pretrained_site_aux_model_path=$SITE_AUX_WEIGHTS \
    --spliceosome_model_training_type spliceosome_only \
    --eval_before_training \
