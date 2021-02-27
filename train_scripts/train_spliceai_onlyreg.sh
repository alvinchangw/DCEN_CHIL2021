#!/bin/sh

export TRAIN_DATA_DIR=dataset/main_inputs_train
export VALID_DATA_DIR=dataset/main_inputs_valid
export EVAL_DATA_DIR=dataset/main_inputs_test
export ACT_REP_TRAIN_FILE=dataset/aux_inputs/aux_RBP_RNAmod.csv
export ACT_REP_EVAL_FILE=dataset/aux_inputs/aux_RBP_RNAmod.csv
export GENE_DICT_FILE=dataset/gene_dict_alltraintest.jsonl
export CACHE_SPLICEAI_DIR=dataset/cache
export OUTPUT_DIR=models/spliceai_250samples_100Klen_onlyreg_2ep_dropprob_seed234

python train_and_infer_new.py \
    --do_train \
    --train_data_dir=$TRAIN_DATA_DIR \
    --valid_data_dir=$VALID_DATA_DIR \
    --eval_data_dir=$EVAL_DATA_DIR \
    --act_rep_train_file=$ACT_REP_TRAIN_FILE \
    --act_rep_eval_file=$ACT_REP_EVAL_FILE \
    --output_dir=$OUTPUT_DIR \
    --gene_seq_dict_file=$GENE_DICT_FILE \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 8 \
    --overwrite_output_dir \
    --num_train_spliceai_epochs 2 \
    --num_train_only_spliceosome_model_epochs 0 \
    --num_train_full_spliceai_spliceosome_model_epochs 0 \
    --save_total_limit 500 \
    --save_steps 2000 \
    --logging_steps 100 \
    --augment_transcript_data \
    --splice_site_cache_dir=$CACHE_SPLICEAI_DIR \
    --spliceai_cache_dataset_type point \
    --max_spliceai_forward_batch 100 \
    --logging_valid_steps 500 \
    --spliceai_model_training_type spliceai_cls_reg \
    --max_valid_step 100 \
    --keep_none_cls_prob 0.0011864 \
    --lambda_loss_class 0 \
    --seed 234 \

