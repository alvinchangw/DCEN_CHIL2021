#!/bin/sh

export TRAIN_DATA_DIR=dataset/main_inputs_train
export VALID_DATA_DIR=dataset/main_inputs_valid
export EVAL_DATA_DIR=dataset/main_inputs_test
export ACT_REP_TRAIN_FILE=dataset/aux_inputs/aux_RBP_RNAmod.csv
export ACT_REP_EVAL_FILE=dataset/aux_inputs/aux_RBP_RNAmod.csv
export GENE_DICT_FILE=dataset/gene_dict_alltraintest.jsonl
export OUTPUT_DIR=saved_models/dcen/spliceosome_train_checkpoint-85000


python train_and_infer.py \
    --train_data_dir=$TRAIN_DATA_DIR \
    --valid_data_dir=$VALID_DATA_DIR \
    --eval_data_dir=$EVAL_DATA_DIR \
    --act_rep_train_file=$ACT_REP_TRAIN_FILE \
    --act_rep_eval_file=$ACT_REP_EVAL_FILE \
    --output_dir=$OUTPUT_DIR \
    --gene_seq_dict_file=$GENE_DICT_FILE \
    --per_gpu_train_batch_size 4 \
    --per_gpu_eval_batch_size 4 \
    --seed 234 \
    --do_infer \
    --spliceosome_model_training_type spliceosome_only \
    --eval_output_dir_prefix splice_site_transcript_predictions \
    --eval_gene_list C1orf21 TTLL7 \
