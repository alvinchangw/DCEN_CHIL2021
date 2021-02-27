import subprocess as sp
import shlex

train_data_dir = 'dataset/main_inputs_train'
valid_data_dir = 'dataset/main_inputs_valid'
eval_data_dir = 'dataset/main_inputs_test'
act_rep_train_file = 'dataset/aux_inputs/aux_RBP_RNAmod.csv'
act_rep_eval_file = 'dataset/aux_inputs/aux_RBP_RNAmod.csv'
gene_dict_file = 'dataset/gene_dict_alltraintest.jsonl'

output_dir = 'saved_models/AblationJuncBaseline/spliceosome_train_checkpoint-50000'

first_id = 1
last_id = 250
patient_ids_to_eval = [i for i in range(first_id, last_id+1)]
tissue_types_to_eval = ['ADP',
                        'BLD',
                        'BRN',
                        'BRS',
                        'CLN',
                        'HRT',
                        'KDN',
                        'LVR',
                        'LNG',
                        'LMP',
                        'PRS',
                        'SKM',
                        'TST',
                        'THR']
                        
chromosomes_to_eval = ['chr1',
                        'chr3',
                        'chr5',
                        'chr7',
                        'chr9',
                        'chr2_long',
                        'chr4_long',
                        'chr6_long',
                        'chr8_long',
                        'chr10_long',
                        'chr11_long',
                        'chr12_long',
                        'chr13_long',
                        'chr14_long',
                        'chr15_long',
                        'chr16_long',
                        'chr17_long',
                        'chr18_long',
                        'chr19_long',
                        'chr20_long',
                        'chr21_long',
                        'chr22_long',
                        'chrX_long',
                        'chrY_long',]
                        
def run_command(command):
    process = sp.Popen(shlex.split(command), stdout=sp.PIPE)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            print("breaking..")
            break
        if output:
            print(output.strip())
        else:
            print("no output, exiting command..")
            break
    rc = process.poll()
    return rc

for patient_id in patient_ids_to_eval:
    print(f"***********Evaluating patient ID:  {patient_id}************")

    for tissue_type in tissue_types_to_eval:
        print(f"***********Evaluating sample:  {tissue_type}{patient_id}************")
        eval_patient_list_str = f"{tissue_type}{patient_id}_"
        print("eval_patient_list_str: ", eval_patient_list_str)

        for chromosome in chromosomes_to_eval:
            print(f"***********Evaluating file:  {tissue_type}{patient_id}_{chromosome}************")
            eval_output_dir_prefix = f'{tissue_type}{patient_id}_{chromosome}'
            eval_chr_list_str = f'{chromosome}.'
            print("eval_chr_list_str: ", eval_chr_list_str)
            print("eval_output_dir_prefix: ", eval_output_dir_prefix)

            cmd = f"python train_and_infer.py \
                --train_data_dir={train_data_dir} \
                --valid_data_dir={valid_data_dir} \
                --eval_data_dir={eval_data_dir} \
                --act_rep_train_file={act_rep_train_file} \
                --act_rep_eval_file={act_rep_eval_file} \
                --output_dir={output_dir} \
                --gene_seq_dict_file={gene_dict_file} \
                --per_gpu_train_batch_size 4 \
                --per_gpu_eval_batch_size 4 \
                --seed 234 \
                --do_eval \
                --spliceosome_model_training_type spliceosome_only \
                --spliceosome_model_type spliceosome_junction_reg \
                --eval_patient_list {eval_patient_list_str} \
                --eval_chr_list {eval_chr_list_str} \
                --eval_output_dir_prefix {eval_output_dir_prefix} "

            print("Running: ", " ".join(cmd))
            run_command(cmd)
            print("Complete: ", " ".join(cmd))
