import os
import time
import numpy as np
from scipy.stats import pearsonr, spearmanr

output_dirs = [
                'saved_models/dcen/spliceosome_train_checkpoint-85000',
                'saved_models/AblationJuncBaseline/spliceosome_train_checkpoint-50000',
                'saved_models/AblationMoreLayerBaseline/spliceai_train_checkpoint-120000',
                'saved_models/spliceai_clsreg/spliceai_train_checkpoint-200000',
                'saved_models/spliceai_onlycls/spliceai_train_checkpoint-70000',
                'saved_models/spliceai_onlyreg/spliceai_train_checkpoint-200000',
                ]
            
results_filename = 'all_long_genes_1to250.txt'

first_id = 1
last_id = 5
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
                    
chromosomes_to_eval = ['chr2_long',
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
          
def time_to_human(time):
    hrs = time//3600
    mins = (time - hrs*3600)//60
    secs = time - hrs*3600 - mins*60
    print('Overall time elapsed: {} hrs {} mins {} seconds'.format(int(hrs), int(mins), round(secs)))
    return hrs, mins, secs


def pearson_and_spearman(preds, labels):
    pearson_corr, pearson_p = pearsonr(preds, labels)
    spearman_corr, spearman_p = spearmanr(preds, labels)
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "pearson_p": pearson_p, 
        "spearmanr_p": spearman_p, 
        "corr": (pearson_corr + spearman_corr) / 2,
    }

for output_dir in output_dirs:
    print(f"***********Computing results for model:  {output_dir}************")
    all_pred_psi_acc_list = []
    all_pred_psi_don_list = []
    all_labels_psi_acc_list = []
    all_labels_psi_don_list = []
    start_time = time.time()

    for patient_id in patient_ids_to_eval:

        for tissue_type in tissue_types_to_eval:

            for chromosome in chromosomes_to_eval:
                print(f"***********Including files from:  {tissue_type}{patient_id}_{chromosome}************")
                eval_output_dir_prefix = f'{tissue_type}{patient_id}_{chromosome}'
                
                pred_psi_acc_path = os.path.join(output_dir, eval_output_dir_prefix, 'pred_psi_acc.npy')
                pred_psi_don_path = os.path.join(output_dir, eval_output_dir_prefix, 'pred_psi_don.npy')
                labels_psi_acc_path = os.path.join(output_dir, eval_output_dir_prefix, 'labels_psi_acc.npy')
                labels_psi_don_path = os.path.join(output_dir, eval_output_dir_prefix, 'labels_psi_don.npy')

                if os.path.exists(pred_psi_acc_path) and os.path.exists(pred_psi_don_path) and os.path.exists(labels_psi_acc_path) and os.path.exists(labels_psi_don_path):
                    with open(pred_psi_acc_path, 'rb') as f:
                        pred_psi_acc_list = np.load(f)
                        all_pred_psi_acc_list.append(pred_psi_acc_list)

                    with open(pred_psi_don_path, 'rb') as f:
                        pred_psi_don_list = np.load(f)
                        all_pred_psi_don_list.append(pred_psi_don_list)

                    with open(labels_psi_acc_path, 'rb') as f:
                        labels_psi_acc_list = np.load(f)
                        all_labels_psi_acc_list.append(labels_psi_acc_list)

                    with open(labels_psi_don_path, 'rb') as f:
                        labels_psi_don_list = np.load(f)
                        all_labels_psi_don_list.append(labels_psi_don_list)
                else:
                    print("Missing npy files in: ", eval_output_dir_prefix )

    for more_sample_name in more_sample_names:
        for chromosome in chromosomes_to_eval:
            print(f"***********Including files from:  {more_sample_name}_{chromosome}************")
            eval_output_dir_prefix = f'{more_sample_name}_{chromosome}'
            
            pred_psi_acc_path = os.path.join(output_dir, eval_output_dir_prefix, 'pred_psi_acc.npy')
            pred_psi_don_path = os.path.join(output_dir, eval_output_dir_prefix, 'pred_psi_don.npy')
            labels_psi_acc_path = os.path.join(output_dir, eval_output_dir_prefix, 'labels_psi_acc.npy')
            labels_psi_don_path = os.path.join(output_dir, eval_output_dir_prefix, 'labels_psi_don.npy')

            if os.path.exists(pred_psi_acc_path) and os.path.exists(pred_psi_don_path) and os.path.exists(labels_psi_acc_path) and os.path.exists(labels_psi_don_path):
                with open(pred_psi_acc_path, 'rb') as f:
                    pred_psi_acc_list = np.load(f)
                    all_pred_psi_acc_list.append(pred_psi_acc_list)

                with open(pred_psi_don_path, 'rb') as f:
                    pred_psi_don_list = np.load(f)
                    all_pred_psi_don_list.append(pred_psi_don_list)

                with open(labels_psi_acc_path, 'rb') as f:
                    labels_psi_acc_list = np.load(f)
                    all_labels_psi_acc_list.append(labels_psi_acc_list)

                with open(labels_psi_don_path, 'rb') as f:
                    labels_psi_don_list = np.load(f)
                    all_labels_psi_don_list.append(labels_psi_don_list)
            else:
                print("Missing npy files in: ", eval_output_dir_prefix )

    if len(all_pred_psi_acc_list) > 0 and len(all_pred_psi_don_list) > 0:
        print("***********Compiling all preds and labels***********")
        all_pred_psi_acc_list = np.concatenate(all_pred_psi_acc_list)
        all_pred_psi_don_list = np.concatenate(all_pred_psi_don_list)
        all_labels_psi_acc_list = np.concatenate(all_labels_psi_acc_list)
        all_labels_psi_don_list = np.concatenate(all_labels_psi_don_list)

        all_pred_psi_ss_list = np.concatenate([all_pred_psi_acc_list, all_pred_psi_don_list])
        all_labels_psi_ss_list = np.concatenate([all_labels_psi_acc_list, all_labels_psi_don_list])

        print("***********Computing correlations***********")
        acc_cor_result = pearson_and_spearman(all_pred_psi_acc_list, all_labels_psi_acc_list)
        don_cor_result = pearson_and_spearman(all_pred_psi_don_list, all_labels_psi_don_list)
        all_ss_cor_result = pearson_and_spearman(all_pred_psi_ss_list, all_labels_psi_ss_list)

        result = {"spearmanr_acc": acc_cor_result['spearmanr'], "pearson_acc": acc_cor_result['pearson'], 
                    "spearmanr_p_acc": acc_cor_result['spearmanr_p'], "pearson_p_acc": acc_cor_result['pearson_p'], 
                    "spearmanr_don": don_cor_result['spearmanr'], "pearson_don": don_cor_result['pearson'], 
                    "spearmanr_p_don": don_cor_result['spearmanr_p'], "pearson_p_don": don_cor_result['pearson_p'], 
                    "spearmanr_ss": all_ss_cor_result['spearmanr'], "pearson_ss": all_ss_cor_result['pearson'],
                    "spearmanr_p_ss": all_ss_cor_result['spearmanr_p'], "pearson_p_ss": all_ss_cor_result['pearson_p']}

        end_time = time.time()
        output_eval_file = os.path.join(output_dir, results_filename)

        with open(output_eval_file, "w") as writer:
            print("***** Compiled correlation results {} *****".format(output_eval_file))
            for key in sorted(result.keys()):
                print("  {} = {}".format(key, str(result[key])))
                writer.write("%s = %s\n" % (key, str(result[key])))
            writer.write("*****\n patient_ids_to_eval: {} \n".format(str(patient_ids_to_eval)))
            writer.write("*****\n tissue_types_to_eval: {} \n".format(str(tissue_types_to_eval)))
            writer.write("*****\n chromosomes_to_eval: {} \n".format(str(chromosomes_to_eval)))

            hrs, mins, secs = time_to_human(end_time - start_time)
            writer.write("Overall time elapsed: {} hrs {} mins {} seconds".format(int(hrs), int(mins), round(secs)))
    else:
        print("No prediction npy files found for: ", output_dir, ", skipping.." )





