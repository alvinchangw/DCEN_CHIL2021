<h3 align="center">
<p>
This is our Pytorch implementation of RNA Alternative Splicing Prediction with Discrete Compositional Energy Network.

<!-- RNA Alternative Splicing Prediction with Discrete Compositional Energy Network (ACM CHIL 2021)
Alvin Chan, Anna Korsakova, Yew Soon Ong, Fernaldo Richtia Winnerdy, Kah Wai Lim, Anh Tuan Phan
https://arxiv.org/abs/1912.05699 -->

</h3>

## Installation
```bash
pip install -r requirements.txt
```

## Data: Context Augmented Psi Dataset (CAPD)
Download the CAPD [here](https://doi.org/10.21979/N9/FFN0XH), unzip and place the files in `dataset/`.


## Saved weights
Saved weights of all the models in the paper are available in: `saved_models`.

## Training codes

Train spliceai on the splice site classification and psi regression objectives:
```shell
sh train_scripts/train_spliceai_clsreg.sh
```
This training script saves two weights files `site_aux_pytorch_model.bin` and `spliceai_pytorch_model.bin` at each checkpoint folder.

Train splicesome net on the psi regression objective with pretrained `site_aux_pytorch_model.bin` and `spliceai_pytorch_model.bin` weights from spliceai:
```shell
sh train_scripts/train_dcen.sh
```


## Evaluation codes

Evaluate and output pred and label npy files for each patient's gene data, output file is formatted as <tissue_type><patient_id>_chr<chromosome_number>:
```shell
python eval_scripts/schedule_eval_dcen.py
```
The evaluation result is computed from the schedule_eval_* script and is stored in each of these folders only cover one particular tissue_type, patient_id and chromosome_number. Other schedule_eval_* scripts are available for other ablation variants. These scripts run `python train_and_infer.py` iteratively over all the relevant <tissue_type><patient_id>_chr<chromosome_number>.


### `python train_and_infer.py` key arguments for evaluation:
`--spliceosome_model_training_type` : determine the ablation variants of the model  
`--eval_patient_list` : patient id(s) to evaluate  
`--eval_chr_list` : chromosome number(s) to evaluate  
`--eval_output_dir_prefix` : output directory of evaluation results, pred and label npy files  

To compute evaluation across multiple tissue types, patients or genes use the aggregate_eval* scripts such as the ones below. 

Compute evaluation across multiple patient and gene pred and label npy files:
```shell
python aggregate_eval_all_models_alltest.py
```


Compute evalation across all patient and only gene from chromosome numbers excluded from training set:
```shell
python aggregate_eval_all_models_exchromosomes.py
```


Compute evalation across all patient and only gene longer than limit in training set:
```shell
python aggregate_eval_all_models_longgenes.py
```

## Prediction codes
Infer and output the splice sites' and transcripts' predictions as a jsonl file, for only C1orf21 TTLL7 genes.
```shell
sh predict_dcen.sh
```
Remove `--eval_gene_list` arg to infer all genes.


### `python train_and_infer.py` key arguments for prediction:
`--valid_data_dir` : directory containing samples to infer probabilities  
`--eval_gene_list` : list of genes to infer, spaced apart (e.g. "C1orf21 TTLL7" to infer only these two genes). Default: infer all genes in samples
`--eval_output_dir_prefix` : output directory of prediction results, pred and label npy files  
`--pred_output_filename` : output prediction json file name  
`--eval_patient_list` : patient id(s) to evaluate  
`--eval_chr_list` : chromosome number(s) to evaluate  

