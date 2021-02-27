# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple
import json
import math
import h5py
import time

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from bisect import bisect
from sklearn.metrics import average_precision_score
from scipy.stats import pearsonr, spearmanr

import sys
import pandas as pd


from modeling import (
    SpliceAINew,
    SiteAuxNet,
    SpliceosomeModel,
    SpliceosomeModelWithTranscriptProbLoss,
    SpliceosomeModelJunctionBaseline,
    SiteAuxMoreLayersExtension,
    AdamW,
    get_exponential_decay_schedule,
    get_linear_schedule_with_warmup,
)


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

WEIGHTS_NAME = "pytorch_model.bin"

logger = logging.getLogger(__name__)

def file_len(file_name):
    with open(file_name) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def time_to_human(time):
    hrs = time//3600
    mins = (time - hrs*3600)//60
    secs = time - hrs*3600 - mins*60
    print('Overall time elapsed: {} hrs {} mins {} seconds'.format(int(hrs), int(mins), round(secs)))
    return hrs, mins, secs

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

IN_MAP = np.asarray([[0, 0, 0, 0],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
# One-hot encoding of the inputs: 0 is for padding, and 1, 2, 3, 4 correspond
# to A, C, G, T respectively.

OUT_MAP = np.asarray([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]])
# One-hot encoding of the outputs: 0 is for no splice, 1 is for acceptor,
# 2 is for donor and -1 is for padding.


tissue_dict = {'ADP': 'adipose',
                'ADR': 'adrenal',
                'BLD': 'blood',
                'BRN': 'brain',
                'BRS': 'breast',
                'CLN': 'colon',
                'HRT': 'heart',
                'KDN': 'kidney',
                'LVR': 'liver',
                'LNG': 'lung',
                'LMP': 'lymph',
                'OVR': 'ovary',
                'PRS': 'prostate',
                'SKM': 'skeletal',
                'TST': 'testis',
                'THR': 'thyroid',}

tissue_list = ['ADP',
            'ADR',
            'BLD',
            'BRN',
            'BRS',
            'CLN',
            'HRT',
            'KDN',
            'LVR',
            'LNG',
            'LMP',
            'OVR',
            'PRS',
            'SKM',
            'TST',
            'THR',]
                
def one_hot_encode_input(X):
    return IN_MAP[X.astype('int8')]

def match_patient_chr_lists(data_file, patient_list=None, chr_list=None):
    # Check whether datafile matches patient filter
    if patient_list is not None:
        patient_match = False
        for patient_id in patient_list:
            if patient_id in data_file:
                patient_match = True
                break
    else:
        patient_match = True
    
    # Check whether datafile matches chromosome filter
    if chr_list is not None:
        chr_match = False
        for chr_name in chr_list:
            if chr_name in data_file:
                chr_match = True
                break
    else:
        chr_match = True

    return (patient_match and chr_match)




class MultipleJsonlDatasetForRegressionTruncatedGene(Dataset):
    # Fairly sample gene seq segments, rather than uniformly sampling gene which oversamples seq segments in short genes
    def __init__(self, args, data_dir, act_rep_file=None, epsilon=1e-7, max_seq_len=None, label_seq_len_spliceai=None, return_sample_metadata=False, tissue_types=None, same_gene_order=True,
            patient_list=None, chr_list=None):
        self.context_len = args.context_len
        self.no_tissue_type_as_feature = args.no_tissue_type_as_feature

        # Whether to return sample_name and gene_name metadata for each sample, used to cache computed spliceai hidden states
        self.return_sample_metadata = return_sample_metadata

        if max_seq_len is None:
            self.max_seq_len = args.max_train_seq_len
        else:
            self.max_seq_len = max_seq_len

        if label_seq_len_spliceai is None:
            self.label_seq_len_spliceai = args.max_main_seq_len_spliceai
        else:
            self.label_seq_len_spliceai = max_main_seq_len_spliceai
        if self.label_seq_len_spliceai != (self.max_seq_len - args.context_len):
            raise ValueError("Incorrect label_seq_len_spliceai, must be equal to (max_seq_len - args.context_len).")

        # Open gene seq dict
        with open(args.gene_seq_dict_file, encoding="utf-8") as f:
            lines = f.readlines()
            self.gene_seq_dict = {}
            for line in lines:
                json_dict = json.loads(line)
                for key in json_dict.keys():
                    self.gene_seq_dict[key] = json_dict[key]
        logger.info(" {} gene sequences in records".format(str(len(self.gene_seq_dict))))

        # Set up main input files
        logger.info(" Setting up main input files ")
        if tissue_types is None:
            self.tissue_types = tissue_list
        self.data_files = os.listdir(data_dir)
        if patient_list is not None or chr_list is not None:
            self.data_files = [ os.path.join(data_dir, data_file) for data_file in self.data_files if ( match_patient_chr_lists(data_file, patient_list, chr_list) and data_file[:3] in self.tissue_types )]
        else:
            self.data_files = [ os.path.join(data_dir, data_file) for data_file in self.data_files if data_file[:3] in self.tissue_types]
        list.sort(self.data_files)
        logger.info(" {} main input files in total ".format(str(len(self.data_files))))

        self.start_indices = [0] * len(self.data_files)
        self.sample_count = 0
        self.chr_start_indices_dict = {}

        # TODO: Compute chr_start_indices_dict
        total_nt_count = 0
        total_acc_don_count = 0
        for index, data_file in tqdm(enumerate(self.data_files), desc="Count lines in files"):
            data_filename = data_file.split('/')[-1]
            chr_name = '_'.join(data_filename.split('_')[1:])
            chr_sample_count = None
            if chr_name in self.chr_start_indices_dict.keys():
                chr_sample_count = self.chr_start_indices_dict[chr_name]['sample_count']
            else:
                self.chr_start_indices_dict[chr_name] = {}
                # compute chr_start_indices_dict and sample_count
                with open(data_file, encoding="utf-8") as f:
                    lines = f.readlines()
                    chr_start_indices = [0] * len(lines)
                    chr_sample_count = 0
                    chr_nt_count = 0
                    chr_acc_don_count = 0
                    for jsonl_ind, jsonl in enumerate(lines):
                        json_dict = json.loads(jsonl)
                        gene_name = json_dict['gene']
                        seq = self.gene_seq_dict[gene_name]
                        seq_len = len(seq)
                        label_seq_len = seq_len - args.context_len
                        chr_nt_count += label_seq_len
                        gene_seq_sample_count = math.ceil(label_seq_len / self.label_seq_len_spliceai)         
                        chr_start_indices[jsonl_ind] = chr_sample_count
                        chr_sample_count += gene_seq_sample_count

                        # Count number of acc and don in gene sequences 
                        exons = json_dict['exons']
                        acc_don_list = []
                        for transcript in exons.split(";"):
                            if len(transcript) > 0:
                                for exon in transcript.split(','):
                                    acc, don = exon.split(' ')
                                    acc = int(acc)
                                    don = int(don)
                                    acc_don_list = acc_don_list + [acc, don]
                        acc_don_list = list(set(acc_don_list))
                        acc_don_count = len(acc_don_list)
                        chr_acc_don_count += acc_don_count

                    self.chr_start_indices_dict[chr_name]['start_indices'] = chr_start_indices
                    self.chr_start_indices_dict[chr_name]['sample_count'] = chr_sample_count
                    self.chr_start_indices_dict[chr_name]['nt_count'] = chr_nt_count
                    self.chr_start_indices_dict[chr_name]['acc_don_count'] = chr_acc_don_count

                    total_nt_count += chr_nt_count
                    total_acc_don_count += chr_acc_don_count

            self.start_indices[index] = self.sample_count
            self.sample_count += chr_sample_count

        total_none_nt_count = total_nt_count - total_acc_don_count
        ratio_none_over_acc_don = total_none_nt_count/total_acc_don_count
        logger.info(" {} nt in gene sequences ".format(str(total_nt_count)))
        logger.info(" {} acc & don in gene sequences ".format(str(total_acc_don_count)))
        logger.info(" {} non-acc/don nts in gene sequences ".format(str(total_none_nt_count)))
        logger.info(" {} non-acc/don imbalance ratio ".format(str(ratio_none_over_acc_don)))

        # Process act_rep csv input file
        # Load training act_rep csv to compute train set stats
        train_df = pd.read_csv(args.act_rep_train_file)
        train_df = train_df.set_index('samples')

        if args.c_log_act_rep is None:
            # use half of min non-zero value to transform data before log op
            c_log_act_rep = train_df[train_df>0].min().min() / 2
        else:
            c_log_act_rep = args.c_log_act_rep
        if args.log_act_rep:
            train_df = np.log(train_df.add(c_log_act_rep))
        self.train_act_rep_means = {}
        self.train_act_rep_stds = {}
        logger.info(" {} aux transcripts for act_rep input ".format(str(len(train_df.columns))))
        for col in train_df.columns:
            if 'ENST' in col:
                self.train_act_rep_means[col] = train_df[col].mean()
                self.train_act_rep_stds[col] = train_df[col].std()

        if act_rep_file is None:
            act_rep_file = args.act_rep_train_file

        # Open and process act_rep_file
        logger.info(" Normalizing act_rep values ")
        df = pd.read_csv(act_rep_file)
        df = df.set_index('samples')
        if args.log_act_rep:
            df = np.log(df.add(c_log_act_rep))
        for col in df.columns:
            if 'ENST' in col:
                # normalize data according to train stats
                df[col] = (df[col] - self.train_act_rep_means[col]) / (self.train_act_rep_stds[col] + epsilon)

        self.act_rep_df = df

    def __len__(self):
        return self.sample_count

    def __getitem__(self, index):
        file_index = bisect(self.start_indices, index) - 1

        index_in_file = index - self.start_indices[file_index]
        data_file = self.data_files[file_index]

        # retrieve act_rep data
        sample_filename = data_file.split('/')[-1]
        sample_name_chr = sample_filename.split('.')[0]
        sample_name = sample_name_chr.split('_')[0]
        act_rep = self.act_rep_df.loc[sample_name, :]
        act_rep_np = act_rep.values

        # Get one-hot encoding of cell type with sample_name
        if not self.no_tissue_type_as_feature:
            cell_type = sample_name[:3]
            cell_type_onehot = np.zeros(len(tissue_list))
            if cell_type.upper() in tissue_list:
                cell_type_ind = tissue_list.index(cell_type.upper())
                cell_type_onehot[cell_type_ind] = 1
            act_rep_np = np.concatenate([act_rep_np, cell_type_onehot], axis=0)

        # retrieve main gene input data
        with open(data_file, encoding="utf-8") as f:
            lines = f.readlines()

            data_filename = data_file.split('/')[-1]
            chr_name = '_'.join(data_filename.split('_')[1:])
            chr_start_indices = self.chr_start_indices_dict[chr_name]['start_indices']

            line_index = bisect(chr_start_indices, index_in_file) - 1       
            jsonl = lines[line_index]
            index_in_gene = index_in_file - chr_start_indices[line_index]

            json_dict = json.loads(jsonl)
            gene_name = json_dict['gene']
            seq = self.gene_seq_dict[gene_name]
            alabels = json_dict['alabels']
            dlabels = json_dict['dlabels']
            exons = json_dict['exons']

            # process input data here
            # truncate to max_seq_len
            seq_len = len(seq)
            if index_in_gene > 0:
                offset_start = index_in_gene * self.label_seq_len_spliceai
                offset_end = offset_start + self.max_seq_len

                trimmed_seq = seq[offset_start:offset_end]
            else:
                offset_start = 0
                trimmed_seq = seq
            
            # generate label seq 0: None, 1: acceptor, 2: donor
            label_seq_len = len(trimmed_seq) - self.context_len
            label_seq = np.zeros(label_seq_len)
            for transcript in exons.split(";"):
                if len(transcript) > 0:
                    for exon in transcript.split(','):
                        acc, don = exon.split(' ')
                        acc = int(acc)
                        don = int(don)
                        if acc >= offset_start and acc < (offset_start + label_seq_len):
                            label_seq[acc-offset_start] = 1
                        if don >= offset_start and don < (offset_start + label_seq_len):
                            label_seq[don-offset_start] = 2

            # generate psi seq [0, 1]
            psi_seq = np.zeros(label_seq_len)
            for a_pos_str in alabels.keys():
                a_pos = int(a_pos_str)
                if a_pos >= offset_start and a_pos < (offset_start + label_seq_len):
                    psi_seq[a_pos-offset_start] = alabels[a_pos_str]
            for d_pos_str in dlabels.keys():
                d_pos = int(d_pos_str)
                if d_pos >= offset_start and d_pos < (offset_start + label_seq_len):
                    psi_seq[d_pos-offset_start] = dlabels[d_pos_str]

        trimmed_seq = trimmed_seq.upper().replace('A', '1').replace('C', '2')
        trimmed_seq = trimmed_seq.replace('G', '3').replace('T', '4').replace('N', '0')
        trimmed_seq_np = np.asarray(list(map(int, list(trimmed_seq))))
        trimmed_seq_one_hot = one_hot_encode_input(trimmed_seq_np)
        if self.return_sample_metadata:
            # Additionally return sample_name, gene_name
            return torch.tensor(trimmed_seq_one_hot, dtype=torch.float), torch.tensor(act_rep_np, dtype=torch.float), torch.tensor(label_seq, dtype=torch.long), torch.tensor(psi_seq, dtype=torch.float), sample_name, gene_name
        else:
            return torch.tensor(trimmed_seq_one_hot, dtype=torch.float), torch.tensor(act_rep_np, dtype=torch.float), torch.tensor(label_seq, dtype=torch.long), torch.tensor(psi_seq, dtype=torch.float)

# construct splice junction data from transcript data
def get_splice_junctions(transcripts, gene_start_token_position=-1, gene_end_token_position=9999999):
    all_transcripts_splice_junctions = []
    for transcript in transcripts:
        transcript_splice_junctions = []

        prev_don = gene_start_token_position
        prev_acc = None
        for exon_ind, exon in enumerate(transcript):
            cur_acc, cur_don = exon
            cur_splice_junction = (prev_don, cur_acc) 

            transcript_splice_junctions.append(cur_splice_junction)

            prev_acc = cur_acc
            prev_don = cur_don

        final_splice_junction = (prev_don, gene_end_token_position)
        transcript_splice_junctions.append(final_splice_junction)

        all_transcripts_splice_junctions.append(transcript_splice_junctions)
    return all_transcripts_splice_junctions


class MultipleJsonlDatasetForRegression(Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, args, data_dir, act_rep_file=None, augment_transcript_data=True, num_sampled_sub_per_acc_don=None, return_sample_metadata=False, epsilon=1e-7, tissue_types=None,
            patient_list=None, chr_list=None):
        self.context_len = args.context_len
        self.no_tissue_type_as_feature = args.no_tissue_type_as_feature

        self.augment_transcript_data = augment_transcript_data
        self.return_sample_metadata = return_sample_metadata
        self.max_fake_inclusion_transcripts = args.max_fake_inclusion_transcripts
        self.max_fake_exclusion_transcripts = args.max_fake_exclusion_transcripts

        self.gene_start_token_position = args.gene_start_token_position # -1
        self.gene_end_token_position = args.gene_end_token_position # 9999999

        self.max_fake_substitution_transcripts = args.max_fake_substitution_transcripts

        # number of sampled non-acceptor/donor nt at neighborhood of each acceptor/donor
        if num_sampled_sub_per_acc_don is None:
            self.num_sampled_sub_per_acc_don = args.num_sampled_sub_per_acc_don  
        else:
            self.num_sampled_sub_per_acc_don = num_sampled_sub_per_acc_don

        self.sampled_sub_nt_distance = args.sampled_sub_nt_distance
        self.prob_sub_per_exon = args.prob_sub_per_exon

        # Set up main input files
        if tissue_types is None:
            self.tissue_types = tissue_list
        self.data_files = os.listdir(data_dir)
        if patient_list is not None or chr_list is not None:
            self.data_files = [ os.path.join(data_dir, data_file) for data_file in self.data_files if ( match_patient_chr_lists(data_file, patient_list, chr_list) and data_file[:3] in self.tissue_types )]
        else:
            self.data_files = [ os.path.join(data_dir, data_file) for data_file in self.data_files if data_file[:3] in self.tissue_types]
        list.sort(self.data_files)

        self.start_indices = [0] * len(self.data_files)
        self.sample_count = 0
        for index, data_file in enumerate(self.data_files):
            self.start_indices[index] = self.sample_count
            self.sample_count += file_len(data_file)

        # Open gene seq dict
        with open(args.gene_seq_dict_file, encoding="utf-8") as f:
            lines = f.readlines()
            self.gene_seq_dict = {}
            for line in lines:
                json_dict = json.loads(line)
                for key in json_dict.keys():
                    self.gene_seq_dict[key] = json_dict[key]
        logger.info(" {} gene sequences in records".format(str(len(self.gene_seq_dict))))

        # Process act_rep csv input file
        # Load training act_rep csv to compute train set stats
        train_df = pd.read_csv(args.act_rep_train_file)
        train_df = train_df.set_index('samples')
        print("train_df: ", train_df)
        if args.c_log_act_rep is None:
            # use half of min non-zero value to transform data before log op
            c_log_act_rep = train_df[train_df>0].min().min() / 2
        else:
            c_log_act_rep = args.c_log_act_rep
        if args.log_act_rep:
            train_df = np.log(train_df.add(c_log_act_rep))
            print("log train_df: ", train_df)
        self.train_act_rep_means = {}
        self.train_act_rep_stds = {}
        for col in train_df.columns:
            if 'ENST' in col:
                self.train_act_rep_means[col] = train_df[col].mean()
                self.train_act_rep_stds[col] = train_df[col].std()

        if act_rep_file is None:
            act_rep_file = args.act_rep_train_file

        # Open and process act_rep_file
        df = pd.read_csv(act_rep_file)
        df = df.set_index('samples')
        if args.log_act_rep:
            df = np.log(df.add(c_log_act_rep))
        for col in df.columns:
            if 'ENST' in col:
                # normalize data according to train stats
                df[col] = (df[col] - self.train_act_rep_means[col]) / (self.train_act_rep_stds[col] + epsilon)

        self.act_rep_df = df

    def __len__(self):
        return self.sample_count


    def __getitem__(self, index):
        file_index = bisect(self.start_indices, index) - 1

        index_in_file = index - self.start_indices[file_index]
        data_file = self.data_files[file_index]

        # retrieve act_rep data
        sample_filename = data_file.split('/')[-1]
        sample_name_chr = sample_filename.split('.')[0]
        sample_name = sample_name_chr.split('_')[0]
        act_rep = self.act_rep_df.loc[sample_name, :]
        act_rep_np = act_rep.values

        # Get one-hot encoding of cell type with sample_name
        if not self.no_tissue_type_as_feature:
            cell_type = sample_name[:3]
            cell_type_onehot = np.zeros(len(tissue_list))
            if cell_type.upper() in tissue_list:
                cell_type_ind = tissue_list.index(cell_type.upper())
                cell_type_onehot[cell_type_ind] = 1
            act_rep_np = np.concatenate([act_rep_np, cell_type_onehot], axis=0)

        act_rep_tensor = torch.tensor(act_rep_np, dtype=torch.float)

        # retrieve main gene input data
        with open(data_file, encoding="utf-8") as f:
            lines = f.readlines()
            jsonl = lines[index_in_file]
            json_dict = json.loads(jsonl)
            gene_name = json_dict['gene']
            seq = self.gene_seq_dict[gene_name]
            main_seq_len = len(seq) - self.context_len
            anno_seq = seq.upper().replace('A', '1').replace('C', '2')
            anno_seq = anno_seq.replace('G', '3').replace('T', '4').replace('N', '0')
            alabels = json_dict['alabels']
            dlabels = json_dict['dlabels']
            exons = json_dict['exons']

            # Process input data here
            # process transcript data as lists of exon tuples
            transcripts = []
            exons_str = []
            acc_don_list = []
            transcripts_str = exons.split(";")[:-1]
            random.shuffle(transcripts_str)
            for transcript in transcripts_str:
                if len(transcript) > 0:
                    transcript_list = []
                    for exon in transcript.split(','):
                        exons_str.append(exon)
                        acc, don = exon.split(' ')
                        acc = int(acc)
                        don = int(don)
                        transcript_list.append((acc, don))
                        acc_don_list += [acc, don]

                    transcripts.append(transcript_list)

            exons_str = list(set(exons_str)) # compute unique exons
            num_exons = len(exons_str)

            if self.max_fake_substitution_transcripts < num_exons:
                transcript_sub_nt_sample_prob = 1
            else:
                transcript_sub_nt_sample_prob = self.max_fake_substitution_transcripts / num_exons

            prob_sub_per_exon = min(self.prob_sub_per_exon, transcript_sub_nt_sample_prob)

            # process non-acceptor, donor nt for fake_substitution_transcripts
            fake_substitution_transcripts = []
            processed_exons_sub = []
            acc_don_sub_list = []
            transcripts_sub_acc = []
            transcripts_sub_don = []
            break_fake_substitution_transcripts = False
            
            for transcript in transcripts_str:
                exons_in_transcript_str = transcript.split(',')
                for exon_ind, exon in enumerate(exons_in_transcript_str):
                    acc, don = exon.split(' ')
                    processed_exons_sub.append(exon)
                    acc = int(acc)
                    don = int(don)

                    if self.num_sampled_sub_per_acc_don > 1:
                        for i in range(int(self.num_sampled_sub_per_acc_don)):
                            # sample acc_sub from neighborhood of acc
                            acc_sub = random.randint(max(0, acc-self.sampled_sub_nt_distance), min(don, acc+self.sampled_sub_nt_distance))
                            # resample acc_sub if similar to acc or already sampled before in acc_don_sub_list
                            while acc_sub == acc:
                                acc_sub = random.randint(max(0, acc-self.sampled_sub_nt_distance), min(don, acc+self.sampled_sub_nt_distance))

                            # sample don_sub from neighborhood of don
                            don_sub = random.randint(max(acc, don-self.sampled_sub_nt_distance), min(main_seq_len-1, don+self.sampled_sub_nt_distance))
                            # resample don_sub if similar to don
                            while don_sub == don:
                                don_sub = random.randint(max(acc, don-self.sampled_sub_nt_distance), min(main_seq_len-1, don+self.sampled_sub_nt_distance))

                            acc_don_sub_list += [acc_sub, don_sub]

                            # create fake_substitution_transcript 
                            if i == 0 and random.random() < prob_sub_per_exon:
                                transcript_comma_edges = ',' + transcript + ','
                                transcript_sub_acc = transcript_comma_edges.replace(",{} {},".format(str(acc), str(don)), ",{} {},".format(str(acc_sub), str(don)))
                                # remove ',' from edges 
                                transcript_sub_acc = transcript_sub_acc[1:-1]
                                if transcript_sub_acc not in transcripts_str:
                                    transcripts_sub_acc.append(transcript_sub_acc)

                                transcript_sub_don = transcript_comma_edges.replace(",{} {},".format(str(acc), str(don)), ",{} {},".format(str(acc), str(don_sub)))
                                # remove ',' from edges 
                                transcript_sub_don = transcript_sub_don[1:-1]
                                if transcript_sub_don not in transcripts_str:
                                    transcripts_sub_don.append(transcript_sub_don)

                            if (len(transcripts_sub_acc) + len(transcripts_sub_don)) >= self.max_fake_substitution_transcripts:
                                # too many fake_substitution_transcripts
                                break

                    elif self.num_sampled_sub_per_acc_don > random.random():
                        # sample acc_sub from neighborhood of acc
                        acc_sub = random.randint(max(0, acc-self.sampled_sub_nt_distance), min(don, acc+self.sampled_sub_nt_distance))
                        # resample acc_sub if similar to acc or already sampled before in acc_don_sub_list
                        while acc_sub == acc:
                        # while acc_sub == acc or acc_sub in acc_don_sub_list:
                            acc_sub = random.randint(max(0, acc-self.sampled_sub_nt_distance), min(don, acc+self.sampled_sub_nt_distance))

                        # sample don_sub from neighborhood of don
                        don_sub = random.randint(max(acc, don-self.sampled_sub_nt_distance), min(main_seq_len-1, don+self.sampled_sub_nt_distance))
                        # resample don_sub if similar to don
                        while don_sub == don:
                        # while don_sub == don or acc_sub in acc_don_sub_list:
                            don_sub = random.randint(max(acc, don-self.sampled_sub_nt_distance), min(main_seq_len-1, don+self.sampled_sub_nt_distance))

                        acc_don_sub_list += [acc_sub, don_sub]

                        # create fake_substitution_transcript 
                        transcript_comma_edges = ',' + transcript + ','
                        transcript_sub_acc = transcript_comma_edges.replace(",{} {},".format(str(acc), str(don)), ",{} {},".format(str(acc_sub), str(don)))
                        # remove ',' from edges 
                        transcript_sub_acc = transcript_sub_acc[1:-1]
                        if transcript_sub_acc not in transcripts_str:
                            transcripts_sub_acc.append(transcript_sub_acc)

                        transcript_sub_don = transcript_comma_edges.replace(",{} {},".format(str(acc), str(don)), ",{} {},".format(str(acc), str(don_sub)))
                        # remove ',' from edges 
                        transcript_sub_don = transcript_sub_don[1:-1]
                        if transcript_sub_don not in transcripts_str:
                            transcripts_sub_don.append(transcript_sub_don)

                    if (len(transcripts_sub_acc) + len(transcripts_sub_don)) >= self.max_fake_substitution_transcripts:
                        # too many fake_substitution_transcripts
                        break

                if (len(transcripts_sub_acc) + len(transcripts_sub_don)) >= self.max_fake_substitution_transcripts:
                    # too many fake_substitution_transcripts
                    break

            acc_don_sub_list = list(set(acc_don_sub_list))
            
            if self.augment_transcript_data:
                processed_transcripts_sub_acc = []
                for transcript_sub_acc in transcripts_sub_acc[:self.max_fake_substitution_transcripts]:
                    if len(transcript_sub_acc) > 0:
                        transcript_list = []
                        for exon in transcript_sub_acc.split(','):
                            exons_str.append(exon)
                            acc, don = exon.split(' ')
                            acc = int(acc)
                            don = int(don)
                            transcript_list.append((acc, don))

                        processed_transcripts_sub_acc.append(transcript_list)
                
                processed_transcripts_sub_don = []
                for transcript_sub_don in transcripts_sub_don[:self.max_fake_substitution_transcripts]:
                    if len(transcript_sub_don) > 0:
                        transcript_list = []
                        for exon in transcript_sub_don.split(','):
                            exons_str.append(exon)
                            acc, don = exon.split(' ')
                            acc = int(acc)
                            don = int(don)
                            transcript_list.append((acc, don))

                        processed_transcripts_sub_don.append(transcript_list)

            if self.augment_transcript_data:
                # process fake inclusion transcripts
                fake_inclusion_transcripts = []
                for transcript in transcripts_str:
                    exons_in_transcript_str = transcript.split(',')
                    for exon_str in exons_str:
                        go_to_next_fake_exon = False
                        if exon_str not in exons_in_transcript_str:
                            fake_inclusion_acc, fake_inclusion_don = exon_str.split(' ')
                            fake_inclusion_acc = int(fake_inclusion_acc)
                            fake_inclusion_don = int(fake_inclusion_don)

                            transcript_list = []
                            prev_don = -1
                            for exon in exons_in_transcript_str:
                                exons_str.append(exon)
                                acc, don = exon.split(' ')

                                cur_acc = int(acc)
                                cur_don = int(don)
                                if fake_inclusion_acc > prev_don:
                                    if fake_inclusion_don < cur_acc:
                                        transcript_list.append((int(fake_inclusion_acc), int(fake_inclusion_don)))
                                    else: # reject and go to next fake exon if overlap occurs
                                        go_to_next_fake_exon = True
                                        break
                                    
                                transcript_list.append((cur_acc, cur_don))

                                prev_acc = cur_acc
                                prev_don = cur_don

                            if go_to_next_fake_exon:
                                break
                            else:
                                fake_inclusion_transcripts.append(transcript_list)

                    if len(fake_inclusion_transcripts) >= self.max_fake_inclusion_transcripts:
                        # enough fake_inclusion_transcripts
                        break

                
                # process fake exclusion transcripts
                fake_exclusion_transcripts = []
                break_fake_exclusion_transcripts = False
                for transcript in transcripts_str:
                    exons_in_transcript_str = transcript.split(',')
                    if len(exons_in_transcript_str) > 1:
                        for ind, exon_str in enumerate(exons_in_transcript_str):
                            # remove an exon to create a fake_exclusion_transcript
                            exons_after_exclusion = exons_in_transcript_str[:ind] + exons_in_transcript_str[ind+1:]
                            exons_after_exclusion = ",".join(exons_after_exclusion)

                            if exons_after_exclusion not in transcripts_str:
                                # fake_exclusion_transcript is not a real transcript, save it
                                transcript_list = []

                                if ',' in exons_after_exclusion:
                                    for exon in exons_after_exclusion.split(','):
                                        acc, don = exon.split(' ')
                                        acc = int(acc)
                                        don = int(don)
                                        transcript_list.append((acc, don))
                                else:
                                    acc, don = exons_after_exclusion.split(' ')
                                    acc = int(acc)
                                    don = int(don)
                                    transcript_list.append((acc, don))

                                fake_exclusion_transcripts.append(transcript_list)

                            if len(fake_exclusion_transcripts) >= self.max_fake_exclusion_transcripts:
                                # enough fake_exclusion_transcripts
                                break_fake_exclusion_transcripts = True
                                break
                        if break_fake_exclusion_transcripts:
                            break

            # process alabels and dlabels to include nt seq within its window
            new_alabels = {}
            for position in alabels.keys():
                new_alabels[int(position)] = {}
                new_alabels[int(position)]['psi'] = alabels[position]
                seq_start_ind_for_window = int(position)
                seq_end_ind_for_window = int(position) + self.context_len + 1 # + 1 to include nt at acc/don site
                windowed_seq = anno_seq[seq_start_ind_for_window:seq_end_ind_for_window]
                windowed_seq_np = np.asarray(list(map(int, list(windowed_seq))))
                windowed_seq_one_hot = one_hot_encode_input(windowed_seq_np)
                windowed_seq_one_hot = np.transpose(windowed_seq_one_hot, axes=(-1,-2)) # (L,C) -> (C,L)

                new_alabels[int(position)]['wseq'] = windowed_seq_one_hot

            new_dlabels = {}
            for position in dlabels.keys():
                new_dlabels[int(position)] = {}
                new_dlabels[int(position)]['psi'] = dlabels[position]
                seq_start_ind_for_window = int(position)
                seq_end_ind_for_window = int(position) + self.context_len + 1 # + 1 to include nt at acc/don site

                windowed_seq = anno_seq[seq_start_ind_for_window:seq_end_ind_for_window]
                windowed_seq_np = np.asarray(list(map(int, list(windowed_seq))))
                windowed_seq_one_hot = one_hot_encode_input(windowed_seq_np)
                windowed_seq_one_hot = np.transpose(windowed_seq_one_hot, axes=(-1,-2)) # (L,C) -> (C,L)
                
                new_dlabels[int(position)]['wseq'] = windowed_seq_one_hot

            # Supplement new_alabels and new_dlabels with acc and don with zero psi
            for transcript in transcripts:
                for acc, don in transcript:
                    if acc not in new_alabels.keys():
                        new_alabels[acc] = {}
                        new_alabels[acc]['psi'] = 0
                        seq_start_ind_for_window = acc
                        seq_end_ind_for_window = acc + self.context_len + 1 # + 1 to include nt at acc/don site
                        windowed_seq = anno_seq[seq_start_ind_for_window:seq_end_ind_for_window]
                        windowed_seq_np = np.asarray(list(map(int, list(windowed_seq))))
                        windowed_seq_one_hot = one_hot_encode_input(windowed_seq_np)
                        windowed_seq_one_hot = np.transpose(windowed_seq_one_hot, axes=(-1,-2)) # (L,C) -> (C,L)

                        new_alabels[acc]['wseq'] = windowed_seq_one_hot

                    if don not in new_dlabels.keys():
                        new_dlabels[don] = {}
                        new_dlabels[don]['psi'] = 0
                        seq_start_ind_for_window = don
                        seq_end_ind_for_window = don + self.context_len + 1 # + 1 to include nt at acc/don site
                        windowed_seq = anno_seq[seq_start_ind_for_window:seq_end_ind_for_window]
                        windowed_seq_np = np.asarray(list(map(int, list(windowed_seq))))
                        windowed_seq_one_hot = one_hot_encode_input(windowed_seq_np)
                        windowed_seq_one_hot = np.transpose(windowed_seq_one_hot, axes=(-1,-2)) # (L,C) -> (C,L)

                        new_dlabels[don]['wseq'] = windowed_seq_one_hot

            # Process sub_nt to include nt seq within its window            
            sub_nts = {}
            for position in acc_don_sub_list:
                sub_nts[int(position)] = {}
                seq_start_ind_for_window = int(position)
                seq_end_ind_for_window = int(position) + self.context_len + 1 # + 1 to include nt at acc/don site
                windowed_seq = anno_seq[seq_start_ind_for_window:seq_end_ind_for_window]
                windowed_seq_np = np.asarray(list(map(int, list(windowed_seq))))
                windowed_seq_one_hot = one_hot_encode_input(windowed_seq_np)
                windowed_seq_one_hot = np.transpose(windowed_seq_one_hot, axes=(-1,-2)) # (L,C) -> (C,L)
                
                sub_nts[int(position)]['wseq'] = windowed_seq_one_hot

        transcripts_splice_junctions = get_splice_junctions(transcripts, self.gene_start_token_position, self.gene_end_token_position)

        if self.augment_transcript_data:
            sub_acc_splice_junctions = get_splice_junctions(processed_transcripts_sub_acc, self.gene_start_token_position, self.gene_end_token_position)
            sub_don_splice_junctions = get_splice_junctions(processed_transcripts_sub_don, self.gene_start_token_position, self.gene_end_token_position)
            fake_inclusion_splice_junctions = get_splice_junctions(fake_inclusion_transcripts, self.gene_start_token_position, self.gene_end_token_position)
            fake_exclusion_splice_junctions = get_splice_junctions(fake_exclusion_transcripts, self.gene_start_token_position, self.gene_end_token_position)
            if self.return_sample_metadata:
                # Additionally return sample_name, gene_name
                return new_alabels, new_dlabels, transcripts_splice_junctions, act_rep_tensor, sub_nts, sub_acc_splice_junctions, sub_don_splice_junctions, fake_inclusion_splice_junctions, fake_exclusion_splice_junctions, sample_name, gene_name
            else:
                return new_alabels, new_dlabels, transcripts_splice_junctions, act_rep_tensor, sub_nts, sub_acc_splice_junctions, sub_don_splice_junctions, fake_inclusion_splice_junctions, fake_exclusion_splice_junctions
        else:
            if self.return_sample_metadata:
                # Additionally return sample_name, gene_name
                return new_alabels, new_dlabels, transcripts_splice_junctions, act_rep_tensor, sub_nts, sample_name, gene_name
            else:
                return new_alabels, new_dlabels, transcripts_splice_junctions, act_rep_tensor, sub_nts

class MultipleJsonlDatasetForTranscriptRegression(Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, args, data_dir, act_rep_file=None, augment_transcript_data=True, num_sampled_sub_per_acc_don=None, return_sample_metadata=False, epsilon=1e-7, tissue_types=None,
            patient_list=None, chr_list=None,
            contain_transcript_probs=False):
        self.contain_transcript_probs = contain_transcript_probs
        self.context_len = args.context_len
        self.no_tissue_type_as_feature = args.no_tissue_type_as_feature

        self.augment_transcript_data = augment_transcript_data
        self.return_sample_metadata = return_sample_metadata
        self.max_fake_inclusion_transcripts = args.max_fake_inclusion_transcripts
        self.max_fake_exclusion_transcripts = args.max_fake_exclusion_transcripts

        self.gene_start_token_position = args.gene_start_token_position # -1
        self.gene_end_token_position = args.gene_end_token_position # 9999999

        self.max_fake_substitution_transcripts = args.max_fake_substitution_transcripts

        # number of sampled non-acceptor/donor nt at neighborhood of each acceptor/donor
        if num_sampled_sub_per_acc_don is None:
            self.num_sampled_sub_per_acc_don = args.num_sampled_sub_per_acc_don  
        else:
            self.num_sampled_sub_per_acc_don = num_sampled_sub_per_acc_don

        self.sampled_sub_nt_distance = args.sampled_sub_nt_distance
        self.prob_sub_per_exon = args.prob_sub_per_exon

        # Set up main input files
        if tissue_types is None:
            self.tissue_types = tissue_list
        self.data_files = os.listdir(data_dir)
        if patient_list is not None or chr_list is not None:
            self.data_files = [ os.path.join(data_dir, data_file) for data_file in self.data_files if ( match_patient_chr_lists(data_file, patient_list, chr_list) and data_file[:3] in self.tissue_types )]
        else:
            self.data_files = [ os.path.join(data_dir, data_file) for data_file in self.data_files if data_file[:3] in self.tissue_types]
        list.sort(self.data_files)

        self.start_indices = [0] * len(self.data_files)
        self.sample_count = 0
        for index, data_file in enumerate(self.data_files):
            self.start_indices[index] = self.sample_count
            self.sample_count += file_len(data_file)

        # Open gene seq dict
        with open(args.gene_seq_dict_file, encoding="utf-8") as f:
            lines = f.readlines()
            self.gene_seq_dict = {}
            for line in lines:
                json_dict = json.loads(line)
                for key in json_dict.keys():
                    self.gene_seq_dict[key] = json_dict[key]
        logger.info(" {} gene sequences in records".format(str(len(self.gene_seq_dict))))

        # Process act_rep csv input file
        # Load training act_rep csv to compute train set stats
        train_df = pd.read_csv(args.act_rep_train_file)
        train_df = train_df.set_index('samples')
        print("train_df: ", train_df)
        if args.c_log_act_rep is None:
            # use half of min non-zero value to transform data before log op
            c_log_act_rep = train_df[train_df>0].min().min() / 2
        else:
            c_log_act_rep = args.c_log_act_rep
        if args.log_act_rep:
            train_df = np.log(train_df.add(c_log_act_rep))
            print("log train_df: ", train_df)
        self.train_act_rep_means = {}
        self.train_act_rep_stds = {}
        for col in train_df.columns:
            if 'ENST' in col:
                self.train_act_rep_means[col] = train_df[col].mean()
                self.train_act_rep_stds[col] = train_df[col].std()

        if act_rep_file is None:
            act_rep_file = args.act_rep_train_file

        # Open and process act_rep_file
        df = pd.read_csv(act_rep_file)
        df = df.set_index('samples')
        if args.log_act_rep:
            df = np.log(df.add(c_log_act_rep))
        for col in df.columns:
            if 'ENST' in col:
                # normalize data according to train stats
                df[col] = (df[col] - self.train_act_rep_means[col]) / (self.train_act_rep_stds[col] + epsilon)

        self.act_rep_df = df

    def __len__(self):
        return self.sample_count

    def __getitem__(self, index):
        file_index = bisect(self.start_indices, index) - 1

        index_in_file = index - self.start_indices[file_index]
        data_file = self.data_files[file_index]

        # retrieve act_rep data
        sample_filename = data_file.split('/')[-1]
        sample_name_chr = sample_filename.split('.')[0]
        sample_name = sample_name_chr.split('_')[0]
        act_rep = self.act_rep_df.loc[sample_name, :]
        act_rep_np = act_rep.values

        # Get one-hot encoding of cell type with sample_name
        if not self.no_tissue_type_as_feature:
            cell_type = sample_name[:3]
            cell_type_onehot = np.zeros(len(tissue_list))
            if cell_type.upper() in tissue_list:
                cell_type_ind = tissue_list.index(cell_type.upper())
                cell_type_onehot[cell_type_ind] = 1
            act_rep_np = np.concatenate([act_rep_np, cell_type_onehot], axis=0)

        act_rep_tensor = torch.tensor(act_rep_np, dtype=torch.float)

        # retrieve main gene input data
        with open(data_file, encoding="utf-8") as f:
            lines = f.readlines()
            jsonl = lines[index_in_file]
            json_dict = json.loads(jsonl)
            gene_name = json_dict['gene']
            seq = self.gene_seq_dict[gene_name]
            main_seq_len = len(seq) - self.context_len
            anno_seq = seq.upper().replace('A', '1').replace('C', '2')
            anno_seq = anno_seq.replace('G', '3').replace('T', '4').replace('N', '0')
            alabels = json_dict['alabels']
            dlabels = json_dict['dlabels']
            if self.contain_transcript_probs:
                tlabels = json_dict['transcript_probs']
            exons = json_dict['exons']

            # Process input data here
            # process transcript data as lists of exon tuples
            transcripts = []
            exons_str = []
            acc_don_list = []
            transcripts_str = exons.split(";")[:-1]
            random.shuffle(transcripts_str)
            for transcript in transcripts_str:
                if len(transcript) > 0:
                    transcript_list = []
                    for exon in transcript.split(','):
                        exons_str.append(exon)
                        acc, don = exon.split(' ')
                        acc = int(acc)
                        don = int(don)
                        transcript_list.append((acc, don))
                        acc_don_list += [acc, don]

                    transcripts.append(transcript_list)

            exons_str = list(set(exons_str)) # compute unique exons
            num_exons = len(exons_str)

            if self.max_fake_substitution_transcripts < num_exons:
                transcript_sub_nt_sample_prob = 1
            else:
                transcript_sub_nt_sample_prob = self.max_fake_substitution_transcripts / num_exons

            prob_sub_per_exon = min(self.prob_sub_per_exon, transcript_sub_nt_sample_prob)

            # process non-acceptor, donor nt for fake_substitution_transcripts
            fake_substitution_transcripts = []
            processed_exons_sub = []
            acc_don_sub_list = []
            transcripts_sub_acc = []
            transcripts_sub_don = []
            break_fake_substitution_transcripts = False
            
            for transcript in transcripts_str:
                exons_in_transcript_str = transcript.split(',')
                for exon_ind, exon in enumerate(exons_in_transcript_str):
                    acc, don = exon.split(' ')
                    processed_exons_sub.append(exon)
                    acc = int(acc)
                    don = int(don)

                    if self.num_sampled_sub_per_acc_don > 1:
                        for i in range(int(self.num_sampled_sub_per_acc_don)):
                            # sample acc_sub from neighborhood of acc
                            acc_sub = random.randint(max(0, acc-self.sampled_sub_nt_distance), min(don, acc+self.sampled_sub_nt_distance))
                            # resample acc_sub if similar to acc or already sampled before in acc_don_sub_list
                            while acc_sub == acc:
                                acc_sub = random.randint(max(0, acc-self.sampled_sub_nt_distance), min(don, acc+self.sampled_sub_nt_distance))

                            # sample don_sub from neighborhood of don
                            don_sub = random.randint(max(acc, don-self.sampled_sub_nt_distance), min(main_seq_len-1, don+self.sampled_sub_nt_distance))
                            # resample don_sub if similar to don
                            while don_sub == don:
                                don_sub = random.randint(max(acc, don-self.sampled_sub_nt_distance), min(main_seq_len-1, don+self.sampled_sub_nt_distance))

                            acc_don_sub_list += [acc_sub, don_sub]

                            # create fake_substitution_transcript 
                            if i == 0 and random.random() < prob_sub_per_exon:
                                transcript_comma_edges = ',' + transcript + ','
                                transcript_sub_acc = transcript_comma_edges.replace(",{} {},".format(str(acc), str(don)), ",{} {},".format(str(acc_sub), str(don)))
                                # remove ',' from edges 
                                transcript_sub_acc = transcript_sub_acc[1:-1]
                                if transcript_sub_acc not in transcripts_str:
                                    transcripts_sub_acc.append(transcript_sub_acc)

                                transcript_sub_don = transcript_comma_edges.replace(",{} {},".format(str(acc), str(don)), ",{} {},".format(str(acc), str(don_sub)))
                                # remove ',' from edges 
                                transcript_sub_don = transcript_sub_don[1:-1]
                                if transcript_sub_don not in transcripts_str:
                                    transcripts_sub_don.append(transcript_sub_don)

                            if (len(transcripts_sub_acc) + len(transcripts_sub_don)) >= self.max_fake_substitution_transcripts:
                                # too many fake_substitution_transcripts
                                break

                    elif self.num_sampled_sub_per_acc_don > random.random():
                        # sample acc_sub from neighborhood of acc
                        acc_sub = random.randint(max(0, acc-self.sampled_sub_nt_distance), min(don, acc+self.sampled_sub_nt_distance))
                        # resample acc_sub if similar to acc or already sampled before in acc_don_sub_list
                        while acc_sub == acc:
                            acc_sub = random.randint(max(0, acc-self.sampled_sub_nt_distance), min(don, acc+self.sampled_sub_nt_distance))

                        # sample don_sub from neighborhood of don
                        don_sub = random.randint(max(acc, don-self.sampled_sub_nt_distance), min(main_seq_len-1, don+self.sampled_sub_nt_distance))
                        # resample don_sub if similar to don
                        while don_sub == don:
                            don_sub = random.randint(max(acc, don-self.sampled_sub_nt_distance), min(main_seq_len-1, don+self.sampled_sub_nt_distance))

                        acc_don_sub_list += [acc_sub, don_sub]

                        # create fake_substitution_transcript 
                        transcript_comma_edges = ',' + transcript + ','
                        transcript_sub_acc = transcript_comma_edges.replace(",{} {},".format(str(acc), str(don)), ",{} {},".format(str(acc_sub), str(don)))
                        # remove ',' from edges 
                        transcript_sub_acc = transcript_sub_acc[1:-1]
                        if transcript_sub_acc not in transcripts_str:
                            transcripts_sub_acc.append(transcript_sub_acc)

                        transcript_sub_don = transcript_comma_edges.replace(",{} {},".format(str(acc), str(don)), ",{} {},".format(str(acc), str(don_sub)))
                        # remove ',' from edges 
                        transcript_sub_don = transcript_sub_don[1:-1]
                        if transcript_sub_don not in transcripts_str:
                            transcripts_sub_don.append(transcript_sub_don)

                    if (len(transcripts_sub_acc) + len(transcripts_sub_don)) >= self.max_fake_substitution_transcripts:
                        # too many fake_substitution_transcripts
                        break

                if (len(transcripts_sub_acc) + len(transcripts_sub_don)) >= self.max_fake_substitution_transcripts:
                    # too many fake_substitution_transcripts
                    break

            acc_don_sub_list = list(set(acc_don_sub_list))
            
            if self.augment_transcript_data:
                processed_transcripts_sub_acc = []
                for transcript_sub_acc in transcripts_sub_acc[:self.max_fake_substitution_transcripts]:
                    if len(transcript_sub_acc) > 0:
                        transcript_list = []
                        for exon in transcript_sub_acc.split(','):
                            exons_str.append(exon)
                            acc, don = exon.split(' ')
                            acc = int(acc)
                            don = int(don)
                            transcript_list.append((acc, don))

                        processed_transcripts_sub_acc.append(transcript_list)
                
                processed_transcripts_sub_don = []
                for transcript_sub_don in transcripts_sub_don[:self.max_fake_substitution_transcripts]:
                    if len(transcript_sub_don) > 0:
                        transcript_list = []
                        for exon in transcript_sub_don.split(','):
                            exons_str.append(exon)
                            acc, don = exon.split(' ')
                            acc = int(acc)
                            don = int(don)
                            transcript_list.append((acc, don))

                        processed_transcripts_sub_don.append(transcript_list)

            if self.augment_transcript_data:
                # process fake inclusion transcripts
                fake_inclusion_transcripts = []
                for transcript in transcripts_str:
                    exons_in_transcript_str = transcript.split(',')
                    for exon_str in exons_str:
                        go_to_next_fake_exon = False
                        if exon_str not in exons_in_transcript_str:
                            fake_inclusion_acc, fake_inclusion_don = exon_str.split(' ')
                            fake_inclusion_acc = int(fake_inclusion_acc)
                            fake_inclusion_don = int(fake_inclusion_don)

                            transcript_list = []
                            prev_don = -1
                            for exon in exons_in_transcript_str:
                                exons_str.append(exon)
                                acc, don = exon.split(' ')

                                cur_acc = int(acc)
                                cur_don = int(don)
                                if fake_inclusion_acc > prev_don:
                                    if fake_inclusion_don < cur_acc:
                                        transcript_list.append((int(fake_inclusion_acc), int(fake_inclusion_don)))
                                    else: # reject and go to next fake exon if overlap occurs
                                        go_to_next_fake_exon = True
                                        break
                                    
                                transcript_list.append((cur_acc, cur_don))

                                prev_acc = cur_acc
                                prev_don = cur_don

                            if go_to_next_fake_exon:
                                break
                            else:
                                fake_inclusion_transcripts.append(transcript_list)

                    if len(fake_inclusion_transcripts) >= self.max_fake_inclusion_transcripts:
                        # enough fake_inclusion_transcripts
                        break
                
                # process fake exclusion transcripts
                fake_exclusion_transcripts = []
                break_fake_exclusion_transcripts = False
                for transcript in transcripts_str:
                    exons_in_transcript_str = transcript.split(',')
                    if len(exons_in_transcript_str) > 1:
                        for ind, exon_str in enumerate(exons_in_transcript_str):
                            # remove an exon to create a fake_exclusion_transcript
                            exons_after_exclusion = exons_in_transcript_str[:ind] + exons_in_transcript_str[ind+1:]
                            exons_after_exclusion = ",".join(exons_after_exclusion)

                            if exons_after_exclusion not in transcripts_str:
                                # fake_exclusion_transcript is not a real transcript, save it
                                transcript_list = []

                                if ',' in exons_after_exclusion:
                                    for exon in exons_after_exclusion.split(','):
                                        acc, don = exon.split(' ')
                                        acc = int(acc)
                                        don = int(don)
                                        transcript_list.append((acc, don))
                                else:
                                    acc, don = exons_after_exclusion.split(' ')
                                    acc = int(acc)
                                    don = int(don)
                                    transcript_list.append((acc, don))

                                fake_exclusion_transcripts.append(transcript_list)

                            if len(fake_exclusion_transcripts) >= self.max_fake_exclusion_transcripts:
                                # enough fake_exclusion_transcripts
                                break_fake_exclusion_transcripts = True
                                break
                        if break_fake_exclusion_transcripts:
                            break

            # process alabels and dlabels to include nt seq within its window
            new_alabels = {}
            for position in alabels.keys():
                new_alabels[int(position)] = {}
                new_alabels[int(position)]['psi'] = alabels[position]
                seq_start_ind_for_window = int(position)
                seq_end_ind_for_window = int(position) + self.context_len + 1 # + 1 to include nt at acc/don site
                windowed_seq = anno_seq[seq_start_ind_for_window:seq_end_ind_for_window]
                windowed_seq_np = np.asarray(list(map(int, list(windowed_seq))))
                windowed_seq_one_hot = one_hot_encode_input(windowed_seq_np)
                windowed_seq_one_hot = np.transpose(windowed_seq_one_hot, axes=(-1,-2)) # (L,C) -> (C,L)

                new_alabels[int(position)]['wseq'] = windowed_seq_one_hot

            new_dlabels = {}
            for position in dlabels.keys():
                new_dlabels[int(position)] = {}
                new_dlabels[int(position)]['psi'] = dlabels[position]
                seq_start_ind_for_window = int(position)
                seq_end_ind_for_window = int(position) + self.context_len + 1 # + 1 to include nt at acc/don site

                windowed_seq = anno_seq[seq_start_ind_for_window:seq_end_ind_for_window]
                windowed_seq_np = np.asarray(list(map(int, list(windowed_seq))))
                windowed_seq_one_hot = one_hot_encode_input(windowed_seq_np)
                windowed_seq_one_hot = np.transpose(windowed_seq_one_hot, axes=(-1,-2)) # (L,C) -> (C,L)
                
                new_dlabels[int(position)]['wseq'] = windowed_seq_one_hot

            # Supplement new_alabels and new_dlabels with acc and don with zero psi
            for transcript in transcripts:
                for acc, don in transcript:
                    if acc not in new_alabels.keys():
                        new_alabels[acc] = {}
                        new_alabels[acc]['psi'] = 0
                        seq_start_ind_for_window = acc
                        seq_end_ind_for_window = acc + self.context_len + 1 # + 1 to include nt at acc/don site
                        windowed_seq = anno_seq[seq_start_ind_for_window:seq_end_ind_for_window]
                        windowed_seq_np = np.asarray(list(map(int, list(windowed_seq))))
                        windowed_seq_one_hot = one_hot_encode_input(windowed_seq_np)
                        windowed_seq_one_hot = np.transpose(windowed_seq_one_hot, axes=(-1,-2)) # (L,C) -> (C,L)

                        new_alabels[acc]['wseq'] = windowed_seq_one_hot

                    if don not in new_dlabels.keys():
                        new_dlabels[don] = {}
                        new_dlabels[don]['psi'] = 0
                        seq_start_ind_for_window = don
                        seq_end_ind_for_window = don + self.context_len + 1 # + 1 to include nt at acc/don site
                        windowed_seq = anno_seq[seq_start_ind_for_window:seq_end_ind_for_window]
                        windowed_seq_np = np.asarray(list(map(int, list(windowed_seq))))
                        windowed_seq_one_hot = one_hot_encode_input(windowed_seq_np)
                        windowed_seq_one_hot = np.transpose(windowed_seq_one_hot, axes=(-1,-2)) # (L,C) -> (C,L)

                        new_dlabels[don]['wseq'] = windowed_seq_one_hot

            # Process sub_nt to include nt seq within its window            
            sub_nts = {}
            for position in acc_don_sub_list:
                sub_nts[int(position)] = {}
                seq_start_ind_for_window = int(position)
                seq_end_ind_for_window = int(position) + self.context_len + 1 # + 1 to include nt at acc/don site
                windowed_seq = anno_seq[seq_start_ind_for_window:seq_end_ind_for_window]
                windowed_seq_np = np.asarray(list(map(int, list(windowed_seq))))
                windowed_seq_one_hot = one_hot_encode_input(windowed_seq_np)
                windowed_seq_one_hot = np.transpose(windowed_seq_one_hot, axes=(-1,-2)) # (L,C) -> (C,L)
                
                sub_nts[int(position)]['wseq'] = windowed_seq_one_hot

        transcripts_splice_junctions = get_splice_junctions(transcripts, self.gene_start_token_position, self.gene_end_token_position)

        if self.augment_transcript_data:
            sub_acc_splice_junctions = get_splice_junctions(processed_transcripts_sub_acc, self.gene_start_token_position, self.gene_end_token_position)
            sub_don_splice_junctions = get_splice_junctions(processed_transcripts_sub_don, self.gene_start_token_position, self.gene_end_token_position)
            fake_inclusion_splice_junctions = get_splice_junctions(fake_inclusion_transcripts, self.gene_start_token_position, self.gene_end_token_position)
            fake_exclusion_splice_junctions = get_splice_junctions(fake_exclusion_transcripts, self.gene_start_token_position, self.gene_end_token_position)
            if self.contain_transcript_probs:
                if self.return_sample_metadata:
                    # Additionally return sample_name, gene_name
                    return new_alabels, new_dlabels, tlabels, transcripts_splice_junctions, act_rep_tensor, sub_nts, sub_acc_splice_junctions, sub_don_splice_junctions, fake_inclusion_splice_junctions, fake_exclusion_splice_junctions, sample_name, gene_name
                else:
                    return new_alabels, new_dlabels, tlabels, transcripts_splice_junctions, act_rep_tensor, sub_nts, sub_acc_splice_junctions, sub_don_splice_junctions, fake_inclusion_splice_junctions, fake_exclusion_splice_junctions
            else:
                if self.return_sample_metadata:
                    # Additionally return sample_name, gene_name
                    return new_alabels, new_dlabels, transcripts_splice_junctions, act_rep_tensor, sub_nts, sub_acc_splice_junctions, sub_don_splice_junctions, fake_inclusion_splice_junctions, fake_exclusion_splice_junctions, sample_name, gene_name
                else:
                    return new_alabels, new_dlabels, transcripts_splice_junctions, act_rep_tensor, sub_nts, sub_acc_splice_junctions, sub_don_splice_junctions, fake_inclusion_splice_junctions, fake_exclusion_splice_junctions
        else:
            if self.contain_transcript_probs:
                if self.return_sample_metadata:
                    # Additionally return sample_name, gene_name
                    return new_alabels, new_dlabels, tlabels, transcripts_splice_junctions, act_rep_tensor, sub_nts, sample_name, gene_name
                else:
                    return new_alabels, new_dlabels, tlabels, transcripts_splice_junctions, act_rep_tensor, sub_nts
            else:
                if self.return_sample_metadata:
                    # Additionally return sample_name, gene_name
                    return new_alabels, new_dlabels, transcripts_splice_junctions, act_rep_tensor, sub_nts, sample_name, gene_name
                else:
                    return new_alabels, new_dlabels, transcripts_splice_junctions, act_rep_tensor, sub_nts

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

def _clear_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)

    for checkpoint in checkpoints_sorted:
        logger.info("Deleting older checkpoint [{}] before rerunning training".format(checkpoint))
        shutil.rmtree(checkpoint)


# https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/22
def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y
    # y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot
    # return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot

def num_correct(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def obj_k_value(ind):
    def get_k_value_from_obj(obj):
        return obj[ind]
    return get_k_value_from_obj
    
def train_spliceai(args, train_dataset, spliceai_model, site_aux_model=None, ablation_finetune_model=None, valid_dataset=None, do_train_spliceai=True, do_train_site_aux_model=True, train_epochs=None, train_steps=None, eval_at_init=False) -> Tuple[int, float]:
    """ Train the spliceai_model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join(args.output_dir, 'train_spliceai_log'))

    if args.per_gpu_train_spliceai_batch_size <= 0:
        args.per_gpu_train_spliceai_batch_size = args.per_gpu_train_batch_size
    args.train_spliceai_batch_size = args.per_gpu_train_spliceai_batch_size * max(1, args.n_gpu)

    def collate(examples: List[torch.Tensor]):
        seq_inputs_list = list(map(obj_k_value(0), examples))
        act_rep_inputs_list = list(map(obj_k_value(1), examples))
        labels_class_list = list(map(obj_k_value(2), examples))
        labels_reg_list = list(map(obj_k_value(3), examples))
        labels_len_list = list(map(len, labels_class_list))

        # seq_inputs, act_rep_inputs, labels_class, labels_reg
        padded_seq_inputs = pad_sequence(seq_inputs_list, batch_first=True, padding_value=0)
        padded_seq_inputs = torch.transpose(padded_seq_inputs, -1,-2)
        batch_act_rep_inputs = torch.stack(act_rep_inputs_list)
        padded_labels_class = pad_sequence(labels_class_list, batch_first=True, padding_value=0)
        padded_labels_reg = pad_sequence(labels_reg_list, batch_first=True, padding_value=0)
        
        # Build label mask to mask out loss values for padded regions
        label_masks = torch.ones_like(padded_labels_class)
        for ind, label_len in enumerate(labels_len_list):
            label_masks[ind, label_len:] = 0

        return padded_seq_inputs, batch_act_rep_inputs, padded_labels_class, padded_labels_reg, label_masks

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        # train_dataset, sampler=train_sampler, batch_size=args.train_spliceai_batch_size
        train_dataset, sampler=train_sampler, batch_size=args.train_spliceai_batch_size, collate_fn=collate
    )

    if train_epochs is None:
        train_epochs = args.num_train_spliceai_epochs
    if train_steps is None:
        train_steps = args.spliceai_max_steps

    if train_steps > 0:
        t_total = train_steps
        train_epochs = train_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    
    optimizer_grouped_parameters = []
    print("do_train_spliceai: ", do_train_spliceai)
    if do_train_spliceai:
        optimizer_grouped_parameters = optimizer_grouped_parameters + [
            {
                "params": [p for n, p in spliceai_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in spliceai_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

    if do_train_site_aux_model:
        optimizer_grouped_parameters = optimizer_grouped_parameters + [
            {
                "params": [p for n, p in site_aux_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in site_aux_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

    if ablation_finetune_model:
        optimizer_grouped_parameters = optimizer_grouped_parameters + [
            {
                "params": [p for n, p in ablation_finetune_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in ablation_finetune_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

    # Change opt scheduler setup to match spliceai baseline's
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    
    if args.train_spliceai_opt_scheduler_type == 'epoch':
        scheduler = get_exponential_decay_schedule(
            optimizer, num_constant_steps=args.num_constant_lr_epochs_spliceai # num_constant_lr_epochs_spliceai = 6 in spliceai paper
        )
    elif args.train_spliceai_opt_scheduler_type == 'step':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps_spliceai, num_training_steps=t_total
        )        
    else:
        raise ValueError(
            "Invalid train_spliceai_opt_scheduler_type, must be either 'epoch' or 'step'."
        )

    # Check if saved optimizer or scheduler states exist
    if (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        logger.info("***** Loading optimizer & scheduler *****")

        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        if do_train_site_aux_model:
            [spliceai_model, site_aux_model], optimizer = amp.initialize([spliceai_model, site_aux_model], optimizer, opt_level=args.fp16_opt_level)
        if ablation_finetune_model:
            ablation_finetune_model, optimizer = amp.initialize(ablation_finetune_model, optimizer, opt_level=args.fp16_opt_level)
        else:
            spliceai_model, optimizer = amp.initialize(spliceai_model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1 and not isinstance(spliceai_model, torch.nn.DataParallel):
        spliceai_model = torch.nn.DataParallel(spliceai_model)
    if args.n_gpu > 1 and not isinstance(site_aux_model, torch.nn.DataParallel):
        site_aux_model = torch.nn.DataParallel(site_aux_model)
    if ablation_finetune_model and args.n_gpu > 1 and not isinstance(ablation_finetune_model, torch.nn.DataParallel):
        ablation_finetune_model = torch.nn.DataParallel(ablation_finetune_model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        spliceai_model = torch.nn.parallel.DistributedDataParallel(
            spliceai_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
        site_aux_model = torch.nn.parallel.DistributedDataParallel(
            site_aux_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
        if ablation_finetune_model:
            ablation_finetune_model = torch.nn.parallel.DistributedDataParallel(
                ablation_finetune_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
            )

    # Train!
    logger.info("***** Running spliceAI training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_spliceai_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from spliceai_model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0

    spliceai_model.zero_grad()
    site_aux_model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    first_save = True

    if args.local_rank in [-1, 0] and args.logging_valid_steps > 0 and eval_at_init:
        # Log metrics
        if (
            args.local_rank == -1 and valid_dataset is not None
        ):  # Only evaluate when single GPU otherwise metrics may not average well
            logger.info("evaluate_spliceai before training")
            if args.lambda_loss_class > 0:
                results = evaluate_spliceai(args, spliceai_model, site_aux_model, valid_dataset, eval_accuracy=True, eval_correlation=True, 
                            max_eval_step=args.max_valid_step, eval_output_filename='valid_results.txt', ablation_finetune_model=ablation_finetune_model)
            else:
                results = evaluate_spliceai(args, spliceai_model, site_aux_model, valid_dataset, eval_accuracy=False, eval_correlation=True, 
                            max_eval_step=args.max_valid_step, eval_output_filename='valid_results.txt', ablation_finetune_model=ablation_finetune_model)
            # results = evaluate_spliceai(args, spliceai_model, site_aux_model, valid_dataset, eval_correlation=do_train_site_aux_model)
            for key, value in results.items():
                tb_writer.add_scalar("valid_spliceai_{}".format(key), value, 0)
                
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])

        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            seq_inputs, act_rep_inputs, labels_class, labels_reg, label_masks = batch

            seq_inputs = seq_inputs.to(args.device)
            act_rep_inputs = act_rep_inputs.to(args.device)
            labels_class = labels_class.to(args.device)
            labels_reg = labels_reg.to(args.device)
            label_masks = label_masks.to(args.device)

            # outputs: (loss_class, logits_class, site_hidden)
            if do_train_spliceai:
                spliceai_model.train()
                outputs = spliceai_model(seq_inputs, labels_class=labels_class, label_masks=label_masks, 
                                            none_class_reweight_factor=args.none_class_reweight_factor, dynamic_reweight_none_class=args.spliceai_dynamic_reweight_none_class, keep_none_cls_prob=args.keep_none_cls_prob)
            else:
                with torch.no_grad():
                    spliceai_model.eval()
                    outputs = spliceai_model(seq_inputs, labels_class=labels_class, label_masks=label_masks, 
                                                none_class_reweight_factor=args.none_class_reweight_factor, dynamic_reweight_none_class=args.spliceai_dynamic_reweight_none_class, keep_none_cls_prob=args.keep_none_cls_prob)
            loss_class, site_hidden = outputs[0], outputs[2]

            if do_train_site_aux_model and args.lambda_loss_reg > 0:
                # site_aux_model_outputs: loss_reg, psi_reg, site_act_rep_final_hidden
                site_aux_model.train()
                site_aux_model_outputs = site_aux_model(site_hidden, act_rep_input=act_rep_inputs, labels_class=labels_class, labels_reg=labels_reg, label_masks=label_masks)
                loss_reg = site_aux_model_outputs[0]
                
                if args.lambda_loss_class > 0 and do_train_spliceai:
                    loss = args.lambda_loss_class * loss_class + args.lambda_loss_reg * loss_reg
                else:
                    loss = args.lambda_loss_reg * loss_reg
            elif ablation_finetune_model:
                # site_aux_model_outputs: loss_reg, psi_reg, site_act_rep_final_hidden
                with torch.no_grad():
                    site_aux_model.eval()
                    site_aux_model_outputs = site_aux_model(site_hidden, act_rep_input=act_rep_inputs)
                site_act_rep_final_hidden = site_aux_model_outputs[-1]

                ablation_finetune_model.train()
                loss_reg, psi_reg = ablation_finetune_model(site_act_rep_final_hidden, labels_class=labels_class, labels_reg=labels_reg, label_masks=label_masks)

                if args.lambda_loss_class > 0 and do_train_spliceai:
                    loss = args.lambda_loss_class * loss_class + args.lambda_loss_reg * loss_reg
                else:
                    loss = args.lambda_loss_reg * loss_reg

            else:
                loss = args.lambda_loss_class * loss_class

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    if do_train_spliceai:
                        torch.nn.utils.clip_grad_norm_(spliceai_model.parameters(), args.max_grad_norm)
                    if do_train_site_aux_model:
                        torch.nn.utils.clip_grad_norm_(site_aux_model.parameters(), args.max_grad_norm)
                    if ablation_finetune_model:
                        torch.nn.utils.clip_grad_norm_(ablation_finetune_model.parameters(), args.max_grad_norm)
                optimizer.step()

                if args.train_spliceai_opt_scheduler_type == 'step':
                    scheduler.step()  # Update learning rate schedule

                if do_train_spliceai:
                    spliceai_model.zero_grad()
                if do_train_site_aux_model:
                    site_aux_model.zero_grad()
                if ablation_finetune_model:
                    ablation_finetune_model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar("spliceai/lr", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar("spliceai/loss_class", torch.mean(loss_class), global_step)
                    if do_train_site_aux_model:
                        tb_writer.add_scalar("spliceai/loss_reg", torch.mean(loss_reg), global_step)
                    tb_writer.add_scalar("spliceai/loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                    # Compute classification accuracy
                    logits_class = outputs[1]
                    probs_class = nn.functional.softmax(logits_class, dim=1)
                    _, pred_class = torch.max(probs_class.data, dim=1)

                    correct_cls = 0
                    correct_none_cls = 0
                    correct_acc_cls = 0
                    correct_don_cls = 0
                    num_sites = 0
                    num_none = 0
                    num_acc = 0
                    num_don = 0

                    for ind, seq_label_mask in enumerate(label_masks):
                        seq_len = torch.sum(seq_label_mask)

                        seq_pred_class = pred_class[ind, :seq_len]
                        seq_labels_class = labels_class[ind, :seq_len]
                        seq_correct = (seq_pred_class == seq_labels_class)

                        seq_none = (seq_labels_class == 0)
                        seq_acc = (seq_labels_class == 1)
                        seq_don = (seq_labels_class == 2)

                        correct_cls += seq_correct.sum().item()
                        correct_none_cls += seq_correct[seq_none].sum().item()
                        correct_acc_cls += seq_correct[seq_acc].sum().item()
                        correct_don_cls += seq_correct[seq_don].sum().item()
                        
                        num_sites += seq_len.item()
                        num_none += seq_none.sum().item()
                        num_acc += seq_acc.sum().item()
                        num_don += seq_don.sum().item()
                    
                    # Compute simple acc
                    accuracy_cls = correct_cls / num_sites
                    accuracy_none_cls = correct_none_cls / num_none
                    accuracy_acc_cls = correct_acc_cls / num_acc
                    accuracy_don_cls = correct_don_cls / num_don
                    tb_writer.add_scalar("spliceai/accuracy_cls", accuracy_cls, global_step)
                    tb_writer.add_scalar("spliceai/accuracy_none_cls", accuracy_none_cls, global_step)
                    tb_writer.add_scalar("spliceai/accuracy_acc_cls", accuracy_acc_cls, global_step)
                    tb_writer.add_scalar("spliceai/accuracy_don_cls", accuracy_don_cls, global_step)

                if args.local_rank in [-1, 0] and args.logging_valid_steps > 0 and global_step % args.logging_valid_steps == 0:
                    # Log metrics
                    if (
                        args.local_rank == -1 and valid_dataset is not None
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        logger.info("evaluate_spliceai")
                        if args.lambda_loss_class > 0:
                            results = evaluate_spliceai(args, spliceai_model, site_aux_model, valid_dataset, eval_accuracy=True, eval_correlation=True, 
                                        max_eval_step=args.max_valid_step, eval_output_filename='valid_results.txt', ablation_finetune_model=ablation_finetune_model)
                        else:
                            results = evaluate_spliceai(args, spliceai_model, site_aux_model, valid_dataset, eval_accuracy=False, eval_correlation=True, 
                                        max_eval_step=args.max_valid_step, eval_output_filename='valid_results.txt', ablation_finetune_model=ablation_finetune_model)
                        for key, value in results.items():
                            tb_writer.add_scalar("valid_spliceai_{}".format(key), value, global_step)

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "spliceai_train_checkpoint"
                    if first_save:
                        _clear_checkpoints(args, checkpoint_prefix)
                        first_save = False

                    # Save spliceai_model checkpoint
                    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    os.makedirs(output_dir, exist_ok=True)

                    if do_train_spliceai:
                        model_to_save = (
                            spliceai_model.module if hasattr(spliceai_model, "module") else spliceai_model
                        )  # Take care of distributed/parallel training
                        
                        # Save spliceai_model weights
                        output_model_file = os.path.join(output_dir, "spliceai_pytorch_model.bin")

                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Spliceai model weights saved in {}".format(output_model_file))

                    if do_train_site_aux_model:
                        # Save site_aux_model checkpoint
                        model_to_save = (
                            site_aux_model.module if hasattr(site_aux_model, "module") else site_aux_model
                        )  # Take care of distributed/parallel training
                        
                        # Save site_aux_model weights
                        output_model_file = os.path.join(output_dir, "site_aux_pytorch_model.bin")

                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Site aux model weights saved in {}".format(output_model_file))

                    if ablation_finetune_model:
                        # Save ablation_finetune_model checkpoint
                        model_to_save = (
                            ablation_finetune_model.module if hasattr(ablation_finetune_model, "module") else ablation_finetune_model
                        )  # Take care of distributed/parallel training
                        
                        # Save ablation_finetune_model weights
                        output_model_file = os.path.join(output_dir, "ablation_finetune_model_pytorch_model.bin")

                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("ablation_finetune_model weights saved in {}".format(output_model_file))

                    # Save training args
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving spliceai_model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    # Save optimizer and scheduler
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving spliceai optimizer and scheduler states to %s", output_dir)

            if train_steps > 0 and global_step > train_steps:
                epoch_iterator.close()
                break
        
        logger.info("End of spliceAI training epoch, global_step: %d", global_step)
        print("End of spliceAI training epoch, global_step: %d", global_step)

        if args.train_spliceai_opt_scheduler_type == 'epoch':
            scheduler.step()  # Update learning rate schedule after each epoch
        if train_steps > 0 and global_step > train_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

        # Save spliceai_model
        os.makedirs(args.output_dir, exist_ok=True)
        if do_train_spliceai:
            model_to_save = (
                spliceai_model.module if hasattr(spliceai_model, "module") else spliceai_model
            )  # Take care of distributed/parallel training
            output_model_file = os.path.join(args.output_dir, "spliceai_pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)
            logger.info("Spliceai model weights saved in {}".format(output_model_file))

        if do_train_site_aux_model:
            # Save site_aux_model
            model_to_save = (
                site_aux_model.module if hasattr(site_aux_model, "module") else site_aux_model
            )  # Take care of distributed/parallel training                    
            output_model_file = os.path.join(args.output_dir, "site_aux_pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)
            logger.info("Site aux model weights saved in {}".format(output_model_file))

        if ablation_finetune_model:
            # Save ablation_finetune_model
            model_to_save = (
                ablation_finetune_model.module if hasattr(ablation_finetune_model, "module") else ablation_finetune_model
            )  # Take care of distributed/parallel training                    
            output_model_file = os.path.join(args.output_dir, "ablation_finetune_model_pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)
            logger.info("ablation_finetune_model weights saved in {}".format(output_model_file))

        # Save training args
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        logger.info("Saving spliceai_model checkpoint to %s", args.output_dir)

        # Save optimizer and scheduler
        torch.save(optimizer.state_dict(), os.path.join(args.output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(args.output_dir, "scheduler.pt"))
        logger.info("Saving spliceai optimizer and scheduler states to %s", args.output_dir)

    return global_step, tr_loss / global_step

def infer_and_save_spliceai_final_hidden(args, infer_dataset, spliceai_model, site_aux_model, cache_dir=None, cache_both_spliceai_n_siteaux_reps=True):
    """ Compute spliceai_model and site_aux_model output states and save them """
    
    cache_file_path = os.path.join(cache_dir, 'spliceai_sequence_representations.hdf5')
    if os.path.isfile(cache_file_path) and args.overwrite_spliceai_cache == False:
        raise ValueError(
            "Output cache file ({}) already exists and is not empty. Use --overwrite_spliceai_cache to overcome.".format(
                cache_file_path
            ))
    else:
        with h5py.File(cache_file_path, 'w') as f:
            metadata = {'datetime': time.time()}
            f.attrs.update(metadata)

    if cache_both_spliceai_n_siteaux_reps:
        site_aux_model_rep_cache_file_path = os.path.join(cache_dir, 'site_aux_model_sequence_representations.hdf5')    
        if os.path.isfile(site_aux_model_rep_cache_file_path) and args.overwrite_spliceai_cache == False:
            raise ValueError(
                "Output cache file ({}) already exists and is not empty. Use --overwrite_spliceai_cache to overcome.".format(
                    site_aux_model_rep_cache_file_path
                ))
        else:
            with h5py.File(site_aux_model_rep_cache_file_path, 'w') as f:
                metadata = {'datetime': time.time()}
                f.attrs.update(metadata)

    

    if args.spliceai_cache_dataset_type == 'point':
        def collate(examples: List[torch.Tensor]):
            """
            examples: new_alabels, new_dlabels, transcripts_splice_junctions, act_rep_tensor, sub_nts,
            sub_acc_splice_junctions, sub_don_splice_junctions, fake_inclusion_splice_junctions, fake_exclusion_splice_junctions
            """

            new_alabels = list(map(obj_k_value(0), examples))
            new_dlabels = list(map(obj_k_value(1), examples))
            transcripts_splice_junctions = list(map(obj_k_value(2), examples))
            act_rep_tensor_list = list(map(obj_k_value(3), examples))
            batch_act_rep_tensor = torch.stack(act_rep_tensor_list)
            sub_nts = list(map(obj_k_value(4), examples))

            sample_name_list = list(map(obj_k_value(5), examples))
            gene_name_list = list(map(obj_k_value(6), examples))
                
            return new_alabels, new_dlabels, transcripts_splice_junctions, batch_act_rep_tensor, sub_nts, sample_name_list, gene_name_list

    elif args.spliceai_cache_dataset_type == 'sequence':
        def collate(examples: List[torch.Tensor]):
            seq_inputs_list = list(map(obj_k_value(0), examples))
            act_rep_inputs_list = list(map(obj_k_value(1), examples))
            labels_class_list = list(map(obj_k_value(2), examples))
            labels_reg_list = list(map(obj_k_value(3), examples))
            labels_len_list = list(map(len, labels_class_list))
            
            sample_name_list = list(map(obj_k_value(4), examples))
            gene_name_list = list(map(obj_k_value(5), examples))

            # seq_inputs, act_rep_inputs, labels_class, labels_reg
            padded_seq_inputs = pad_sequence(seq_inputs_list, batch_first=True, padding_value=0)
            padded_seq_inputs = torch.transpose(padded_seq_inputs, -1,-2)
            batch_act_rep_inputs = torch.stack(act_rep_inputs_list)
            padded_labels_class = pad_sequence(labels_class_list, batch_first=True, padding_value=0)
            padded_labels_reg = pad_sequence(labels_reg_list, batch_first=True, padding_value=0)
            
            # Build label mask to mask out loss values for padded regions
            label_masks = torch.ones_like(padded_labels_class)
            for ind, label_len in enumerate(labels_len_list):
                label_masks[ind, label_len:] = 0

            return padded_seq_inputs, batch_act_rep_inputs, padded_labels_class, padded_labels_reg, label_masks, sample_name_list, gene_name_list, labels_len_list

    infer_sampler = SequentialSampler(infer_dataset) if args.local_rank == -1 else DistributedSampler(infer_dataset)
    infer_dataloader = DataLoader(
        infer_dataset, sampler=infer_sampler, batch_size=args.infer_save_spliceai_batch_size, collate_fn=collate
    )

    t_total = len(infer_dataloader)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1 and not isinstance(spliceai_model, torch.nn.DataParallel):
        spliceai_model = torch.nn.DataParallel(spliceai_model)
    if args.n_gpu > 1 and not isinstance(site_aux_model, torch.nn.DataParallel):
        site_aux_model = torch.nn.DataParallel(site_aux_model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        spliceai_model = torch.nn.parallel.DistributedDataParallel(
            spliceai_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
        site_aux_model = torch.nn.parallel.DistributedDataParallel(
            site_aux_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Infer!
    logger.info("***** Inferring spliceAI and site_aux_model representations *****")
    logger.info("  Num examples = %d", len(infer_dataset))

    spliceai_model.eval()
    site_aux_model.eval()

    epoch_iterator = tqdm(infer_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator):
            
            if args.spliceai_cache_dataset_type == 'point':
                alabels, dlabels, transcripts_splice_junctions, act_rep_inputs, sub_nts, sample_name_list, gene_name_list = batch

                # compute splice_site_embs with spliceai_model from alabels, dlabels and sub_nts
                splice_site_embs = []

                for gene_ind, gene_alabels in enumerate(alabels):
                    gene_dlabels = dlabels[gene_ind]
                    gene_sub_nts = sub_nts[gene_ind]
                    # Save path parameters
                    sample_name = sample_name_list[gene_ind]
                    gene_name = gene_name_list[gene_ind]
    
                    gene_splice_sites = { **gene_alabels, **gene_dlabels, **gene_sub_nts }
                    
                    inputs = []
                    splice_sites_pos = []
                    for splice_site in gene_splice_sites.keys():
                        wseq = gene_splice_sites[splice_site]['wseq']
                        inputs.append(torch.tensor(wseq, dtype=torch.float))
                        splice_sites_pos.append(splice_site)

                    inputs = torch.stack(inputs, dim=0)
                    gene_act_rep_input = act_rep_inputs[gene_ind:gene_ind+1]
                    gene_act_rep_input = gene_act_rep_input.to(args.device)
                    
                    if args.max_spliceai_forward_batch > inputs.shape[0]:
                        inputs_batch = inputs.to(args.device)

                        spliceai_model.eval()
                        site_aux_model.eval()
                        _, site_hidden = spliceai_model(inputs_batch)


                        for site_ind, site_emb in enumerate(site_hidden):
                            splice_site_pos = splice_sites_pos[site_ind]
                            site_emb_np = site_emb.cpu().numpy()

                            h5_dataset_path = '/'.join([sample_name, gene_name, str(splice_site_pos)]) # <sample_name>/<gene_name>/<splice_site_pos>
                            with h5py.File(cache_file_path, 'a') as f:
                                dset = f.create_dataset(h5_dataset_path, data=site_emb_np)

                        if cache_both_spliceai_n_siteaux_reps:
                            _, site_act_rep_final_hidden = site_aux_model(site_hidden, act_rep_input=gene_act_rep_input)
                            for site_ind, site_emb in enumerate(site_act_rep_final_hidden):
                                splice_site_pos = splice_sites_pos[site_ind]
                                site_emb_np = site_emb.cpu().numpy()

                                h5_dataset_path = '/'.join([sample_name, gene_name, str(splice_site_pos)]) # <sample_name>/<gene_name>/<splice_site_pos>
                                with h5py.File(site_aux_model_rep_cache_file_path, 'a') as f:
                                    dset = f.create_dataset(h5_dataset_path, data=site_emb_np)

                    else:
                        # Chunk inputs into max_spliceai_forward_batch
                        num_chunks = math.ceil(inputs.shape[0] / args.max_spliceai_forward_batch)

                        for chunk_ind in range(num_chunks):
                            inputs_batch = inputs[chunk_ind*args.max_spliceai_forward_batch:(chunk_ind+1)*args.max_spliceai_forward_batch]
                            inputs_batch = inputs_batch.to(args.device)

                            spliceai_model.eval()
                            site_aux_model.eval()
                            _, site_hidden = spliceai_model(inputs_batch)

                            for site_ind, site_emb in enumerate(site_hidden):
                                real_site_ind = site_ind + (chunk_ind*args.max_spliceai_forward_batch)
                                splice_site_pos = splice_sites_pos[real_site_ind]
                                site_emb_np = site_emb.cpu().numpy()

                                h5_dataset_path = '/'.join([sample_name, gene_name, str(splice_site_pos)]) # <sample_name>/<gene_name>/<splice_site_pos>
                                with h5py.File(cache_file_path, 'a') as f:
                                    dset = f.create_dataset(h5_dataset_path, data=site_emb_np)


                            if cache_both_spliceai_n_siteaux_reps:
                                _, site_act_rep_final_hidden = site_aux_model(site_hidden, act_rep_input=gene_act_rep_input)
                                for site_ind, site_emb in enumerate(site_act_rep_final_hidden):
                                    real_site_ind = site_ind + (chunk_ind*args.max_spliceai_forward_batch)
                                    splice_site_pos = splice_sites_pos[real_site_ind]
                                    site_emb_np = site_emb.cpu().numpy()

                                    h5_dataset_path = '/'.join([sample_name, gene_name, str(splice_site_pos)]) # <sample_name>/<gene_name>/<splice_site_pos>
                                    with h5py.File(site_aux_model_rep_cache_file_path, 'a') as f:
                                        dset = f.create_dataset(h5_dataset_path, data=site_emb_np)

                    
            elif args.spliceai_cache_dataset_type == 'sequence':
                seq_inputs, act_rep_inputs, labels_class, labels_reg, label_masks, sample_name_list, gene_name_list, labels_len_list = batch

                for ind, label_seq_length in enumerate(labels_len_list):
                    # Save path parameters
                    sample_name = sample_name_list[ind]
                    gene_name = gene_name_list[ind]
                    h5_dataset_path = '/'.join([sample_name, gene_name])

                    if label_seq_length > args.max_main_seq_len_spliceai:
                        
                        num_infer_step = math.ceil(label_seq_length / args.max_main_seq_len_spliceai)
                        last_label_seq_len = label_seq_length % args.max_main_seq_len_spliceai
                        max_input_len = args.max_main_seq_len_spliceai
                        
                        # Create an empty h5 dataset first
                        with h5py.File(cache_file_path, 'a') as f:
                            dset = f.create_dataset(h5_dataset_path, (args.site_act_rep_channels, label_seq_length))

                        for i in range(num_infer_step):
                            # Retrieve individual sample and its sequence
                            start_ind = i*max_input_len
                            if i == num_infer_step-1:
                                end_ind = i*max_input_len + last_label_seq_len + args.context_len
                            else:
                                end_ind = (i+1)*max_input_len + args.context_len

                            sample_seq_inputs = seq_inputs[ind:ind+1, :, start_ind:end_ind].to(args.device)
                            sample_act_rep_inputs = act_rep_inputs[ind:ind+1].to(args.device)

                            _, site_hidden = spliceai_model(sample_seq_inputs)
                            outputs = site_aux_model(site_hidden, act_rep_input=sample_act_rep_inputs)

                            sample_site_act_rep_final_hidden = outputs[-1] # (1, 32, label_seq_length)
                            sample_site_act_rep_final_hidden = torch.squeeze(sample_site_act_rep_final_hidden, dim=0) # (32, label_seq_length)

                            # Save sample_site_act_rep_final_hidden in hdf5 file
                            sample_site_act_rep_final_hidden_np = sample_site_act_rep_final_hidden.cpu().numpy()
                            
                            # Fill in the sliced h5 dataset
                            h5_start_ind = start_ind 
                            h5_end_ind = end_ind - args.context_len
                            with h5py.File(cache_file_path, 'a') as f:
                                dset = f[h5_dataset_path]
                                dset[:, h5_start_ind:h5_end_ind] = sample_site_act_rep_final_hidden_np
                    else:
                        # Retrieve individual sample and its sequence
                        sample_seq_inputs = seq_inputs[ind:ind+1, :, :(label_seq_length+args.context_len)].to(args.device)
                        sample_act_rep_inputs = act_rep_inputs[ind:ind+1].to(args.device)

                        # outputs: (logits_class, psi_reg, site_act_rep_final_hidden)
                        _, site_hidden = spliceai_model(sample_seq_inputs)
                        outputs = site_aux_model(site_hidden, act_rep_input=sample_act_rep_inputs)
                        sample_site_act_rep_final_hidden = outputs[-1] # (1, 32, label_seq_length)
                        sample_site_act_rep_final_hidden = torch.squeeze(sample_site_act_rep_final_hidden, dim=0) # (32, label_seq_length)

                        # Save sample_site_act_rep_final_hidden in hdf5 file
                        sample_site_act_rep_final_hidden_np = sample_site_act_rep_final_hidden.cpu().numpy()

                        with h5py.File(cache_file_path, 'a') as f:
                            dset = f.create_dataset(h5_dataset_path, data=sample_site_act_rep_final_hidden_np)
            else:
                raise ValueError(
                    "--spliceai_cache_dataset_type must be either point or sequence"
                )
    return cache_file_path


def train_spliceosome_model(args, train_dataset, spliceosome_model, spliceai_model, site_aux_model, train_epochs=None, train_steps=None, valid_dataset=None, do_train_spliceai=False, do_train_site_aux_model=False, do_train_spliceai_cls=False, augment_transcript_data=None, eval_at_init=False) -> Tuple[int, float]:
    """ Train the spliceosome_model """
    if augment_transcript_data is None:
        augment_transcript_data = args.augment_transcript_data
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join(args.output_dir, 'train_spliceosome_log'))

    if args.per_gpu_train_spliceosome_model_batch_size <= 0:
        args.per_gpu_train_spliceosome_model_batch_size = args.per_gpu_train_batch_size
    args.train_spliceosome_model_batch_size = args.per_gpu_train_spliceosome_model_batch_size * max(1, args.n_gpu)

    # cached representation dir
    if args.splice_site_cache_dir is not None and args.use_cached_rep_to_train:
        cache_file_path = os.path.join(args.splice_site_cache_dir, 'sequence_representations.hdf5')

    def collate(examples: List[torch.Tensor]):
        """
        examples: new_alabels, new_dlabels, transcripts_splice_junctions, act_rep_tensor, sub_nts,
        sub_acc_splice_junctions, sub_don_splice_junctions, fake_inclusion_splice_junctions, fake_exclusion_splice_junctions
        """

        if len(examples[0]) == 11 or len(examples[0]) == 9 or len(examples[0]) == 7:
            new_alabels = list(map(obj_k_value(0), examples))
            new_dlabels = list(map(obj_k_value(1), examples))
            transcripts_splice_junctions = list(map(obj_k_value(2), examples))
            act_rep_tensor_list = list(map(obj_k_value(3), examples))
            batch_act_rep_tensor = torch.stack(act_rep_tensor_list)
            sub_nts = list(map(obj_k_value(4), examples))
            if len(examples[0]) >= 9:
                sub_acc_splice_junctions = list(map(obj_k_value(5), examples))
                sub_don_splice_junctions = list(map(obj_k_value(6), examples))
                fake_inclusion_splice_junctions = list(map(obj_k_value(7), examples))
                fake_exclusion_splice_junctions = list(map(obj_k_value(8), examples))

                if len(examples[0]) == 11:
                    sample_name_list = list(map(obj_k_value(9), examples))
                    gene_name_list = list(map(obj_k_value(10), examples))
                    return new_alabels, new_dlabels, transcripts_splice_junctions, batch_act_rep_tensor, sub_nts, sub_acc_splice_junctions, sub_don_splice_junctions, fake_inclusion_splice_junctions, fake_exclusion_splice_junctions, sample_name_list, gene_name_list
                
                return new_alabels, new_dlabels, transcripts_splice_junctions, batch_act_rep_tensor, sub_nts, sub_acc_splice_junctions, sub_don_splice_junctions, fake_inclusion_splice_junctions, fake_exclusion_splice_junctions

            sample_name_list = list(map(obj_k_value(5), examples))
            gene_name_list = list(map(obj_k_value(6), examples))
            return new_alabels, new_dlabels, transcripts_splice_junctions, batch_act_rep_tensor, sub_nts, sample_name_list, gene_name_list
        else:
            raise ValueError(
                "Invalid number of entries for training examples."
            )

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_spliceosome_model_batch_size, collate_fn=collate
    )

    if train_epochs is None:
        train_epochs = args.num_train_only_spliceosome_model_epochs
    if train_steps is None:
        train_steps = args.spliceosome_max_steps

    if train_steps > 0:
        t_total = train_steps
        train_epochs = train_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in spliceosome_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in spliceosome_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    if do_train_site_aux_model:
        optimizer_grouped_parameters = optimizer_grouped_parameters + [
            {
                "params": [p for n, p in site_aux_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in site_aux_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
    if do_train_spliceai:
        optimizer_grouped_parameters = optimizer_grouped_parameters + [
            {
                "params": [p for n, p in spliceai_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in spliceai_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if args.train_spliceosomenet_opt_scheduler_type == 'epoch':
        scheduler = get_exponential_decay_schedule(
            optimizer, num_constant_steps=args.num_constant_lr_epochs_spliceosomenet 
        )
    elif args.train_spliceosomenet_opt_scheduler_type == 'step':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps_spliceosomenet, num_training_steps=t_total
        ) 
    else:
        raise ValueError(
            "Invalid train_spliceai_opt_scheduler_type, must be either 'epoch' or 'step'."
        )

    # Check if saved optimizer or scheduler states exist
    if (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        logger.info("***** Loading optimizer & scheduler *****")

        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        if do_train_spliceai:
            [spliceai_model, site_aux_model, spliceosome_model], optimizer = amp.initialize([spliceai_model, site_aux_model, spliceosome_model], optimizer, opt_level=args.fp16_opt_level)
        elif do_train_site_aux_model:
            [site_aux_model, spliceosome_model], optimizer = amp.initialize([site_aux_model, spliceosome_model], optimizer, opt_level=args.fp16_opt_level)
        else:
            spliceosome_model, optimizer = amp.initialize(spliceosome_model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1 and not isinstance(spliceosome_model, torch.nn.DataParallel):
        spliceosome_model = torch.nn.DataParallel(spliceosome_model)
    if args.n_gpu > 1 and not isinstance(site_aux_model, torch.nn.DataParallel):
        site_aux_model = torch.nn.DataParallel(site_aux_model)
    if args.n_gpu > 1 and not isinstance(spliceai_model, torch.nn.DataParallel):
        spliceai_model = torch.nn.DataParallel(spliceai_model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        spliceosome_model = torch.nn.parallel.DistributedDataParallel(
            spliceosome_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
        spliceai_model = torch.nn.parallel.DistributedDataParallel(
            spliceai_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
        site_aux_model = torch.nn.parallel.DistributedDataParallel(
            site_aux_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
    # Train!
    logger.info("***** Running spliceosome training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_spliceosome_model_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from spliceosome_model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0

    spliceosome_model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    first_save = True

    if args.local_rank in [-1, 0] and args.logging_valid_steps > 0 and eval_at_init:
        # Log metrics
        if (
            args.local_rank == -1 and valid_dataset is not None
        ):  # Only evaluate when single GPU otherwise metrics may not average well
            logger.info("evaluate_spliceosome_model before training")
            results = evaluate_spliceosome_model(args, spliceai_model, site_aux_model, spliceosome_model, valid_dataset, max_eval_step=args.max_valid_step, 
                        eval_output_filename='spliceosome_valid_results.txt', eval_cls=do_train_spliceai_cls)
            for key, value in results.items():
                tb_writer.add_scalar("valid_spliceosome_{}".format(key), value, 0)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            
            if augment_transcript_data:
                alabels, dlabels, transcripts_splice_junctions, act_rep_inputs, sub_nts, sub_acc_splice_junctions, sub_don_splice_junctions, fake_inclusion_splice_junctions, fake_exclusion_splice_junctions, sample_name_list, gene_name_list = batch
            else:
                alabels, dlabels, transcripts_splice_junctions, act_rep_inputs, sub_nts, sample_name_list, gene_name_list = batch
                
            # compute splice_site_embs with spliceai_model from alabels, dlabels and sub_nts
            splice_site_embs = []

            if do_train_spliceai:
                total_loss_class = 0
            for gene_ind, gene_alabels in enumerate(alabels):
                gene_dlabels = dlabels[gene_ind]
                gene_sub_nts = sub_nts[gene_ind]

                # Load cache parameters
                sample_name = sample_name_list[gene_ind]
                gene_name = gene_name_list[gene_ind]
                gene_splice_sites = { **gene_alabels, **gene_dlabels, **gene_sub_nts }
                
                inputs = []
                splice_sites_pos = []
                cached_splice_sites_pos = []
                cached_site_embs = []

                if do_train_spliceai_cls:
                    labels_class = []
                    alabel_tensor = torch.from_numpy(np.array([1]))
                    dlabel_tensor = torch.from_numpy(np.array([2]))
                    sublabel_tensor = torch.from_numpy(np.array([0]))
                    for splice_site in gene_alabels.keys():
                        wseq = gene_alabels[splice_site]['wseq']
                        inputs.append(torch.tensor(wseq, dtype=torch.float))
                        splice_sites_pos.append(splice_site)
                        labels_class.append(alabel_tensor)

                    for splice_site in gene_dlabels.keys():
                        wseq = gene_dlabels[splice_site]['wseq']
                        inputs.append(torch.tensor(wseq, dtype=torch.float))
                        splice_sites_pos.append(splice_site)
                        labels_class.append(dlabel_tensor)

                    for splice_site in gene_sub_nts.keys():
                        wseq = gene_sub_nts[splice_site]['wseq']
                        inputs.append(torch.tensor(wseq, dtype=torch.float))
                        splice_sites_pos.append(splice_site)
                        labels_class.append(sublabel_tensor)
                else:
                    for splice_site in gene_splice_sites.keys():
                        if args.use_cached_rep_to_train and do_train_spliceai==False:
                            with h5py.File(cache_file_path, 'r') as f:
                                h5_group_path = '/'.join([sample_name, gene_name]) # <sample_name>/<gene_name>/<splice_site_pos>
                                if str(splice_site) in f[h5_group_path]:
                                    h5_dataset_path = '/'.join([sample_name, gene_name, str(splice_site)]) # <sample_name>/<gene_name>/<splice_site_pos>
                                    site_emb = f[h5_dataset_path][()] # (dim,)
                                    cached_splice_sites_pos.append(splice_site)
                                    cached_site_embs.append(site_emb)
                                else:
                                    wseq = gene_splice_sites[splice_site]['wseq']
                                    inputs.append(torch.tensor(wseq, dtype=torch.float))
                                    splice_sites_pos.append(splice_site)
                        else:
                            wseq = gene_splice_sites[splice_site]['wseq']
                            inputs.append(torch.tensor(wseq, dtype=torch.float))
                            splice_sites_pos.append(splice_site)

                gene_splice_site_embs = {}
                if len(inputs) != 0:
                    inputs = torch.stack(inputs, dim=0)
                    inputs = inputs.to(args.device)
                    gene_act_rep_input = act_rep_inputs[gene_ind:gene_ind+1]
                    gene_act_rep_input = gene_act_rep_input.to(args.device)
                    
                    if do_train_spliceai:
                        spliceai_model.train()
                        if do_train_spliceai_cls:
                            labels_class = torch.stack(labels_class, dim=0)
                            labels_class = labels_class.to(args.device)
                            loss_class, _, site_hidden = spliceai_model(inputs, labels_class=labels_class)
                            total_loss_class = total_loss_class + loss_class
                        else:
                            _, site_hidden = spliceai_model(inputs)
                    else:
                        spliceai_model.eval()
                        with torch.no_grad():
                            _, site_hidden = spliceai_model(inputs)

                    if do_train_site_aux_model:
                        site_aux_model.train()
                        # site_aux_model_outputs without label input: psi_reg, site_act_rep_final_hidden
                        _, site_act_rep_final_hidden = site_aux_model(site_hidden, act_rep_input=gene_act_rep_input)
                    else:
                        site_aux_model.eval()
                        with torch.no_grad():
                            _, site_act_rep_final_hidden = site_aux_model(site_hidden, act_rep_input=gene_act_rep_input)

                    for site_ind, site_emb in enumerate(site_act_rep_final_hidden):
                        splice_site_pos = splice_sites_pos[site_ind]
                        gene_splice_site_embs[splice_site_pos] = site_emb
                
                gene_cached_splice_site_embs = {}
                if len(cached_site_embs) != 0:
                    for site_ind, site_emb in enumerate(cached_site_embs):
                        splice_site_pos = cached_splice_sites_pos[site_ind]
                        site_emb_tensor = torch.tensor(site_emb).to(args.device)
                        gene_cached_splice_site_embs[splice_site_pos] = site_emb_tensor

                gene_all_splice_site_embs = {**gene_cached_splice_site_embs, **gene_splice_site_embs}

                splice_site_embs.append(gene_all_splice_site_embs)

            # Output splice_site_embs: [ {pos1: Tensor, pos2: Tensor }, {..}, .. ], array of dict where each dict stores splice sites' embedding

            spliceosome_model.train()
            if augment_transcript_data:
                fake_transcripts_splice_junctions = [sub_acc + sub_don + fake_inc + fake_exc for sub_acc, sub_don, fake_inc, fake_exc in zip(sub_acc_splice_junctions, sub_don_splice_junctions, fake_inclusion_splice_junctions, fake_exclusion_splice_junctions)]
            else:
                fake_transcripts_splice_junctions = None

            outputs = spliceosome_model(splice_site_embs, transcripts_splice_junctions, fake_transcripts_splice_junctions=fake_transcripts_splice_junctions, alabels=alabels, dlabels=dlabels)

            loss_reg, splice_sites_prob = outputs
            if do_train_spliceai_cls:
                loss = args.lambda_loss_class_spliceosome_train * total_loss_class + args.lambda_loss_reg_spliceosome_train * loss_reg
            else:
                loss = loss_reg

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(spliceosome_model.parameters(), args.max_grad_norm)
                    if do_train_site_aux_model:
                        torch.nn.utils.clip_grad_norm_(site_aux_model.parameters(), args.max_grad_norm)
                    if do_train_spliceai:
                        torch.nn.utils.clip_grad_norm_(spliceai_model.parameters(), args.max_grad_norm)

                optimizer.step()

                if args.train_spliceosomenet_opt_scheduler_type == 'step':
                    scheduler.step()  # Update learning rate schedule
                spliceosome_model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar("spliceosome_model/lr", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar("spliceosome_model/loss_reg", torch.mean(loss_reg), global_step)
                    tb_writer.add_scalar("spliceosome_model/loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    if do_train_spliceai_cls:
                        tb_writer.add_scalar("spliceosome_model/loss_class", torch.mean(loss_class), global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.logging_valid_steps > 0 and global_step % args.logging_valid_steps == 0:
                    # Log metrics
                    if (
                        args.local_rank == -1 and valid_dataset is not None
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        logger.info("evaluate_spliceosome_model")
                        results = evaluate_spliceosome_model(args, spliceai_model, site_aux_model, spliceosome_model, valid_dataset, max_eval_step=args.max_valid_step, 
                                    eval_output_filename='spliceosome_valid_results.txt', eval_cls=do_train_spliceai_cls)
                        for key, value in results.items():
                            tb_writer.add_scalar("valid_spliceosome_{}".format(key), value, global_step)

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "spliceosome_train_checkpoint"
                    if first_save:
                        _clear_checkpoints(args, checkpoint_prefix)
                        first_save = False

                    # Save spliceosome_model checkpoint
                    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        spliceosome_model.module if hasattr(spliceosome_model, "module") else spliceosome_model
                    )  # Take care of distributed/parallel training

                    # Save spliceosome model weights
                    output_model_file = os.path.join(output_dir, "spliceosomenet_pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Spliceosome model weights saved in {}".format(output_model_file))

                    # Save training args
                    torch.save(args, os.path.join(output_dir, "spliceosomenet_training_args.bin"))
                    logger.info("Saving spliceosome_model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    # Save optimizer and scheduler
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "spliceosomenet_optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "spliceosomenet_scheduler.pt"))
                    logger.info("Saving spliceosomenet optimizer and scheduler states to %s", output_dir)

                    # Save spliceai model weights if spliceai training was done
                    if do_train_spliceai:
                        model_to_save = (
                            spliceai_model.module if hasattr(spliceai_model, "module") else spliceai_model
                        )  # Take care of distributed/parallel training
                        
                        # Save model weights
                        output_model_file = os.path.join(output_dir, "spliceai_pytorch_model.bin")

                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Spliceai model weights saved in {}".format(output_model_file))

                    # Save spliceai model weights if spliceai training was done
                    if do_train_site_aux_model:
                        model_to_save = (
                            site_aux_model.module if hasattr(site_aux_model, "module") else site_aux_model
                        )  # Take care of distributed/parallel training
                        
                        # Save model weights
                        output_model_file = os.path.join(output_dir, "site_aux_pytorch_model.bin")

                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Site aux model weights saved in {}".format(output_model_file))

            if train_steps > 0 and global_step > train_steps:
                epoch_iterator.close()
                break
        
        if args.train_spliceosomenet_opt_scheduler_type == 'epoch':
            scheduler.step()
            
        if train_steps > 0 and global_step > train_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

        # Save spliceai_model
        os.makedirs(args.output_dir, exist_ok=True)
        model_to_save = (
            spliceosome_model.module if hasattr(spliceosome_model, "module") else spliceosome_model
        )  # Take care of distributed/parallel training

        # Save spliceosome model weights
        output_model_file = os.path.join(args.output_dir, "spliceosomenet_pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Spliceosome model weights saved in {}".format(output_model_file))

        if do_train_spliceai:
            # Save spliceai_model
            model_to_save = (
                spliceai_model.module if hasattr(spliceai_model, "module") else spliceai_model
            )  # Take care of distributed/parallel training
            output_model_file = os.path.join(args.output_dir, "spliceai_pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)
            logger.info("Spliceai model weights saved in {}".format(output_model_file))

        if do_train_site_aux_model:
            # Save site_aux_model
            model_to_save = (
                site_aux_model.module if hasattr(site_aux_model, "module") else site_aux_model
            )  # Take care of distributed/parallel training                    
            output_model_file = os.path.join(args.output_dir, "site_aux_pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)
            logger.info("Site aux model weights saved in {}".format(output_model_file))

        # Save training args
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        logger.info("Saving spliceai_model checkpoint to %s", args.output_dir)

        # Save optimizer and scheduler
        torch.save(optimizer.state_dict(), os.path.join(args.output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(args.output_dir, "scheduler.pt"))
        logger.info("Saving spliceai optimizer and scheduler states to %s", args.output_dir)

    return global_step, tr_loss / global_step

def train_spliceosome_model_wtranscriptprobs(args, train_dataset, spliceosome_model, spliceai_model, site_aux_model, train_epochs=None, train_steps=None, valid_dataset=None, do_transcript_reg=False, do_train_spliceai=False, do_train_site_aux_model=False, do_train_spliceai_cls=False, augment_transcript_data=None, eval_at_init=False) -> Tuple[int, float]:
    """ Train the spliceosome_model """
    if augment_transcript_data is None:
        augment_transcript_data = args.augment_transcript_data
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join(args.output_dir, 'train_spliceosome_log'))

    if args.per_gpu_train_spliceosome_model_batch_size <= 0:
        args.per_gpu_train_spliceosome_model_batch_size = args.per_gpu_train_batch_size
    args.train_spliceosome_model_batch_size = args.per_gpu_train_spliceosome_model_batch_size * max(1, args.n_gpu)

    # cached representation dir
    if args.splice_site_cache_dir is not None and args.use_cached_rep_to_train:
        cache_file_path = os.path.join(args.splice_site_cache_dir, 'sequence_representations.hdf5')

    def collate(examples: List[torch.Tensor]):
        """
        examples: new_alabels, new_dlabels, transcripts_splice_junctions, act_rep_tensor, sub_nts,
        sub_acc_splice_junctions, sub_don_splice_junctions, fake_inclusion_splice_junctions, fake_exclusion_splice_junctions
        """
        if len(examples[0]) == 11 or len(examples[0]) == 9 or len(examples[0]) == 7:
            new_alabels = list(map(obj_k_value(0), examples))
            new_dlabels = list(map(obj_k_value(1), examples))
            transcripts_splice_junctions = list(map(obj_k_value(2), examples))
            act_rep_tensor_list = list(map(obj_k_value(3), examples))
            batch_act_rep_tensor = torch.stack(act_rep_tensor_list)
            sub_nts = list(map(obj_k_value(4), examples))
            if len(examples[0]) >= 9:
                sub_acc_splice_junctions = list(map(obj_k_value(5), examples))
                sub_don_splice_junctions = list(map(obj_k_value(6), examples))
                fake_inclusion_splice_junctions = list(map(obj_k_value(7), examples))
                fake_exclusion_splice_junctions = list(map(obj_k_value(8), examples))

                if len(examples[0]) == 11:
                    sample_name_list = list(map(obj_k_value(9), examples))
                    gene_name_list = list(map(obj_k_value(10), examples))
                    return new_alabels, new_dlabels, transcripts_splice_junctions, batch_act_rep_tensor, sub_nts, sub_acc_splice_junctions, sub_don_splice_junctions, fake_inclusion_splice_junctions, fake_exclusion_splice_junctions, sample_name_list, gene_name_list
                
                return new_alabels, new_dlabels, transcripts_splice_junctions, batch_act_rep_tensor, sub_nts, sub_acc_splice_junctions, sub_don_splice_junctions, fake_inclusion_splice_junctions, fake_exclusion_splice_junctions

            sample_name_list = list(map(obj_k_value(5), examples))
            gene_name_list = list(map(obj_k_value(6), examples))
            return new_alabels, new_dlabels, transcripts_splice_junctions, batch_act_rep_tensor, sub_nts, sample_name_list, gene_name_list
        elif len(examples[0]) == 12 or len(examples[0]) == 10 or len(examples[0]) == 8:
            new_alabels = list(map(obj_k_value(0), examples))
            new_dlabels = list(map(obj_k_value(1), examples))
            tlabels = list(map(obj_k_value(2), examples))
            transcripts_splice_junctions = list(map(obj_k_value(3), examples))
            act_rep_tensor_list = list(map(obj_k_value(4), examples))
            batch_act_rep_tensor = torch.stack(act_rep_tensor_list)
            sub_nts = list(map(obj_k_value(5), examples))
            if len(examples[0]) >= 10:
                sub_acc_splice_junctions = list(map(obj_k_value(6), examples))
                sub_don_splice_junctions = list(map(obj_k_value(7), examples))
                fake_inclusion_splice_junctions = list(map(obj_k_value(8), examples))
                fake_exclusion_splice_junctions = list(map(obj_k_value(9), examples))

                if len(examples[0]) == 12:
                    sample_name_list = list(map(obj_k_value(10), examples))
                    gene_name_list = list(map(obj_k_value(11), examples))
                    return new_alabels, new_dlabels, tlabels, transcripts_splice_junctions, batch_act_rep_tensor, sub_nts, sub_acc_splice_junctions, sub_don_splice_junctions, fake_inclusion_splice_junctions, fake_exclusion_splice_junctions, sample_name_list, gene_name_list
                
                return new_alabels, new_dlabels, tlabels, transcripts_splice_junctions, batch_act_rep_tensor, sub_nts, sub_acc_splice_junctions, sub_don_splice_junctions, fake_inclusion_splice_junctions, fake_exclusion_splice_junctions

            sample_name_list = list(map(obj_k_value(6), examples))
            gene_name_list = list(map(obj_k_value(7), examples))
            return new_alabels, new_dlabels, tlabels, transcripts_splice_junctions, batch_act_rep_tensor, sub_nts, sample_name_list, gene_name_list
        else:
            raise ValueError(
                "Invalid number of entries for training examples."
            )

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_spliceosome_model_batch_size, collate_fn=collate
    )

    if train_epochs is None:
        train_epochs = args.num_train_only_spliceosome_model_epochs
    if train_steps is None:
        train_steps = args.spliceosome_max_steps

    if train_steps > 0:
        t_total = train_steps
        train_epochs = train_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in spliceosome_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in spliceosome_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    if do_train_site_aux_model:
        optimizer_grouped_parameters = optimizer_grouped_parameters + [
            {
                "params": [p for n, p in site_aux_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in site_aux_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
    if do_train_spliceai:
        optimizer_grouped_parameters = optimizer_grouped_parameters + [
            {
                "params": [p for n, p in spliceai_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in spliceai_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if args.train_spliceosomenet_opt_scheduler_type == 'epoch':
        scheduler = get_exponential_decay_schedule(
            optimizer, num_constant_steps=args.num_constant_lr_epochs_spliceosomenet 
        )
    elif args.train_spliceosomenet_opt_scheduler_type == 'step':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps_spliceosomenet, num_training_steps=t_total
        ) 
    else:
        raise ValueError(
            "Invalid train_spliceai_opt_scheduler_type, must be either 'epoch' or 'step'."
        )

    # Check if saved optimizer or scheduler states exist
    if (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        logger.info("***** Loading optimizer & scheduler *****")

        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        if do_train_spliceai:
            [spliceai_model, site_aux_model, spliceosome_model], optimizer = amp.initialize([spliceai_model, site_aux_model, spliceosome_model], optimizer, opt_level=args.fp16_opt_level)
        elif do_train_site_aux_model:
            [site_aux_model, spliceosome_model], optimizer = amp.initialize([site_aux_model, spliceosome_model], optimizer, opt_level=args.fp16_opt_level)
        else:
            spliceosome_model, optimizer = amp.initialize(spliceosome_model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1 and not isinstance(spliceosome_model, torch.nn.DataParallel):
        spliceosome_model = torch.nn.DataParallel(spliceosome_model)
    if args.n_gpu > 1 and not isinstance(site_aux_model, torch.nn.DataParallel):
        site_aux_model = torch.nn.DataParallel(site_aux_model)
    if args.n_gpu > 1 and not isinstance(spliceai_model, torch.nn.DataParallel):
        spliceai_model = torch.nn.DataParallel(spliceai_model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        spliceosome_model = torch.nn.parallel.DistributedDataParallel(
            spliceosome_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
        spliceai_model = torch.nn.parallel.DistributedDataParallel(
            spliceai_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
        site_aux_model = torch.nn.parallel.DistributedDataParallel(
            site_aux_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
    # Train!
    logger.info("***** Running spliceosome training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_spliceosome_model_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from spliceosome_model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0

    spliceosome_model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    first_save = True

    if args.local_rank in [-1, 0] and args.logging_valid_steps > 0 and eval_at_init:
        # Log metrics
        if (
            args.local_rank == -1 and valid_dataset is not None
        ):  # Only evaluate when single GPU otherwise metrics may not average well
            logger.info("evaluate_spliceosome_model before training")
            results = evaluate_spliceosome_model(args, spliceai_model, site_aux_model, spliceosome_model, valid_dataset, max_eval_step=args.max_valid_step, 
                        eval_output_filename='spliceosome_valid_results.txt', eval_cls=do_train_spliceai_cls)
            for key, value in results.items():
                tb_writer.add_scalar("valid_spliceosome_{}".format(key), value, 0)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            if do_transcript_reg:
                if augment_transcript_data:
                    alabels, dlabels, tlabels, transcripts_splice_junctions, act_rep_inputs, sub_nts, sub_acc_splice_junctions, sub_don_splice_junctions, fake_inclusion_splice_junctions, fake_exclusion_splice_junctions, sample_name_list, gene_name_list = batch
                else:
                    alabels, dlabels, tlabels, transcripts_splice_junctions, act_rep_inputs, sub_nts, sample_name_list, gene_name_list = batch
            else:
                if augment_transcript_data:
                    alabels, dlabels, transcripts_splice_junctions, act_rep_inputs, sub_nts, sub_acc_splice_junctions, sub_don_splice_junctions, fake_inclusion_splice_junctions, fake_exclusion_splice_junctions, sample_name_list, gene_name_list = batch
                else:
                    alabels, dlabels, transcripts_splice_junctions, act_rep_inputs, sub_nts, sample_name_list, gene_name_list = batch
                
            # compute splice_site_embs with spliceai_model from alabels, dlabels and sub_nts
            splice_site_embs = []

            if do_train_spliceai:
                total_loss_class = 0
            for gene_ind, gene_alabels in enumerate(alabels):
                gene_dlabels = dlabels[gene_ind]
                gene_sub_nts = sub_nts[gene_ind]

                # Load cache parameters
                sample_name = sample_name_list[gene_ind]
                gene_name = gene_name_list[gene_ind]
                gene_splice_sites = { **gene_alabels, **gene_dlabels, **gene_sub_nts }
                
                inputs = []
                splice_sites_pos = []
                cached_splice_sites_pos = []
                cached_site_embs = []

                if do_train_spliceai_cls:
                    labels_class = []
                    alabel_tensor = torch.from_numpy(np.array([1]))
                    dlabel_tensor = torch.from_numpy(np.array([2]))
                    sublabel_tensor = torch.from_numpy(np.array([0]))
                    for splice_site in gene_alabels.keys():
                        wseq = gene_alabels[splice_site]['wseq']
                        inputs.append(torch.tensor(wseq, dtype=torch.float))
                        splice_sites_pos.append(splice_site)
                        labels_class.append(alabel_tensor)

                    for splice_site in gene_dlabels.keys():
                        wseq = gene_dlabels[splice_site]['wseq']
                        inputs.append(torch.tensor(wseq, dtype=torch.float))
                        splice_sites_pos.append(splice_site)
                        labels_class.append(dlabel_tensor)

                    for splice_site in gene_sub_nts.keys():
                        wseq = gene_sub_nts[splice_site]['wseq']
                        inputs.append(torch.tensor(wseq, dtype=torch.float))
                        splice_sites_pos.append(splice_site)
                        labels_class.append(sublabel_tensor)
                else:
                    for splice_site in gene_splice_sites.keys():
                        if args.use_cached_rep_to_train and do_train_spliceai==False:
                            with h5py.File(cache_file_path, 'r') as f:
                                h5_group_path = '/'.join([sample_name, gene_name]) # <sample_name>/<gene_name>/<splice_site_pos>
                                if str(splice_site) in f[h5_group_path]:
                                    h5_dataset_path = '/'.join([sample_name, gene_name, str(splice_site)]) # <sample_name>/<gene_name>/<splice_site_pos>
                                    site_emb = f[h5_dataset_path][()] # (dim,)
                                    cached_splice_sites_pos.append(splice_site)
                                    cached_site_embs.append(site_emb)
                                else:
                                    wseq = gene_splice_sites[splice_site]['wseq']
                                    inputs.append(torch.tensor(wseq, dtype=torch.float))
                                    splice_sites_pos.append(splice_site)

                        else:
                            wseq = gene_splice_sites[splice_site]['wseq']
                            inputs.append(torch.tensor(wseq, dtype=torch.float))
                            splice_sites_pos.append(splice_site)

                gene_splice_site_embs = {}
                if len(inputs) != 0:
                    inputs = torch.stack(inputs, dim=0)
                    inputs = inputs.to(args.device)
                    gene_act_rep_input = act_rep_inputs[gene_ind:gene_ind+1]
                    gene_act_rep_input = gene_act_rep_input.to(args.device)
                    
                    if do_train_spliceai:
                        spliceai_model.train()
                        if do_train_spliceai_cls:
                            labels_class = torch.stack(labels_class, dim=0)
                            labels_class = labels_class.to(args.device)
                            loss_class, _, site_hidden = spliceai_model(inputs, labels_class=labels_class)
                            total_loss_class = total_loss_class + loss_class
                        else:
                            _, site_hidden = spliceai_model(inputs)
                    else:
                        spliceai_model.eval()
                        with torch.no_grad():
                            _, site_hidden = spliceai_model(inputs)

                    if do_train_site_aux_model:
                        site_aux_model.train()
                        # site_aux_model_outputs without label input: psi_reg, site_act_rep_final_hidden
                        _, site_act_rep_final_hidden = site_aux_model(site_hidden, act_rep_input=gene_act_rep_input)
                    else:
                        site_aux_model.eval()
                        with torch.no_grad():
                            _, site_act_rep_final_hidden = site_aux_model(site_hidden, act_rep_input=gene_act_rep_input)

                    for site_ind, site_emb in enumerate(site_act_rep_final_hidden):
                        splice_site_pos = splice_sites_pos[site_ind]
                        gene_splice_site_embs[splice_site_pos] = site_emb
                
                gene_cached_splice_site_embs = {}
                if len(cached_site_embs) != 0:
                    for site_ind, site_emb in enumerate(cached_site_embs):
                        splice_site_pos = cached_splice_sites_pos[site_ind]
                        site_emb_tensor = torch.tensor(site_emb).to(args.device)
                        gene_cached_splice_site_embs[splice_site_pos] = site_emb_tensor

                gene_all_splice_site_embs = {**gene_cached_splice_site_embs, **gene_splice_site_embs}

                splice_site_embs.append(gene_all_splice_site_embs)

            # Output splice_site_embs: [ {pos1: Tensor, pos2: Tensor }, {..}, .. ], array of dict where each dict stores splice sites' embedding

            spliceosome_model.train()
            if augment_transcript_data:
                fake_transcripts_splice_junctions = [sub_acc + sub_don + fake_inc + fake_exc for sub_acc, sub_don, fake_inc, fake_exc in zip(sub_acc_splice_junctions, sub_don_splice_junctions, fake_inclusion_splice_junctions, fake_exclusion_splice_junctions)]
            else:
                fake_transcripts_splice_junctions = None

            if do_transcript_reg:
                outputs = spliceosome_model(splice_site_embs, transcripts_splice_junctions, fake_transcripts_splice_junctions=fake_transcripts_splice_junctions, alabels=alabels, dlabels=dlabels, tlabels=tlabels)
                loss_reg, loss_reg_transcript, splice_sites_prob, _ = outputs
                if do_train_spliceai_cls:
                    if args.lambda_loss_reg_spliceosome_train != 0:
                        if args.lambda_loss_reg_transcript_spliceosome_train != 0:
                            loss = args.lambda_loss_class_spliceosome_train * total_loss_class + args.lambda_loss_reg_spliceosome_train * loss_reg + args.lambda_loss_reg_transcript_spliceosome_train * loss_reg_transcript
                        else:
                            loss = args.lambda_loss_class_spliceosome_train * total_loss_class + args.lambda_loss_reg_spliceosome_train * loss_reg
                    else:
                        loss = args.lambda_loss_class_spliceosome_train * total_loss_class + args.lambda_loss_reg_transcript_spliceosome_train * loss_reg_transcript
                else:
                    if args.lambda_loss_reg_spliceosome_train != 0:
                        if args.lambda_loss_reg_transcript_spliceosome_train != 0:
                            loss = args.lambda_loss_reg_spliceosome_train * loss_reg + args.lambda_loss_reg_transcript_spliceosome_train * loss_reg_transcript
                        else:
                            loss = args.lambda_loss_reg_spliceosome_train * loss_reg
                    else:
                        loss = loss_reg_transcript
            else:
                outputs = spliceosome_model(splice_site_embs, transcripts_splice_junctions, fake_transcripts_splice_junctions=fake_transcripts_splice_junctions, alabels=alabels, dlabels=dlabels)
                loss_reg, splice_sites_prob = outputs
                # outputs: (loss_reg, splice_sites_prob)
                if do_train_spliceai_cls:
                    loss = args.lambda_loss_class_spliceosome_train * total_loss_class + args.lambda_loss_reg_spliceosome_train * loss_reg
                else:
                    loss = loss_reg

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(spliceosome_model.parameters(), args.max_grad_norm)
                    if do_train_site_aux_model:
                        torch.nn.utils.clip_grad_norm_(site_aux_model.parameters(), args.max_grad_norm)
                    if do_train_spliceai:
                        torch.nn.utils.clip_grad_norm_(spliceai_model.parameters(), args.max_grad_norm)

                optimizer.step()

                if args.train_spliceosomenet_opt_scheduler_type == 'step':
                    scheduler.step()  # Update learning rate schedule
                spliceosome_model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar("spliceosome_model/lr", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar("spliceosome_model/loss_reg", torch.mean(loss_reg), global_step)

                    if args.do_transcript_reg:
                        tb_writer.add_scalar("spliceosome_model/loss_reg_transcript", torch.mean(loss_reg_transcript), global_step)

                    tb_writer.add_scalar("spliceosome_model/loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    if do_train_spliceai_cls:
                        tb_writer.add_scalar("spliceosome_model/loss_class", torch.mean(loss_class), global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.logging_valid_steps > 0 and global_step % args.logging_valid_steps == 0:
                    # Log metrics
                    if (
                        args.local_rank == -1 and valid_dataset is not None
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        logger.info("evaluate_spliceosome_model")
                        results = evaluate_spliceosome_model(args, spliceai_model, site_aux_model, spliceosome_model, valid_dataset, max_eval_step=args.max_valid_step, 
                                    eval_output_filename='spliceosome_valid_results.txt', eval_cls=do_train_spliceai_cls)
                        for key, value in results.items():
                            tb_writer.add_scalar("valid_spliceosome_{}".format(key), value, global_step)

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "spliceosome_train_checkpoint"
                    if first_save:
                        _clear_checkpoints(args, checkpoint_prefix)
                        first_save = False

                    # Save spliceosome_model checkpoint
                    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        spliceosome_model.module if hasattr(spliceosome_model, "module") else spliceosome_model
                    )  # Take care of distributed/parallel training

                    # Save spliceosome model weights
                    output_model_file = os.path.join(output_dir, "spliceosomenet_pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Spliceosome model weights saved in {}".format(output_model_file))

                    # Save training args
                    torch.save(args, os.path.join(output_dir, "spliceosomenet_training_args.bin"))
                    logger.info("Saving spliceosome_model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    # Save optimizer and scheduler
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "spliceosomenet_optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "spliceosomenet_scheduler.pt"))
                    logger.info("Saving spliceosomenet optimizer and scheduler states to %s", output_dir)

                    # Save spliceai model weights if spliceai training was done
                    if do_train_spliceai:
                        model_to_save = (
                            spliceai_model.module if hasattr(spliceai_model, "module") else spliceai_model
                        )  # Take care of distributed/parallel training
                        
                        # Save model weights
                        output_model_file = os.path.join(output_dir, "spliceai_pytorch_model.bin")

                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Spliceai model weights saved in {}".format(output_model_file))

                    # Save spliceai model weights if spliceai training was done
                    if do_train_site_aux_model:
                        model_to_save = (
                            site_aux_model.module if hasattr(site_aux_model, "module") else site_aux_model
                        )  # Take care of distributed/parallel training
                        
                        # Save model weights
                        output_model_file = os.path.join(output_dir, "site_aux_pytorch_model.bin")

                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Site aux model weights saved in {}".format(output_model_file))

            if train_steps > 0 and global_step > train_steps:
                epoch_iterator.close()
                break
        
        if args.train_spliceosomenet_opt_scheduler_type == 'epoch':
            scheduler.step()
            
        if train_steps > 0 and global_step > train_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

        # Save spliceai_model
        os.makedirs(args.output_dir, exist_ok=True)
        model_to_save = (
            spliceosome_model.module if hasattr(spliceosome_model, "module") else spliceosome_model
        )  # Take care of distributed/parallel training

        # Save spliceosome model weights
        output_model_file = os.path.join(args.output_dir, "spliceosomenet_pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Spliceosome model weights saved in {}".format(output_model_file))

        if do_train_spliceai:
            # Save spliceai_model
            model_to_save = (
                spliceai_model.module if hasattr(spliceai_model, "module") else spliceai_model
            )  # Take care of distributed/parallel training
            output_model_file = os.path.join(args.output_dir, "spliceai_pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)
            logger.info("Spliceai model weights saved in {}".format(output_model_file))

        if do_train_site_aux_model:
            # Save site_aux_model
            model_to_save = (
                site_aux_model.module if hasattr(site_aux_model, "module") else site_aux_model
            )  # Take care of distributed/parallel training                    
            output_model_file = os.path.join(args.output_dir, "site_aux_pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)
            logger.info("Site aux model weights saved in {}".format(output_model_file))

        # Save training args
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        logger.info("Saving spliceai_model checkpoint to %s", args.output_dir)

        # Save optimizer and scheduler
        torch.save(optimizer.state_dict(), os.path.join(args.output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(args.output_dir, "scheduler.pt"))
        logger.info("Saving spliceai optimizer and scheduler states to %s", args.output_dir)

    return global_step, tr_loss / global_step

# Adapted from spliceai source code
def print_topl_statistics(preds, labels):
    # Prints the following information: top-kL statistics for k=0.5,1,2,4,
    # auprc, thresholds for k=0.5,1,2,4, number of true splice sites.

    idx_true = np.nonzero(labels == 1)[0]
    argsorted_y_pred = np.argsort(preds)
    sorted_y_pred = np.sort(preds)

    topkl_accuracy = []
    threshold = []

    for top_length in [0.5, 1, 2, 4]:

        idx_pred = argsorted_y_pred[-int(top_length*len(idx_true)):]

        topkl_accuracy += [np.size(np.intersect1d(idx_true, idx_pred)) \
                  / float(min(len(idx_pred), len(idx_true)))]
        threshold += [sorted_y_pred[-int(top_length*len(idx_true))]]

    auprc = average_precision_score(labels, preds)

    return topkl_accuracy, auprc, threshold, len(idx_true)

# Adapted from huggingface glue code
def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def evaluate_spliceai(args, spliceai_model, site_aux_model, eval_dataset, eval_output_dir=None, prefix="", 
    eval_accuracy=True, eval_correlation=True, eval_cor_with_spliceaicls=True, max_eval_step=None, eval_output_filename=None, ablation_finetune_model=None, save_predlabels_as_npy=True) -> Dict:
    if eval_output_dir is None:
        eval_output_dir = args.output_dir

    if eval_output_filename is None:
        eval_output_filename = args.eval_output_filename

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    def collate(examples: List[torch.Tensor]):
        seq_inputs_list = list(map(obj_k_value(0), examples))
        act_rep_inputs_list = list(map(obj_k_value(1), examples))
        labels_class_list = list(map(obj_k_value(2), examples))
        labels_reg_list = list(map(obj_k_value(3), examples))
        labels_len_list = list(map(len, labels_class_list))

        # seq_inputs, act_rep_inputs, labels_class, labels_reg
        padded_seq_inputs = pad_sequence(seq_inputs_list, batch_first=True, padding_value=0)
        padded_seq_inputs = torch.transpose(padded_seq_inputs, -1,-2)
        batch_act_rep_inputs = torch.stack(act_rep_inputs_list)
        padded_labels_class = pad_sequence(labels_class_list, batch_first=True, padding_value=0)
        padded_labels_reg = pad_sequence(labels_reg_list, batch_first=True, padding_value=0)
        
        # Build label mask to mask out loss values for padded regions
        label_masks = torch.ones_like(padded_labels_class)
        for ind, label_len in enumerate(labels_len_list):
            label_masks[ind, label_len:] = 0

        return padded_seq_inputs, batch_act_rep_inputs, padded_labels_class, padded_labels_reg, label_masks

    if max_eval_step is not None and max_eval_step < len(eval_dataset):
        eval_sampler = RandomSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    else:
        eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(spliceai_model, torch.nn.DataParallel):
        spliceai_model = torch.nn.DataParallel(spliceai_model)
    if args.n_gpu > 1 and not isinstance(site_aux_model, torch.nn.DataParallel):
        site_aux_model = torch.nn.DataParallel(site_aux_model)
    if ablation_finetune_model and args.n_gpu > 1 and not isinstance(ablation_finetune_model, torch.nn.DataParallel):
        ablation_finetune_model = torch.nn.DataParallel(ablation_finetune_model)
        
    # Eval!
    logger.info("***** Running spliceAI evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    start_time = time.time()
    eval_loss_class = 0.0
    eval_loss_reg = 0.0
    nb_eval_steps = 0
    spliceai_model.eval()
    site_aux_model.eval()
    if ablation_finetune_model:
        ablation_finetune_model.eval()

    probs_acc_list = []
    probs_don_list = []
    labels_acc_list = []
    labels_don_list = []

    pred_psi_acc_list = []
    pred_psi_don_list = []
    labels_psi_acc_list = []
    labels_psi_don_list = []
    
    if eval_cor_with_spliceaicls:
        pred_psi_cls_acc_list = []
        pred_psi_cls_don_list = []

    correct_cls = 0
    correct_none_cls = 0
    correct_acc_cls = 0
    correct_don_cls = 0
    num_sites = 0
    num_none = 0
    num_acc = 0
    num_don = 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        seq_inputs, act_rep_inputs, labels_class, labels_reg, label_masks = batch

        seq_inputs = seq_inputs.to(args.device)
        act_rep_inputs = act_rep_inputs.to(args.device)
        labels_class = labels_class.to(args.device)
        labels_reg = labels_reg.to(args.device)
        label_masks = label_masks.to(args.device)
        
        with torch.no_grad():
            outputs = spliceai_model(seq_inputs, labels_class=labels_class, label_masks=label_masks)
            # outputs: (loss_class, logits_class, site_hidden)
            loss_class, site_hidden = outputs[0], outputs[2]

            # site_aux_model_outputs: loss_reg, psi_reg, site_act_rep_final_hidden
            site_aux_model_outputs = site_aux_model(site_hidden, act_rep_input=act_rep_inputs, labels_class=labels_class, labels_reg=labels_reg, label_masks=label_masks)

            if ablation_finetune_model:
                site_act_rep_final_hidden = site_aux_model_outputs[-1]
                ablation_finetune_model_outputs = ablation_finetune_model(site_act_rep_final_hidden, labels_class=labels_class, labels_reg=labels_reg, label_masks=label_masks)
                loss_reg = ablation_finetune_model_outputs[0]
            else:
                loss_reg = site_aux_model_outputs[0]

            eval_loss_class += loss_class.mean().item()
            eval_loss_reg += loss_reg.mean().item()
            
            logits_class = outputs[1]
            probs_class = nn.functional.softmax(logits_class, dim=1)
            if eval_accuracy:
                # Compute top-k accuracy
                probs_acc = probs_class[:,1,:]
                probs_don = probs_class[:,2,:]

                labels_acc = torch.zeros_like(probs_acc)
                labels_acc[labels_class == 1] = 1
                labels_don = torch.zeros_like(probs_don)
                labels_don[labels_class == 2] = 1

                # for simple accuracy computation
                _, pred_class = torch.max(probs_class.data, dim=1)

                for ind, seq_label_mask in enumerate(label_masks):
                    seq_len = torch.sum(seq_label_mask)
                    seq_probs_acc = probs_acc[ind, :seq_len]
                    seq_probs_don = probs_don[ind, :seq_len]

                    seq_labels_acc = labels_acc[ind, :seq_len]
                    seq_labels_don = labels_don[ind, :seq_len]
                    
                    probs_acc_list.append(seq_probs_acc.cpu().numpy())
                    probs_don_list.append(seq_probs_don.cpu().numpy())
                    labels_acc_list.append(seq_labels_acc.cpu().numpy())
                    labels_don_list.append(seq_labels_don.cpu().numpy())

                    # for simple accuracy computation
                    seq_pred_class = pred_class[ind, :seq_len]
                    seq_labels_class = labels_class[ind, :seq_len]
                    seq_correct = (seq_pred_class == seq_labels_class)

                    seq_none = (seq_labels_class == 0)
                    seq_acc = (seq_labels_class == 1)
                    seq_don = (seq_labels_class == 2)

                    correct_cls += seq_correct.sum().item()
                    correct_none_cls += seq_correct[seq_none].sum().item()
                    correct_acc_cls += seq_correct[seq_acc].sum().item()
                    correct_don_cls += seq_correct[seq_don].sum().item()

                    num_sites += seq_len.item()
                    num_none += seq_none.sum().item()
                    num_acc += seq_acc.sum().item()
                    num_don += seq_don.sum().item()

            if eval_correlation:
                # Compute spearman rank and pearson correlation
                if ablation_finetune_model:
                    psi_reg = ablation_finetune_model_outputs[1]
                else:
                    psi_reg = site_aux_model_outputs[1]
                psi_acc = psi_reg[labels_class==1]
                psi_don = psi_reg[labels_class==2]

                            
                pred_psi_acc_list.append(psi_acc.cpu().numpy())
                pred_psi_don_list.append(psi_don.cpu().numpy())

            labels_psi_acc = labels_reg[labels_class==1]
            labels_psi_don = labels_reg[labels_class==2]
            labels_psi_acc_list.append(labels_psi_acc.cpu().numpy())
            labels_psi_don_list.append(labels_psi_don.cpu().numpy())

            if eval_cor_with_spliceaicls:
                prob_cls_acc = probs_class[:, 1]
                prob_cls_don = probs_class[:, 2]
                psi_cls_acc = prob_cls_acc[labels_class==1]
                psi_cls_don = prob_cls_don[labels_class==2]
                pred_psi_cls_acc_list.append(psi_cls_acc.cpu().numpy())
                pred_psi_cls_don_list.append(psi_cls_don.cpu().numpy())

        nb_eval_steps += 1

        if max_eval_step is not None and nb_eval_steps > max_eval_step:
            break

    eval_loss_class = eval_loss_class / nb_eval_steps
    eval_loss_reg = eval_loss_reg / nb_eval_steps
    result = {"eval_loss_class": eval_loss_class, "eval_loss_reg": eval_loss_reg}

    if eval_accuracy:
        # Compute full dataset top-k acc
        probs_acc_list = np.concatenate(probs_acc_list)
        probs_don_list = np.concatenate(probs_don_list)
        labels_acc_list = np.concatenate(labels_acc_list)
        labels_don_list = np.concatenate(labels_don_list)
        
        # Compute simple acc
        accuracy_cls = correct_cls / num_sites
        accuracy_none_cls = correct_none_cls / num_none

        accuracy_result = {"accuracy_cls": accuracy_cls, "accuracy_none_cls": accuracy_none_cls}

        if np.sum(labels_acc_list) > 0:
            topkl_accuracy_acc, auprc_acc, threshold_acc, num_acc = print_topl_statistics(probs_acc_list, labels_acc_list)
            accuracy_acc_cls = correct_acc_cls / num_acc

            accuracy_result= {**accuracy_result,
                            "topk1x_acc": topkl_accuracy_acc[1], "auprc_acc": auprc_acc, "accuracy_acc_cls": accuracy_acc_cls}

        if np.sum(labels_don_list) > 0:
            topkl_accuracy_don, auprc_don, threshold_don, num_don = print_topl_statistics(probs_don_list, labels_don_list)
            accuracy_don_cls = correct_don_cls / num_don

            accuracy_result= {**accuracy_result,
                            "topk1x_don": topkl_accuracy_don[1], "auprc_don": auprc_don, "accuracy_don_cls": accuracy_don_cls}
        
        if np.sum(labels_acc_list) > 0 and  np.sum(labels_don_list) > 0:
            accuracy_result= {**accuracy_result,
                            "topk1x_avg": (topkl_accuracy_acc[1] + topkl_accuracy_don[1]) / 2, "auprc_avg": (auprc_acc + auprc_don) / 2}
                            
        result = {**accuracy_result, **result}

    correlation_result = {}
    labels_psi_acc_list = np.concatenate(labels_psi_acc_list)
    labels_psi_don_list = np.concatenate(labels_psi_don_list)
    labels_psi_ss_list = np.concatenate([labels_psi_acc_list, labels_psi_don_list])

    if eval_correlation:
        # Compute spearman rank and pearson correlation
        pred_psi_acc_list = np.concatenate(pred_psi_acc_list)
        pred_psi_don_list = np.concatenate(pred_psi_don_list)

        pred_psi_ss_list = np.concatenate([pred_psi_acc_list, pred_psi_don_list])

        acc_cor_result = pearson_and_spearman(pred_psi_acc_list, labels_psi_acc_list)
        don_cor_result = pearson_and_spearman(pred_psi_don_list, labels_psi_don_list)
        all_ss_cor_result = pearson_and_spearman(pred_psi_ss_list, labels_psi_ss_list)

        correlation_result = {**correlation_result,
                            "spearmanr_acc": acc_cor_result['spearmanr'], "pearson_acc": acc_cor_result['pearson'], 
                            "spearmanr_don": don_cor_result['spearmanr'], "pearson_don": don_cor_result['pearson'], 
                            "spearmanr_ss": all_ss_cor_result['spearmanr'], "pearson_ss": all_ss_cor_result['pearson']}

    if eval_cor_with_spliceaicls:
        pred_psi_cls_acc_list = np.concatenate(pred_psi_cls_acc_list)
        pred_psi_cls_don_list = np.concatenate(pred_psi_cls_don_list)

        pred_psi_cls_ss_list = np.concatenate([pred_psi_cls_acc_list, pred_psi_cls_don_list])

        cls_acc_cor_result = pearson_and_spearman(pred_psi_cls_acc_list, labels_psi_acc_list)
        cls_don_cor_result = pearson_and_spearman(pred_psi_cls_don_list, labels_psi_don_list)
        cls_all_ss_cor_result = pearson_and_spearman(pred_psi_cls_ss_list, labels_psi_ss_list)
        correlation_result = {**correlation_result,
                            "spearmanr_acc_cls": cls_acc_cor_result['spearmanr'], "pearson_acc_cls": cls_acc_cor_result['pearson'], 
                            "spearmanr_don_cls": cls_don_cor_result['spearmanr'], "pearson_don_cls": cls_don_cor_result['pearson'], 
                            "spearmanr_ss_cls": cls_all_ss_cor_result['spearmanr'], "pearson_ss_cls": cls_all_ss_cor_result['pearson']}
                                    
        result = {**correlation_result, **result}


    output_eval_file = os.path.join(eval_output_dir, prefix, eval_output_filename)
    if args.local_rank in [-1, 0]:
        os.makedirs(os.path.join(eval_output_dir, prefix), exist_ok=True)
    end_time = time.time()
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("***** Eval spliceai models results {} *****".format(prefix))
            writer.write("%s = %s\n" % (key, str(result[key])))

        hrs, mins, secs = time_to_human(end_time - start_time)
        writer.write("Overall time elapsed: {} hrs {} mins {} seconds".format(int(hrs), int(mins), round(secs)))
    
    if save_predlabels_as_npy:
        if eval_correlation:
            with open(os.path.join(eval_output_dir, prefix, 'pred_psi_acc.npy'), 'wb') as f:
                np.save(f, pred_psi_acc_list)
            with open(os.path.join(eval_output_dir, prefix, 'pred_psi_don.npy'), 'wb') as f:
                np.save(f, pred_psi_don_list)
        elif eval_cor_with_spliceaicls:
            with open(os.path.join(eval_output_dir, prefix, 'pred_psi_acc.npy'), 'wb') as f:
                np.save(f, pred_psi_cls_acc_list)
            with open(os.path.join(eval_output_dir, prefix, 'pred_psi_don.npy'), 'wb') as f:
                np.save(f, pred_psi_cls_don_list)

        with open(os.path.join(eval_output_dir, prefix, 'labels_psi_acc.npy'), 'wb') as f:
            np.save(f, labels_psi_acc_list)
        with open(os.path.join(eval_output_dir, prefix, 'labels_psi_don.npy'), 'wb') as f:
            np.save(f, labels_psi_don_list)

    return result


def evaluate_spliceosome_model(args, spliceai_model, site_aux_model, spliceosome_model, eval_dataset, eval_output_dir=None, eval_output_filename=None, prefix="", eval_correlation=True, max_eval_step=None, eval_cls=False, save_predlabels_as_npy=True) -> Dict:
    if eval_output_dir is None:
        eval_output_dir = args.output_dir

    if eval_output_filename is None:
        eval_output_filename = args.eval_output_filename

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    # cached representation dir
    if args.splice_site_cache_dir is not None and args.use_cached_rep_to_train:
        cache_file_path = os.path.join(args.splice_site_cache_dir, 'sequence_representations.hdf5')

    def collate(examples: List[torch.Tensor]):
        """
        examples: new_alabels, new_dlabels, transcripts_splice_junctions, act_rep_tensor, sub_nts,
        sub_acc_splice_junctions, sub_don_splice_junctions, fake_inclusion_splice_junctions, fake_exclusion_splice_junctions
        """
        if len(examples[0]) == 11 or len(examples[0]) == 9 or len(examples[0]) == 7:
            new_alabels = list(map(obj_k_value(0), examples))
            new_dlabels = list(map(obj_k_value(1), examples))
            transcripts_splice_junctions = list(map(obj_k_value(2), examples))
            act_rep_tensor_list = list(map(obj_k_value(3), examples))
            batch_act_rep_tensor = torch.stack(act_rep_tensor_list)
            sub_nts = list(map(obj_k_value(4), examples))
            if len(examples[0]) >= 9:
                sub_acc_splice_junctions = list(map(obj_k_value(5), examples))
                sub_don_splice_junctions = list(map(obj_k_value(6), examples))
                fake_inclusion_splice_junctions = list(map(obj_k_value(7), examples))
                fake_exclusion_splice_junctions = list(map(obj_k_value(8), examples))
                if len(examples[0]) == 11:
                    sample_name_list = list(map(obj_k_value(9), examples))
                    gene_name_list = list(map(obj_k_value(10), examples))
                    return new_alabels, new_dlabels, transcripts_splice_junctions, batch_act_rep_tensor, sub_nts, sub_acc_splice_junctions, sub_don_splice_junctions, fake_inclusion_splice_junctions, fake_exclusion_splice_junctions, sample_name_list, gene_name_list
                return new_alabels, new_dlabels, transcripts_splice_junctions, batch_act_rep_tensor, sub_nts, sub_acc_splice_junctions, sub_don_splice_junctions, fake_inclusion_splice_junctions, fake_exclusion_splice_junctions

            
            sample_name_list = list(map(obj_k_value(5), examples))
            gene_name_list = list(map(obj_k_value(6), examples))
            return new_alabels, new_dlabels, transcripts_splice_junctions, batch_act_rep_tensor, sub_nts, sample_name_list, gene_name_list
        elif len(examples[0]) == 12 or len(examples[0]) == 10 or len(examples[0]) == 8:
            new_alabels = list(map(obj_k_value(0), examples))
            new_dlabels = list(map(obj_k_value(1), examples))
            tlabels = list(map(obj_k_value(2), examples))
            transcripts_splice_junctions = list(map(obj_k_value(3), examples))
            act_rep_tensor_list = list(map(obj_k_value(4), examples))
            batch_act_rep_tensor = torch.stack(act_rep_tensor_list)
            sub_nts = list(map(obj_k_value(5), examples))
            if len(examples[0]) >= 10:
                sub_acc_splice_junctions = list(map(obj_k_value(6), examples))
                sub_don_splice_junctions = list(map(obj_k_value(7), examples))
                fake_inclusion_splice_junctions = list(map(obj_k_value(8), examples))
                fake_exclusion_splice_junctions = list(map(obj_k_value(9), examples))

                if len(examples[0]) == 12:
                    sample_name_list = list(map(obj_k_value(10), examples))
                    gene_name_list = list(map(obj_k_value(11), examples))
                    return new_alabels, new_dlabels, tlabels, transcripts_splice_junctions, batch_act_rep_tensor, sub_nts, sub_acc_splice_junctions, sub_don_splice_junctions, fake_inclusion_splice_junctions, fake_exclusion_splice_junctions, sample_name_list, gene_name_list
                
                return new_alabels, new_dlabels, tlabels, transcripts_splice_junctions, batch_act_rep_tensor, sub_nts, sub_acc_splice_junctions, sub_don_splice_junctions, fake_inclusion_splice_junctions, fake_exclusion_splice_junctions

            sample_name_list = list(map(obj_k_value(6), examples))
            gene_name_list = list(map(obj_k_value(7), examples))
            return new_alabels, new_dlabels, tlabels, transcripts_splice_junctions, batch_act_rep_tensor, sub_nts, sample_name_list, gene_name_list
        else:
            raise ValueError(
                "Invalid number of entries for training examples."
            )
    if max_eval_step is not None and max_eval_step < len(eval_dataset):
        eval_sampler = RandomSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    else:
        eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(spliceai_model, torch.nn.DataParallel):
        spliceai_model = torch.nn.DataParallel(spliceai_model)
    if args.n_gpu > 1 and not isinstance(site_aux_model, torch.nn.DataParallel):
        site_aux_model = torch.nn.DataParallel(site_aux_model)
    if args.n_gpu > 1 and not isinstance(spliceosome_model, torch.nn.DataParallel):
        spliceosome_model = torch.nn.DataParallel(spliceosome_model)

    # Eval!
    logger.info("***** Running spliceosome evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    start_time = time.time()
    eval_loss_reg = 0.0
    eval_loss_reg_transcript = 0.0
    eval_loss_cls = 0.0
    nb_eval_steps = 0
    spliceai_model.eval()
    site_aux_model.eval()
    spliceosome_model.eval()
    
    labels_psi_acc_list = []
    labels_psi_don_list = []
    pred_psi_acc_list = []
    pred_psi_don_list = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):


        if args.do_transcript_reg:
            alabels, dlabels, tlabels, transcripts_splice_junctions, act_rep_inputs, sub_nts, sample_name_list, gene_name_list = batch
        else:
            alabels, dlabels, transcripts_splice_junctions, act_rep_inputs, sub_nts, sample_name_list, gene_name_list = batch

        with torch.no_grad():
            # compute splice_site_embs with spliceai_model from alabels, dlabels and sub_nts
            splice_site_embs = []

            for gene_ind, gene_alabels in enumerate(alabels):
                gene_dlabels = dlabels[gene_ind]
                gene_sub_nts = sub_nts[gene_ind]

                # Load cache parameters
                sample_name = sample_name_list[gene_ind]
                gene_name = gene_name_list[gene_ind]
                gene_splice_sites = { **gene_alabels, **gene_dlabels, **gene_sub_nts }
                
                inputs = []
                splice_sites_pos = []
                cached_splice_sites_pos = []
                cached_site_embs = []

                if eval_cls:
                    labels_class = []
                    alabel_tensor = torch.from_numpy(np.array([1]))
                    dlabel_tensor = torch.from_numpy(np.array([2]))
                    sublabel_tensor = torch.from_numpy(np.array([0]))
                    for splice_site in gene_alabels.keys():
                        wseq = gene_alabels[splice_site]['wseq']
                        inputs.append(torch.tensor(wseq, dtype=torch.float))
                        splice_sites_pos.append(splice_site)
                        labels_class.append(alabel_tensor)

                    for splice_site in gene_dlabels.keys():
                        wseq = gene_dlabels[splice_site]['wseq']
                        inputs.append(torch.tensor(wseq, dtype=torch.float))
                        splice_sites_pos.append(splice_site)
                        labels_class.append(dlabel_tensor)

                    for splice_site in gene_sub_nts.keys():
                        wseq = gene_sub_nts[splice_site]['wseq']
                        inputs.append(torch.tensor(wseq, dtype=torch.float))
                        splice_sites_pos.append(splice_site)
                        labels_class.append(sublabel_tensor)
                else:
                    for splice_site in gene_splice_sites.keys():
                        if args.use_cached_rep_to_train and do_train_spliceai==False:
                            with h5py.File(cache_file_path, 'r') as f:
                                h5_group_path = '/'.join([sample_name, gene_name]) # <sample_name>/<gene_name>/<splice_site_pos>
                                if str(splice_site) in f[h5_group_path]:
                                    h5_dataset_path = '/'.join([sample_name, gene_name, str(splice_site)]) # <sample_name>/<gene_name>/<splice_site_pos>
                                    site_emb = f[h5_dataset_path][()] # (dim,)
                                    cached_splice_sites_pos.append(splice_site)
                                    cached_site_embs.append(site_emb)
                                else:
                                    wseq = gene_splice_sites[splice_site]['wseq']
                                    inputs.append(torch.tensor(wseq, dtype=torch.float))
                                    splice_sites_pos.append(splice_site)
                        else:
                            wseq = gene_splice_sites[splice_site]['wseq']
                            inputs.append(torch.tensor(wseq, dtype=torch.float))
                            splice_sites_pos.append(splice_site)

                gene_splice_site_embs = {}
                if len(inputs) != 0:
                    inputs = torch.stack(inputs, dim=0)
                    inputs = inputs.to(args.device)
                    gene_act_rep_input = act_rep_inputs[gene_ind:gene_ind+1]
                    gene_act_rep_input = gene_act_rep_input.to(args.device)

                    if eval_cls:
                        labels_class = torch.stack(labels_class, dim=0)
                        labels_class = labels_class.to(args.device)
                        loss_cls, _, site_hidden = spliceai_model(inputs, labels_class=labels_class)
                        eval_loss_cls += loss_cls.mean().item()
                    else:
                        _, site_hidden = spliceai_model(inputs)
                    site_aux_model_psi_reg, site_act_rep_final_hidden = site_aux_model(site_hidden, act_rep_input=gene_act_rep_input)

                    for site_ind, site_emb in enumerate(site_act_rep_final_hidden):
                        splice_site_pos = splice_sites_pos[site_ind]
                        gene_splice_site_embs[splice_site_pos] = site_emb
                
                gene_cached_splice_site_embs = {}
                if len(cached_site_embs) != 0:
                    for site_ind, site_emb in enumerate(cached_site_embs):
                        splice_site_pos = cached_splice_sites_pos[site_ind]
                        site_emb_tensor = torch.tensor(site_emb).to(args.device)
                        gene_cached_splice_site_embs[splice_site_pos] = site_emb_tensor

                gene_all_splice_site_embs = {**gene_cached_splice_site_embs, **gene_splice_site_embs}

                splice_site_embs.append(gene_all_splice_site_embs)

            # fake_transcripts_splice_junctions = [sub_acc + sub_don + fake_inc + fake_exc for sub_acc, sub_don, fake_inc, fake_exc in zip(sub_acc_splice_junctions, sub_don_splice_junctions, fake_inclusion_splice_junctions, fake_exclusion_splice_junctions)]
            if args.do_transcript_reg:
                outputs = spliceosome_model(splice_site_embs, transcripts_splice_junctions, fake_transcripts_splice_junctions=None, alabels=alabels, dlabels=dlabels, tlabels=tlabels)
                # outputs: (mean_reduced_loss_reg, mean_reduced_loss_reg_transcript, splice_sites_prob, transcripts_prob)
                loss_reg, loss_reg_transcript, splice_sites_prob, _ = outputs
                eval_loss_reg += loss_reg.mean().item()
                eval_loss_reg_transcript += loss_reg_transcript.mean().item()
            else:
                outputs = spliceosome_model(splice_site_embs, transcripts_splice_junctions, fake_transcripts_splice_junctions=None, alabels=alabels, dlabels=dlabels)
                # outputs: (loss_reg, splice_sites_prob)
                loss_reg, splice_sites_prob = outputs
                eval_loss_reg += loss_reg.mean().item()

            if eval_correlation:
                for gene_splice_site_prob_dict, gene_alabels, gene_dlabels in zip(splice_sites_prob, alabels, dlabels):     
            
                    for acc_pos in gene_alabels.keys():
                        label = gene_alabels[acc_pos]['psi']
                        labels_psi_acc_list.append(label)

                        pred = gene_splice_site_prob_dict[acc_pos].cpu().numpy()
                        pred_psi_acc_list.append(pred)

                    for don_pos in gene_dlabels.keys():
                        label = gene_dlabels[don_pos]['psi']
                        labels_psi_don_list.append(label)

                        pred = gene_splice_site_prob_dict[don_pos].cpu().numpy()
                        pred_psi_don_list.append(pred)
                              
            nb_eval_steps += 1

        if max_eval_step is not None and nb_eval_steps > max_eval_step:
            break

    eval_loss_reg = eval_loss_reg / nb_eval_steps
    if eval_cls:
        eval_loss_cls = eval_loss_cls / nb_eval_steps
        result = {"eval_loss_reg": eval_loss_reg, "eval_loss_cls": eval_loss_cls}
    else:
        result = {"eval_loss_reg": eval_loss_reg}

    if args.do_transcript_reg:
        eval_loss_reg_transcript = eval_loss_reg_transcript / nb_eval_steps
        result = {**result, "result": eval_loss_reg_transcript}

    if eval_correlation:
        pred_psi_acc_list = np.stack(pred_psi_acc_list)
        pred_psi_don_list = np.stack(pred_psi_don_list)
        labels_psi_acc_list = np.stack(labels_psi_acc_list)
        labels_psi_don_list = np.stack(labels_psi_don_list)

        pred_psi_ss_list = np.concatenate([pred_psi_acc_list, pred_psi_don_list])
        labels_psi_ss_list = np.concatenate([labels_psi_acc_list, labels_psi_don_list])
        
        acc_cor_result = pearson_and_spearman(pred_psi_acc_list, labels_psi_acc_list)
        don_cor_result = pearson_and_spearman(pred_psi_don_list, labels_psi_don_list)
        all_ss_cor_result = pearson_and_spearman(pred_psi_ss_list, labels_psi_ss_list)

        correlation_result = {"spearmanr_acc": acc_cor_result['spearmanr'], "pearson_acc": acc_cor_result['pearson'], 
                            "spearmanr_don": don_cor_result['spearmanr'], "pearson_don": don_cor_result['pearson'], 
                            "spearmanr_ss": all_ss_cor_result['spearmanr'], "pearson_ss": all_ss_cor_result['pearson']}
        result = {**correlation_result, **result}


    output_eval_file = os.path.join(eval_output_dir, prefix, eval_output_filename)
    if args.local_rank in [-1, 0]:
        os.makedirs(os.path.join(eval_output_dir, prefix), exist_ok=True)
    end_time = time.time()
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval splicesome models results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("***** Eval splicesome models results {} *****".format(prefix))
            writer.write("%s = %s\n" % (key, str(result[key])))

        hrs, mins, secs = time_to_human(end_time - start_time)
        writer.write("Overall time elapsed: {} hrs {} mins {} seconds".format(int(hrs), int(mins), round(secs)))
    
    if save_predlabels_as_npy:
        with open(os.path.join(eval_output_dir, prefix, 'pred_psi_acc.npy'), 'wb') as f:
            np.save(f, pred_psi_acc_list)
        with open(os.path.join(eval_output_dir, prefix, 'pred_psi_don.npy'), 'wb') as f:
            np.save(f, pred_psi_don_list)
        with open(os.path.join(eval_output_dir, prefix, 'labels_psi_acc.npy'), 'wb') as f:
            np.save(f, labels_psi_acc_list)
        with open(os.path.join(eval_output_dir, prefix, 'labels_psi_don.npy'), 'wb') as f:
            np.save(f, labels_psi_don_list)

    return result

def predict_ss_transcript_prob_spliceosome_model(args, spliceai_model, site_aux_model, spliceosome_model, pred_dataset, pred_output_dir=None, pred_output_filename=None, prefix="", gene_list=None) -> Dict:

    """
    predict and save splice site and transcript probabilities in json format
    {<sample_name> : 
        {<gene>:
            {
            "transcript_probs": ..,
            "splice_site_probs": ..,
            "splice_site_alabels": ..,
            "splice_site_dlabels": ..,
            }
            ..
        }
    }
    transcript string formatted as :"-1^exon1start_exon1end^exon2start_exon2end^...^9999999": 
    """
    
    if pred_output_dir is None:
        pred_output_dir = args.output_dir

    if pred_output_filename is None:
        pred_output_filename = args.pred_output_filename

    if args.local_rank in [-1, 0]:
        os.makedirs(pred_output_dir, exist_ok=True)

    args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    def collate(examples: List[torch.Tensor]):
        # new_alabels, new_dlabels, transcripts_splice_junctions, act_rep_tensor, sub_nts,
        # sub_acc_splice_junctions, sub_don_splice_junctions, fake_inclusion_splice_junctions, fake_exclusion_splice_junctions
        if len(examples[0]) == 11 or len(examples[0]) == 9 or len(examples[0]) == 7:
            new_alabels = list(map(obj_k_value(0), examples))
            new_dlabels = list(map(obj_k_value(1), examples))
            transcripts_splice_junctions = list(map(obj_k_value(2), examples))
            act_rep_tensor_list = list(map(obj_k_value(3), examples))
            batch_act_rep_tensor = torch.stack(act_rep_tensor_list)
            sub_nts = list(map(obj_k_value(4), examples))
            if len(examples[0]) >= 9:
                sub_acc_splice_junctions = list(map(obj_k_value(5), examples))
                sub_don_splice_junctions = list(map(obj_k_value(6), examples))
                fake_inclusion_splice_junctions = list(map(obj_k_value(7), examples))
                fake_exclusion_splice_junctions = list(map(obj_k_value(8), examples))
                if len(examples[0]) == 11:
                    sample_name_list = list(map(obj_k_value(9), examples))
                    gene_name_list = list(map(obj_k_value(10), examples))
                    return new_alabels, new_dlabels, transcripts_splice_junctions, batch_act_rep_tensor, sub_nts, sub_acc_splice_junctions, sub_don_splice_junctions, fake_inclusion_splice_junctions, fake_exclusion_splice_junctions, sample_name_list, gene_name_list
                return new_alabels, new_dlabels, transcripts_splice_junctions, batch_act_rep_tensor, sub_nts, sub_acc_splice_junctions, sub_don_splice_junctions, fake_inclusion_splice_junctions, fake_exclusion_splice_junctions

            
            sample_name_list = list(map(obj_k_value(5), examples))
            gene_name_list = list(map(obj_k_value(6), examples))
            return new_alabels, new_dlabels, transcripts_splice_junctions, batch_act_rep_tensor, sub_nts, sample_name_list, gene_name_list
        else:
            raise ValueError(
                "Invalid number of entries for training examples."
            )
  
    pred_sampler = SequentialSampler(pred_dataset)
    pred_dataloader = DataLoader(
        pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size, collate_fn=collate
    )

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(spliceai_model, torch.nn.DataParallel):
        spliceai_model = torch.nn.DataParallel(spliceai_model)
    if args.n_gpu > 1 and not isinstance(site_aux_model, torch.nn.DataParallel):
        site_aux_model = torch.nn.DataParallel(site_aux_model)
    if args.n_gpu > 1 and not isinstance(spliceosome_model, torch.nn.DataParallel):
        spliceosome_model = torch.nn.DataParallel(spliceosome_model)

    # Eval!
    logger.info("***** Running spliceosome evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(pred_dataset))
    logger.info("  Batch size = %d", args.pred_batch_size)
    start_time = time.time()
    nb_pred_steps = 0
    spliceai_model.eval()
    site_aux_model.eval()
    spliceosome_model.eval()
    
    # dictionary of transcript probabilties (pred) and splice sites' psi values (pred and label)
    """
    {<sample_name> : 
        {<gene>:
            {
            "transcript_probs": ..,
            "splice_site_probs": ..,
            "splice_site_alabels": ..,
            "splice_site_dlabels": ..,
            }
            ..
        }
    }
    """
    pred_dict = {}

    for batch in tqdm(pred_dataloader, desc="Evaluating"):

        alabels, dlabels, transcripts_splice_junctions, act_rep_inputs, sub_nts, sample_name_list, gene_name_list = batch

        with torch.no_grad():
            # compute splice_site_embs with spliceai_model from alabels, dlabels and sub_nts
            splice_site_embs = []
            batch_sample_name = []
            batch_gene_name = []

            for gene_ind, gene_alabels in enumerate(alabels):
                gene_dlabels = dlabels[gene_ind]
                gene_sub_nts = sub_nts[gene_ind]

                # Load cache parameters
                sample_name = sample_name_list[gene_ind]
                gene_name = gene_name_list[gene_ind]

                if gene_list is None or (gene_list is not None and gene_name in gene_list):
                    
                    batch_sample_name.append(sample_name)
                    batch_gene_name.append(gene_name)

                    if sample_name not in pred_dict:
                        pred_dict[sample_name] = {}

                    if gene_name in pred_dict[sample_name]:
                        raise ValueError(
                            "pred_dict[sample_name] already has results for gene_name"
                        )

                    pred_dict[sample_name][gene_name] = {}

                    saved_gene_alabels = {}
                    for ss in gene_alabels:
                        saved_gene_alabels[ss] = gene_alabels[ss]['psi']
                    saved_gene_dlabels = {}
                    for ss in gene_dlabels:
                        saved_gene_dlabels[ss] = gene_dlabels[ss]['psi']

                    pred_dict[sample_name][gene_name]["splice_site_alabels"] = saved_gene_alabels
                    pred_dict[sample_name][gene_name]["splice_site_dlabels"] = saved_gene_dlabels

                    gene_splice_sites = { **gene_alabels, **gene_dlabels, **gene_sub_nts }
                    
                    inputs = []
                    splice_sites_pos = []

                    for splice_site in gene_splice_sites.keys():
                        wseq = gene_splice_sites[splice_site]['wseq']
                        inputs.append(torch.tensor(wseq, dtype=torch.float))
                        splice_sites_pos.append(splice_site)

                    gene_splice_site_embs = {}
                    if len(inputs) != 0:
                        inputs = torch.stack(inputs, dim=0)
                        inputs = inputs.to(args.device)
                        gene_act_rep_input = act_rep_inputs[gene_ind:gene_ind+1]
                        gene_act_rep_input = gene_act_rep_input.to(args.device)

                        _, site_hidden = spliceai_model(inputs)
                        site_aux_model_psi_reg, site_act_rep_final_hidden = site_aux_model(site_hidden, act_rep_input=gene_act_rep_input)

                        for site_ind, site_emb in enumerate(site_act_rep_final_hidden):
                            splice_site_pos = splice_sites_pos[site_ind]
                            gene_splice_site_embs[splice_site_pos] = site_emb                    

                    gene_all_splice_site_embs = gene_splice_site_embs

                    splice_site_embs.append(gene_all_splice_site_embs)
            
            if len(splice_site_embs) != 0:
                outputs = spliceosome_model(splice_site_embs, transcripts_splice_junctions, fake_transcripts_splice_junctions=None, alabels=alabels, dlabels=dlabels, return_transcript_probs=True, output_np_prob=True)
                # outputs: (loss_reg, splice_sites_prob)
                loss_reg, splice_sites_prob, transcripts_prob = outputs
                for pred_ind, sample_name in enumerate(batch_sample_name):
                    gene_name = batch_gene_name[pred_ind]
                    sample_splice_sites_prob = splice_sites_prob[pred_ind]
                    sample_transcripts_prob = transcripts_prob[pred_ind]

                    pred_dict[sample_name][gene_name]["splice_site_probs"] = sample_splice_sites_prob
                    pred_dict[sample_name][gene_name]["transcript_probs"] = sample_transcripts_prob

    output_pred_file = os.path.join(pred_output_dir, prefix, pred_output_filename)
    if args.local_rank in [-1, 0]:
        os.makedirs(os.path.join(pred_output_dir, prefix), exist_ok=True)
    end_time = time.time()
    with open(output_pred_file, "w") as writer:
        json.dump(pred_dict, writer, cls=NumpyEncoder)

    hrs, mins, secs = time_to_human(end_time - start_time)
    logger.info("Overall time elapsed: {} hrs {} mins {} seconds".format(int(hrs), int(mins), round(secs)))

    return 

def evaluate_spliceai_fast(args, spliceai_model, site_aux_model, eval_dataset, eval_output_dir=None, eval_output_filename=None, prefix="", 
        eval_correlation=True, max_eval_step=None, eval_cls=False, save_predlabels_as_npy=True, ablation_finetune_model=None, eval_cor_with_spliceaicls=False) -> Dict:
    logger.info("!! evaluate_spliceai_fast !!")
    if eval_output_dir is None:
        eval_output_dir = args.output_dir

    if eval_output_filename is None:
        eval_output_filename = args.eval_output_filename

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    # cached representation dir
    if args.splice_site_cache_dir is not None and args.use_cached_rep_to_train:
        cache_file_path = os.path.join(args.splice_site_cache_dir, 'sequence_representations.hdf5')

    def collate(examples: List[torch.Tensor]):
        """
        examples: new_alabels, new_dlabels, transcripts_splice_junctions, act_rep_tensor, sub_nts,
        sub_acc_splice_junctions, sub_don_splice_junctions, fake_inclusion_splice_junctions, fake_exclusion_splice_junctions
        """
        if len(examples[0]) == 11 or len(examples[0]) == 9 or len(examples[0]) == 7:
            new_alabels = list(map(obj_k_value(0), examples))
            new_dlabels = list(map(obj_k_value(1), examples))
            transcripts_splice_junctions = list(map(obj_k_value(2), examples))
            act_rep_tensor_list = list(map(obj_k_value(3), examples))
            batch_act_rep_tensor = torch.stack(act_rep_tensor_list)
            sub_nts = list(map(obj_k_value(4), examples))
            if len(examples[0]) >= 9:
                sub_acc_splice_junctions = list(map(obj_k_value(5), examples))
                sub_don_splice_junctions = list(map(obj_k_value(6), examples))
                fake_inclusion_splice_junctions = list(map(obj_k_value(7), examples))
                fake_exclusion_splice_junctions = list(map(obj_k_value(8), examples))
                if len(examples[0]) == 11:
                    sample_name_list = list(map(obj_k_value(9), examples))
                    gene_name_list = list(map(obj_k_value(10), examples))
                    return new_alabels, new_dlabels, transcripts_splice_junctions, batch_act_rep_tensor, sub_nts, sub_acc_splice_junctions, sub_don_splice_junctions, fake_inclusion_splice_junctions, fake_exclusion_splice_junctions, sample_name_list, gene_name_list
                return new_alabels, new_dlabels, transcripts_splice_junctions, batch_act_rep_tensor, sub_nts, sub_acc_splice_junctions, sub_don_splice_junctions, fake_inclusion_splice_junctions, fake_exclusion_splice_junctions

            
            sample_name_list = list(map(obj_k_value(5), examples))
            gene_name_list = list(map(obj_k_value(6), examples))
            return new_alabels, new_dlabels, transcripts_splice_junctions, batch_act_rep_tensor, sub_nts, sample_name_list, gene_name_list
        else:
            raise ValueError(
                "Invalid number of entries for training examples."
            )
    if max_eval_step is not None and max_eval_step < len(eval_dataset):
        eval_sampler = RandomSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    else:
        eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(spliceai_model, torch.nn.DataParallel):
        spliceai_model = torch.nn.DataParallel(spliceai_model)
    if args.n_gpu > 1 and not isinstance(site_aux_model, torch.nn.DataParallel):
        site_aux_model = torch.nn.DataParallel(site_aux_model)
    if ablation_finetune_model and args.n_gpu > 1 and not isinstance(ablation_finetune_model, torch.nn.DataParallel):
        ablation_finetune_model = torch.nn.DataParallel(ablation_finetune_model)

    # Eval!
    logger.info("***** Running FAST spliceai evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    start_time = time.time()
    eval_loss_reg = 0.0
    eval_loss_cls = 0.0
    nb_eval_steps = 0
    spliceai_model.eval()
    site_aux_model.eval()
    if ablation_finetune_model:
        ablation_finetune_model.eval()
    
    labels_psi_acc_list = []
    labels_psi_don_list = []
    pred_psi_acc_list = []
    pred_psi_don_list = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):

        alabels, dlabels, transcripts_splice_junctions, act_rep_inputs, sub_nts, sample_name_list, gene_name_list = batch

        with torch.no_grad():
            # compute splice_site_embs with spliceai_model from alabels, dlabels and sub_nts
            splice_site_embs = []

            for gene_ind, gene_alabels in enumerate(alabels):
                gene_dlabels = dlabels[gene_ind]
                gene_sub_nts = sub_nts[gene_ind]

                # Load cache parameters
                sample_name = sample_name_list[gene_ind]
                gene_name = gene_name_list[gene_ind]
                gene_splice_sites = { **gene_alabels, **gene_dlabels, **gene_sub_nts }
                
                inputs = []
                splice_sites_pos = []
                cached_splice_sites_pos = []
                cached_site_embs = []

                if eval_cls:
                    labels_class = []
                    alabel_tensor = torch.from_numpy(np.array([1]))
                    dlabel_tensor = torch.from_numpy(np.array([2]))
                    sublabel_tensor = torch.from_numpy(np.array([0]))
                    for splice_site in gene_alabels.keys():
                        wseq = gene_alabels[splice_site]['wseq']
                        inputs.append(torch.tensor(wseq, dtype=torch.float))
                        splice_sites_pos.append(splice_site)
                        labels_class.append(alabel_tensor)

                    for splice_site in gene_dlabels.keys():
                        wseq = gene_dlabels[splice_site]['wseq']
                        inputs.append(torch.tensor(wseq, dtype=torch.float))
                        splice_sites_pos.append(splice_site)
                        labels_class.append(dlabel_tensor)

                    for splice_site in gene_sub_nts.keys():
                        wseq = gene_sub_nts[splice_site]['wseq']
                        inputs.append(torch.tensor(wseq, dtype=torch.float))
                        splice_sites_pos.append(splice_site)
                        labels_class.append(sublabel_tensor)
                else:
                    for splice_site in gene_splice_sites.keys():
                        wseq = gene_splice_sites[splice_site]['wseq']
                        inputs.append(torch.tensor(wseq, dtype=torch.float))
                        splice_sites_pos.append(splice_site)

                gene_splice_site_prob_dict = {}
                if len(inputs) != 0:
                    inputs = torch.stack(inputs, dim=0)
                    inputs = inputs.to(args.device)
                    gene_act_rep_input = act_rep_inputs[gene_ind:gene_ind+1]
                    gene_act_rep_input = gene_act_rep_input.to(args.device)

                    if eval_cls:
                        labels_class = torch.stack(labels_class, dim=0)
                        labels_class = labels_class.to(args.device)
                        loss_cls, logits_class, site_hidden = spliceai_model(inputs, labels_class=labels_class)
                        eval_loss_cls += loss_cls.mean().item()
                    else:
                        logits_class, site_hidden = spliceai_model(inputs)

                    if eval_cor_with_spliceaicls:
                        probs_class = nn.functional.softmax(logits_class, dim=1)
                        for site_ind, site_prob in enumerate(probs_class):
                            splice_site_pos = splice_sites_pos[site_ind]
                            gene_splice_site_prob_dict[splice_site_pos] = site_prob
                    else:
                        if ablation_finetune_model:
                            _, site_act_rep_final_hidden = site_aux_model(site_hidden, act_rep_input=gene_act_rep_input, squeeze_site_act_rep_final_hidden=False)
                            ablation_finetune_model_outputs = ablation_finetune_model(site_act_rep_final_hidden)
                            psi_reg = ablation_finetune_model_outputs[0]
                        else:
                            psi_reg, site_act_rep_final_hidden = site_aux_model(site_hidden, act_rep_input=gene_act_rep_input)

                        for site_ind, site_psi in enumerate(psi_reg):
                            splice_site_pos = splice_sites_pos[site_ind]
                            gene_splice_site_prob_dict[splice_site_pos] = site_psi[0]
                
                if eval_cor_with_spliceaicls:
                    for acc_pos in gene_alabels.keys():
                        label = gene_alabels[acc_pos]['psi']
                        labels_psi_acc_list.append(label)

                        pred = gene_splice_site_prob_dict[acc_pos][1][0].cpu().numpy()
                        pred_psi_acc_list.append(pred)

                    for don_pos in gene_dlabels.keys():
                        label = gene_dlabels[don_pos]['psi']
                        labels_psi_don_list.append(label)

                        pred = gene_splice_site_prob_dict[don_pos][2][0].cpu().numpy()
                        pred_psi_don_list.append(pred)

                elif eval_correlation:
                    for acc_pos in gene_alabels.keys():
                        label = gene_alabels[acc_pos]['psi']
                        labels_psi_acc_list.append(label)

                        pred = gene_splice_site_prob_dict[acc_pos].cpu().numpy()
                        pred_psi_acc_list.append(pred)

                    for don_pos in gene_dlabels.keys():
                        label = gene_dlabels[don_pos]['psi']
                        labels_psi_don_list.append(label)

                        pred = gene_splice_site_prob_dict[don_pos].cpu().numpy()
                        pred_psi_don_list.append(pred)

                              
            nb_eval_steps += 1

        if max_eval_step is not None and nb_eval_steps > max_eval_step:
            break

    if eval_correlation:
        pred_psi_acc_list = np.stack(pred_psi_acc_list)
        pred_psi_don_list = np.stack(pred_psi_don_list)
        labels_psi_acc_list = np.stack(labels_psi_acc_list)
        labels_psi_don_list = np.stack(labels_psi_don_list)

        pred_psi_ss_list = np.concatenate([pred_psi_acc_list, pred_psi_don_list])
        labels_psi_ss_list = np.concatenate([labels_psi_acc_list, labels_psi_don_list])
        
        acc_cor_result = pearson_and_spearman(pred_psi_acc_list, labels_psi_acc_list)
        don_cor_result = pearson_and_spearman(pred_psi_don_list, labels_psi_don_list)
        all_ss_cor_result = pearson_and_spearman(pred_psi_ss_list, labels_psi_ss_list)

        correlation_result = {"spearmanr_acc": acc_cor_result['spearmanr'], "pearson_acc": acc_cor_result['pearson'], 
                            "spearmanr_don": don_cor_result['spearmanr'], "pearson_don": don_cor_result['pearson'], 
                            "spearmanr_ss": all_ss_cor_result['spearmanr'], "pearson_ss": all_ss_cor_result['pearson']}
        result = {**correlation_result}


    output_eval_file = os.path.join(eval_output_dir, prefix, eval_output_filename)
    if args.local_rank in [-1, 0]:
        os.makedirs(os.path.join(eval_output_dir, prefix), exist_ok=True)
    end_time = time.time()
    with open(output_eval_file, "a") as writer:
        logger.info("***** FAST Eval spliceai models results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("***** FAST Eval spliceai models results {} *****".format(prefix))
            writer.write("%s = %s\n" % (key, str(result[key])))

        hrs, mins, secs = time_to_human(end_time - start_time)
        writer.write("Overall time elapsed: {} hrs {} mins {} seconds".format(int(hrs), int(mins), round(secs)))
    
    if save_predlabels_as_npy:
        with open(os.path.join(eval_output_dir, prefix, 'pred_psi_acc.npy'), 'wb') as f:
            np.save(f, pred_psi_acc_list)
        with open(os.path.join(eval_output_dir, prefix, 'pred_psi_don.npy'), 'wb') as f:
            np.save(f, pred_psi_don_list)
        with open(os.path.join(eval_output_dir, prefix, 'labels_psi_acc.npy'), 'wb') as f:
            np.save(f, labels_psi_acc_list)
        with open(os.path.join(eval_output_dir, prefix, 'labels_psi_don.npy'), 'wb') as f:
            np.save(f, labels_psi_don_list)

    return result

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_data_dir", default=None, type=str, required=True, help="The input training data directory for training sample (each patient as a jsonl file)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--eval_data_dir",
        default=None,
        type=str,
        help="An optional input evaluation data directory to evaluate the model on (each patient as a jsonl file).",
    )
    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")

    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_train_spliceai_batch_size", default=0, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_train_spliceosome_model_batch_size", default=0, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=1, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_spliceai_epochs.",
    )
    parser.add_argument(
        "--num_train_spliceai_epochs", default=1, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--num_train_only_spliceosome_model_epochs", default=0, type=float, help="Total number of training epochs to perform for only spliceosome_model."
    )
    parser.add_argument(
        "--num_train_full_spliceai_spliceosome_model_epochs", default=0, type=float, help="Total number of training epochs to perform for both spliceosome_model and spliceai_model on the exon inclusion regression task."
    )
    
    parser.add_argument(
        "--spliceai_max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_spliceai_epochs.",
    )
    
    parser.add_argument(
        "--spliceosome_max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_only_spliceosome_model_epochs.",
    )
    
    parser.add_argument(
        "--full_spliceai_spliceosome_max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_full_spliceai_spliceosome_model_epochs.",
    )
    
    parser.add_argument("--num_constant_lr_epochs_spliceai", default=1, type=int, help="Constant LR for spliceai over constant_lr_epochs.")
    parser.add_argument("--num_constant_lr_epochs_spliceosomenet", default=6, type=int, help="Constant LR for spliceai over constant_lr_epochs.")
    parser.add_argument("--warmup_steps_spliceai", default=0, type=int, help="Linear warmup over warmup_steps_spliceai in spliceai training.")
    parser.add_argument("--warmup_steps_spliceosomenet", default=0, type=int, help="Linear warmup over warmup_steps_spliceosomenet in spliceosomenet training.")

    parser.add_argument(
        "--train_spliceai_opt_scheduler_type",
        type=str,
        default="epoch",
        help="Type of time interval to step up optimization scheduler ['epoch', 'step'].",
    )
    parser.add_argument(
        "--train_spliceosomenet_opt_scheduler_type",
        type=str,
        default="step",
        help="Type of time interval to step up optimization scheduler ['epoch', 'step'].",
    )
    
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--logging_valid_steps", type=int, default=1000, help="Log evaluation results every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=-1, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )

    parser.add_argument(
        "--eval_output_filename",
        type=str,
        default="eval_results.txt",
        help="The output file to save eval results.",
    )
    parser.add_argument("--eval_compute_without_checkpoint", action="store_true", help="Whether to use saved checkpoint or use pretrained spliceai to evaluate and compute stats.")

    parser.add_argument("--track_loss_gradnorms", action="store_true", help="Whether to log all loss gradnorm to tb.")
    
    parser.add_argument("--only_classification", action="store_true", help="Whether to train and infer only lm model, without adaIN.")

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    parser.add_argument("--generate_length", type=int, default=20)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument("--initializer_range", default=0.02, type=float, help="Random initialization range.")

    # rna splicing model args

    parser.add_argument(
        "--spliceai_model_training_type",
        default='spliceai_cls_reg',
        type=str,
        help="Type of splice model, choices: [spliceai_cls, spliceai_cls_reg(default) ]",
    )
    parser.add_argument(
        "--spliceosome_model_training_type",
        default='site_aux_spliceosome',
        type=str,
        help="Type of spliceosome model training, full: train the spliceai, site_aux and spliceosome models all tgt, choices: [spliceosome_only, site_aux_spliceosome(default) ]",
    )
    parser.add_argument(
        "--spliceosome_model_type",
        default='spliceosome_full',
        type=str,
        help="Type of splicesome models, choices: [spliceosome_junction_reg, spliceosome_full (default)]",
    )
    
    # data processing args
    parser.add_argument(
        "--max_main_seq_len_spliceai", 
        type=int, 
        default=5000, 
        help="Max input sequence length for spliceai model while computing splice site spliceai representations."
    )
    parser.add_argument(
        "--input_flank_context",
        type=float,
        default=1000,
        help="number of additional nt added to the two ends of gene sequence to provide context for the input",
    )
    parser.add_argument(
        "--max_train_seq_len",
        type=float,
        default=7000,
        help="maximum length of gene nt sequences used to train spliceai model for each training sample",
    )
    parser.add_argument(
        "--log_act_rep",
        action="store_true",
        help="Whether to apply log op to activator/repressor input values before normalization.",
    )
    parser.add_argument(
        "--c_log_act_rep",
        type=float,
        default=None,
        help="Constant to add to rep values before applying log transform on activator/repressor input values.",
    )
    parser.add_argument(
        "--act_rep_train_file",
        default=None,
        type=str,
        help="File containing the activator and repressor expression levels for the training (patient) samples.",
    )
    parser.add_argument(
        "--act_rep_eval_file",
        default=None,
        type=str,
        help="File containing the activator and repressor expression levels for the evaluation (patient) samples.",
    )
    parser.add_argument(
        "--gene_seq_dict_file", default=None, type=str, required=True, help="The file for gene sequences."
    )
    parser.add_argument(
        "--lambda_loss_class",
        type=float,
        default=1.0,
        help="Weightage of classification loss in spliceai training",
    )
    parser.add_argument(
        "--lambda_loss_reg",
        type=float,
        default=1.0,
        help="Weightage of regression loss in spliceai training",
    )
    parser.add_argument(
        "--lambda_loss_class_spliceosome_train",
        type=float,
        default=1.0,
        help="Weightage of classification loss in spliceosome model training",
    )
    parser.add_argument(
        "--lambda_loss_reg_spliceosome_train",
        type=float,
        default=1.0,
        help="Weightage of regression loss in spliceosome model training",
    )
    parser.add_argument(
        "--lambda_loss_reg_transcript_spliceosome_train",
        type=float,
        default=1.0,
        help="Weightage of regression loss in spliceosome model training for transcript probs",
    )  
    parser.add_argument(
        "--valid_data_dir",
        default=None,
        type=str,
        help="An optional input valid data directory to evaluate the model on (each patient as a jsonl file).",
    )
    parser.add_argument(
        "--act_rep_valid_file",
        default=None,
        type=str,
        help="File containing the activator and repressor expression levels for the valid (patient) samples.",
    )
    parser.add_argument(
        "--max_valid_step", type=int, default=500,
        help="Number of validation step to take.",
    )
    parser.add_argument(
        "--no_tissue_type_as_feature", action="store_true", help="Whether to use tissue type as input feature"
    )

    # model loading and saving args
    parser.add_argument(
        "--load_pretrained_spliceai_path",
        default=None,
        type=str,
        help="File containing saved spliceai model weights.",
    )
    parser.add_argument(
        "--load_pretrained_site_aux_model_path",
        default=None,
        type=str,
        help="File containing saved Site Aux model weights.",
    )
    parser.add_argument(
        "--load_pretrained_spliceosome_model_path",
        default=None,
        type=str,
        help="File containing saved spliceosome model weights.",
    )
    parser.add_argument("--save_spliceai_model", action="store_true", help="Whether to save spliceai model to output_dir even without training.")
    parser.add_argument("--save_spliceosome_model", action="store_true", help="Whether to save spliceosome model to output_dir even without training.")
    parser.add_argument("--save_model_weights_even_wo_train", action="store_true", help="Whether to save spliceai and spliceosome model weights in output_dir no matter what.")
    parser.add_argument(
        "--eval_before_training", action="store_true", help="Whether to evaluate model before training"
    )
    
    # regression task args
    parser.add_argument(
        "--max_fake_inclusion_transcripts", type=int, default=5,
        help="Number of fake transcripts with false inclusion to create to augment training sample in exon-inclusion regression task.",
    )
    parser.add_argument(
        "--max_fake_exclusion_transcripts", type=int, default=5,
        help="Number of fake transcripts with false exclusion to create to augment training sample in exon-inclusion regression task.",
    )
    parser.add_argument(
        "--gene_start_token_position", type=int, default=-1,
        help="Integer token to represent start of a gene.",
    )
    parser.add_argument(
        "--gene_end_token_position", type=int, default=9999999,
        help="Integer token to represent start of a gene.",
    )
    parser.add_argument(
        "--augment_transcript_data", action="store_true", help="Whether to augment training data with fake transcripts"
    )
    parser.add_argument(
        "--reg_include_nonss_prob",
        type=float,
        default=0.00001,
        help="Probability of including a non-splice site nt in SpliceAI regression training.",
    )
    parser.add_argument("--do_spliceai_cls_in_full_spliceosome_train", action="store_true", help="Whether to train spliceai cls during full spliceosome training.")
    
    # splice site subsitution args
    parser.add_argument(
        "--max_fake_substitution_transcripts", type=int, default=5,
        help="Number of fake transcripts with splice site substituted out to create to augment training sample in exon-inclusion regression task.",
    )
    parser.add_argument(
        "--num_sampled_sub_per_acc_don", type=float, default=1,
        help="Number of randomly sampled nt to substitute or augment with each acc_don junction.",
    )
    parser.add_argument(
        "--sampled_sub_nt_distance", type=int, default=5,
        help="Max distance from the original splice site where sub_nt are been sampled.",
    )
    parser.add_argument(
        "--prob_sub_per_exon",
        type=float,
        default=0.1,
        help="Probability of a particular exon being augmented with a splice site being substituted",
    )

    # SpliceAI model args
    parser.add_argument(
        "--seq_input_dim", type=int, default=4,
        help="Gene sequence input dimension, defaults to 4 (number of possible nucleotides, ATCG).",
    )
    parser.add_argument(
        "--spliceai_channels", type=int, default=32,
        help="Number of channels in SpliceAI module, default=32 to match original spliceai paper.",
    )
    parser.add_argument(
        "--num_act_rep_layers", type=int, default=3,
        help="Number of dense layers for the act_rep module.",
    )
    parser.add_argument(
        "--act_rep_channels", type=int, default=32,
        help="Number of channels in each dense layer for the act_rep module.",
    )
    parser.add_argument(
        "--num_site_act_rep_layers", type=int, default=2,
        help="Number of dense layers for the site_act_rep merging module.",
    )
    parser.add_argument(
        "--site_act_rep_channels", type=int, default=32,
        help="Number of channels in each dense layer for the site_act_rep merging module.",
    )
    parser.add_argument(
        "--act_rep_dim", type=int, default=3987,
        help="Dimension of act_rep input, i.e. the number of expression levels in the activator/repressor array data, includes cell type onehot dim.",
    )

    parser.add_argument(
        "--act_rep_input_dropout_prob", type=float, default=0,
        help="Dropout probability of act_rep input before passing it into the SiteAuxNet model.",
    )
    parser.add_argument(
        "--act_rep_activation_dropout_prob", type=float, default=0,
        help="Dropout probability of act_rep_hidden inside the SiteAuxNet model.",
    )
    
    parser.add_argument("--spliceai_dynamic_reweight_none_class", action="store_true", help="Whether to dynamically reweight none class loss values in classification task, to mitigate none class imbalance.")
    parser.add_argument(
        "--none_class_reweight_factor",
        type=float,
        default=None,
        help="Reweighting factor for none class loss values in classification task, to mitigate none class imbalance.",
    )    
    parser.add_argument(
        "--keep_none_cls_prob",
        type=float,
        default=None,
        help="Probability to keep none class loss values in classification task, to mitigate none class imbalance, ideal: 0.0016327469416563408=(1/612.4647821943244) or 0.0011864588546026463=(1/842.8442302239864) for 10K len cutoff.",
    )
    

    # SpliceosomeNet model args
    parser.add_argument(
        "--spliceosome_model_channels", type=int, default=32,
        help="Number of channels in each dense layer for the spliceosome model.",
    )
    parser.add_argument(
        "--spliceosome_num_layer", type=int, default=2,
        help="Number of dense layers for the spliceosome model.",
    )
    parser.add_argument("--train_spliceai_with_spliceosome_model", action="store_true", help="Whether to also train spliceai during spliceosome model training.")

    # Ablation baseline with more layers args
    parser.add_argument(
        "--ablation_num_more_layers", type=int, default=1,
        help="Number of layers for ablation model that adds more layer to match spliceai model with spliceosome model.",
    )
    parser.add_argument(
        "--num_train_ablation_more_layers_epochs", type=int, default=0,
        help="Number of training epochs for ablation baseline.",
    )
    parser.add_argument(
        "--ablation_more_layers_max_steps", type=int, default=-1,
        help="Max number of training steps for ablation baseline.",
    )

    # Cached representation args
    parser.add_argument(
        "--splice_site_cache_dir",
        default=None,
        type=str,
        help="Directory to store the cached splice site spliceai representations",
    )
    parser.add_argument("--cache_spliceai_final_hidden", action="store_true", help="Whether to cache splice site spliceai representations.")
    parser.add_argument("--infer_save_spliceai_batch_size", default=1, type=int, help="Batch size per GPU/CPU for inferring and saving splice site spliceai representations.")
    parser.add_argument(
        "--overwrite_spliceai_cache", action="store_true", help="Overwrite the cached spliceai representations."
    )
    parser.add_argument(
        "--spliceai_cache_dataset_type",
        default='point',
        type=str,
        help="Types of dataset to store spliceai representations, [point, sequence]",
    )
    parser.add_argument("--max_spliceai_forward_batch", default=500, type=int, help="Batch size per GPU/CPU for inferring spliceai representations.")
    parser.add_argument("--use_cached_rep_to_train", action="store_true", help="Whether to use cached splice site spliceai representations for spliceosome model training.")
    
    # Evaluation args
    parser.add_argument('--eval_patient_list', nargs='+', default=None, help='Patient filter which includes tissue type and patient ID to use for evaluation, default uses all in eval_data_dir')
    parser.add_argument('--eval_chr_list', nargs='+', default=None, help='Chromosome data to use for evaluation, default uses all in eval_data_dir')
    parser.add_argument('--eval_gene_list', nargs='+', default=None, help='Gene names to use for evaluation, default uses all in eval_data_dir')
    parser.add_argument(
        "--eval_output_dir_prefix",
        default="",
        type=str,
        help="Prefix for evaluation output results.",
    )

    # Inference args
    parser.add_argument("--do_infer", action="store_true", help="Whether to infer splice site and transcript probability predictions.")
    parser.add_argument(
        "--pred_output_filename",
        type=str,
        default="predictions.jsonl",
        help="The output file to save pred results.",
    )
    parser.add_argument(
        "--pred_output_dir_prefix",
        default="",
        type=str,
        help="Prefix for evaluation output results.",
    )
    parser.add_argument(
        "--per_gpu_pred_batch_size", default=1, type=int, help="Batch size per GPU/CPU for evaluation."
    )

    # transcript prob inference
    parser.add_argument("--do_transcript_reg", action="store_true", help="Whether to do transcript regression training.")
    
    args = parser.parse_args() 

    args.context_len = args.input_flank_context*2
    args.num_sampled_sub_per_acc_don = min(args.num_sampled_sub_per_acc_don, args.sampled_sub_nt_distance*2)

    if args.eval_data_dir is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_dir "
            "or remove the --do_eval argument."
        )
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    if int(args.context_len) == 80:
        kernel_sizes = np.asarray([11, 11, 11, 11])
        dilation_rates = np.asarray([1, 1, 1, 1])
        # BATCH_SIZE = 18*N_GPUS
    elif int(args.context_len) == 400:
        kernel_sizes = np.asarray([11, 11, 11, 11, 11, 11, 11, 11])
        dilation_rates = np.asarray([1, 1, 1, 1, 4, 4, 4, 4])
        # BATCH_SIZE = 18*N_GPUS
    elif int(args.context_len) == 2000:
        kernel_sizes = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21])
        dilation_rates = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                        10, 10, 10, 10])
        # BATCH_SIZE = 12*N_GPUS
    elif int(args.context_len) == 10000:
        kernel_sizes = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21, 41, 41, 41, 41])
        dilation_rates = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                        10, 10, 10, 10, 25, 25, 25, 25])
        # BATCH_SIZE = 6*N_GPUS
        
    spliceai_model = SpliceAINew(args, args.spliceai_channels, kernel_sizes, dilation_rates)
    spliceai_model.to(args.device)

    site_aux_model = SiteAuxNet(args, args.spliceai_channels, act_rep_dim=args.act_rep_dim, 
                        act_rep_input_dropout_prob=args.act_rep_input_dropout_prob, act_rep_activation_dropout_prob=args.act_rep_activation_dropout_prob)
    site_aux_model.to(args.device)

    if args.spliceosome_model_type == 'spliceosome_full':
        if args.do_transcript_reg:
            spliceosome_model = SpliceosomeModelWithTranscriptProbLoss(args, args.site_act_rep_channels*2, args.spliceosome_model_channels, args.spliceosome_num_layer) # in_channels = args.site_act_rep_channels*2 due to [don, acc]
        else:
            spliceosome_model = SpliceosomeModel(args, args.site_act_rep_channels*2, args.spliceosome_model_channels, args.spliceosome_num_layer) # in_channels = args.site_act_rep_channels*2 due to [don, acc]
    elif args.spliceosome_model_type == 'spliceosome_junction_reg':
        spliceosome_model = SpliceosomeModelJunctionBaseline(args, args.site_act_rep_channels*2, args.spliceosome_model_channels, args.spliceosome_num_layer) # in_channels = args.site_act_rep_channels*2 due to [don, acc]
    elif args.spliceosome_model_type == 'more_layers_baseline':
        spliceosome_model = SiteAuxMoreLayersExtension(args, args.ablation_num_more_layers)
    spliceosome_model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Load saved model weights (if defined)
    if args.load_pretrained_spliceai_path is not None:
        logger.info("Loading pretrained spliceai_model weights from: %s", args.load_pretrained_spliceai_path)
        spliceai_state_dict = torch.load(args.load_pretrained_spliceai_path)
        spliceai_model.load_state_dict(spliceai_state_dict)

    if args.load_pretrained_site_aux_model_path is not None:
        logger.info("Loading pretrained site_aux_model weights from: %s", args.load_pretrained_site_aux_model_path)
        site_aux_model_state_dict = torch.load(args.load_pretrained_site_aux_model_path)
        site_aux_model.load_state_dict(site_aux_model_state_dict)

    if args.load_pretrained_spliceosome_model_path is not None:
        logger.info("Loading pretrained spliceosome_model weights from: %s", args.load_pretrained_spliceosome_model_path)
        spliceosome_state_dict = torch.load(args.load_pretrained_spliceosome_model_path)
        spliceosome_model.load_state_dict(spliceosome_state_dict)

    # Training
    if args.do_train:
        if args.valid_data_dir is None:
            args.valid_data_dir = args.train_data_dir
        if args.act_rep_valid_file is None:
            args.act_rep_valid_file = args.act_rep_train_file

        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        if args.local_rank == 0:
            torch.distributed.barrier()

        # Split train and validation datasets' gene here

        if args.num_train_spliceai_epochs > 0 or args.spliceai_max_steps > 0:
            logger.info(" Preparing train_spliceai_dataset ")
            train_spliceai_dataset = MultipleJsonlDatasetForRegressionTruncatedGene(args, args.train_data_dir, act_rep_file=args.act_rep_train_file)
            if args.logging_valid_steps > 0:
                logger.info(" Preparing valid_spliceai_dataset ")
                valid_spliceai_dataset = MultipleJsonlDatasetForRegressionTruncatedGene(args, args.valid_data_dir, act_rep_file=args.act_rep_valid_file)
            else:
                valid_spliceai_dataset = None
            logger.info(" Training spliceai model ")
            if args.spliceai_model_training_type == 'spliceai_cls_reg':
                global_step, tr_loss = train_spliceai(args, train_spliceai_dataset, spliceai_model, site_aux_model, valid_dataset=valid_spliceai_dataset, do_train_site_aux_model=True, eval_at_init=args.eval_before_training)
            elif args.spliceai_model_training_type == 'spliceai_cls':
                global_step, tr_loss = train_spliceai(args, train_spliceai_dataset, spliceai_model, site_aux_model, valid_dataset=valid_spliceai_dataset, do_train_site_aux_model=False, eval_at_init=args.eval_before_training)
            else:
                raise ValueError("--spliceai_model_training_type must be either spliceai_cls_reg or spliceai_cls")

        if args.cache_spliceai_final_hidden:
            logger.info(" Caching spliceai final hidden states ")
            if args.spliceai_cache_dataset_type == 'point':
                cache_spliceai_dataset = MultipleJsonlDatasetForRegression(args, args.train_data_dir, augment_transcript_data=False, num_sampled_sub_per_acc_don=0, return_sample_metadata=True)
            elif args.spliceai_cache_dataset_type == 'sequence':
                cache_spliceai_dataset = MultipleJsonlDatasetForRegressionTruncatedGene(args, args.train_data_dir, max_seq_len=math.inf, return_sample_metadata=True)
            if args.spliceai_model_training_type == 'spliceai_cls_reg':
                cache_file_path = infer_and_save_spliceai_final_hidden(args, cache_spliceai_dataset, spliceai_model, site_aux_model, cache_dir=args.splice_site_cache_dir)
            elif args.spliceai_model_training_type == 'spliceai_cls':
                cache_file_path = infer_and_save_spliceai_final_hidden(args, cache_spliceai_dataset, spliceai_model, site_aux_model, cache_dir=args.splice_site_cache_dir, cache_both_spliceai_n_siteaux_reps=False)            
            else:
                raise ValueError("--spliceai_model_training_type must be either spliceai_cls_reg or spliceai_cls")
            logger.info(" Saved spliceai representations in h5 dataset file: %s", cache_file_path)

        if (args.num_train_only_spliceosome_model_epochs > 0 or args.spliceosome_max_steps > 0) or (args.num_train_full_spliceai_spliceosome_model_epochs > 0 or args.full_spliceai_spliceosome_max_steps > 0):
            logger.info(" Preparing train_spliceosome_model_dataset")
            if args.do_transcript_reg:
                train_spliceosome_model_dataset = MultipleJsonlDatasetForTranscriptRegression(args, args.train_data_dir, augment_transcript_data=args.augment_transcript_data, return_sample_metadata=True, contain_transcript_probs=True)
            else:
                train_spliceosome_model_dataset = MultipleJsonlDatasetForRegression(args, args.train_data_dir, augment_transcript_data=args.augment_transcript_data, return_sample_metadata=True)
            if args.logging_valid_steps > 0:
                logger.info(" Preparing valid_spliceosome_model_dataset")
                if args.do_transcript_reg:
                    valid_spliceosome_model_dataset = MultipleJsonlDatasetForTranscriptRegression(args, args.valid_data_dir, act_rep_file=args.act_rep_valid_file, augment_transcript_data=False, return_sample_metadata=True, num_sampled_sub_per_acc_don=0, contain_transcript_probs=True)
                else:
                    valid_spliceosome_model_dataset = MultipleJsonlDatasetForRegression(args, args.valid_data_dir, act_rep_file=args.act_rep_valid_file, augment_transcript_data=False, return_sample_metadata=True, num_sampled_sub_per_acc_don=0)
            else:
                valid_spliceosome_model_dataset = None
        
        if args.num_train_only_spliceosome_model_epochs > 0 or args.spliceosome_max_steps > 0:
            logger.info(" Training spliceosome model (frozen spliceai weights) ")
            if args.do_transcript_reg:
                if args.spliceosome_model_training_type == 'spliceosome_only': 
                    global_step, tr_loss = train_spliceosome_model_wtranscriptprobs(args, train_spliceosome_model_dataset, spliceosome_model, spliceai_model, site_aux_model,
                                                train_epochs=args.num_train_only_spliceosome_model_epochs, train_steps=args.spliceosome_max_steps, valid_dataset=valid_spliceosome_model_dataset, 
                                                do_train_spliceai=False, do_train_site_aux_model=False, augment_transcript_data=args.augment_transcript_data, eval_at_init=args.eval_before_training, do_transcript_reg=True)
                else:
                    global_step, tr_loss = train_spliceosome_model_wtranscriptprobs(args, train_spliceosome_model_dataset, spliceosome_model, spliceai_model, site_aux_model,
                                                train_epochs=args.num_train_only_spliceosome_model_epochs, train_steps=args.spliceosome_max_steps, valid_dataset=valid_spliceosome_model_dataset, 
                                                do_train_spliceai=False, do_train_site_aux_model=True, augment_transcript_data=args.augment_transcript_data, eval_at_init=args.eval_before_training, do_transcript_reg=True)
            else:
                if args.spliceosome_model_training_type == 'spliceosome_only': 
                    global_step, tr_loss = train_spliceosome_model(args, train_spliceosome_model_dataset, spliceosome_model, spliceai_model, site_aux_model,
                                                train_epochs=args.num_train_only_spliceosome_model_epochs, train_steps=args.spliceosome_max_steps, valid_dataset=valid_spliceosome_model_dataset, 
                                                do_train_spliceai=False, do_train_site_aux_model=False, augment_transcript_data=args.augment_transcript_data, eval_at_init=args.eval_before_training)
                else:
                    global_step, tr_loss = train_spliceosome_model(args, train_spliceosome_model_dataset, spliceosome_model, spliceai_model, site_aux_model,
                                                train_epochs=args.num_train_only_spliceosome_model_epochs, train_steps=args.spliceosome_max_steps, valid_dataset=valid_spliceosome_model_dataset, 
                                                do_train_spliceai=False, do_train_site_aux_model=True, augment_transcript_data=args.augment_transcript_data, eval_at_init=args.eval_before_training)

        if args.num_train_full_spliceai_spliceosome_model_epochs > 0 or args.full_spliceai_spliceosome_max_steps > 0:
            logger.info(" Training full spliceosome model and spliceai model ")
            if args.do_transcript_reg:
                global_step, tr_loss = train_spliceosome_model_wtranscriptprobs(args, train_spliceosome_model_dataset, spliceosome_model, spliceai_model, site_aux_model,
                                            train_epochs=args.num_train_full_spliceai_spliceosome_model_epochs, train_steps=args.full_spliceai_spliceosome_max_steps, valid_dataset=valid_spliceosome_model_dataset, 
                                            do_train_spliceai=True, do_train_site_aux_model=True, do_train_spliceai_cls=args.do_spliceai_cls_in_full_spliceosome_train, augment_transcript_data=args.augment_transcript_data, eval_at_init=args.eval_before_training,
                                            do_transcript_reg=True)
            else:
                global_step, tr_loss = train_spliceosome_model(args, train_spliceosome_model_dataset, spliceosome_model, spliceai_model, site_aux_model,
                                            train_epochs=args.num_train_full_spliceai_spliceosome_model_epochs, train_steps=args.full_spliceai_spliceosome_max_steps, valid_dataset=valid_spliceosome_model_dataset, 
                                            do_train_spliceai=True, do_train_site_aux_model=True, do_train_spliceai_cls=args.do_spliceai_cls_in_full_spliceosome_train, augment_transcript_data=args.augment_transcript_data, eval_at_init=args.eval_before_training)


        if args.spliceosome_model_type == 'more_layers_baseline' and (args.num_train_ablation_more_layers_epochs > 0 or args.ablation_more_layers_max_steps > 0):
            logger.info(" Preparing train_spliceai_dataset ")
            train_spliceai_dataset = MultipleJsonlDatasetForRegressionTruncatedGene(args, args.train_data_dir, act_rep_file=args.act_rep_train_file)
            if args.logging_valid_steps > 0:
                logger.info(" Preparing valid_spliceai_dataset ")
                valid_spliceai_dataset = MultipleJsonlDatasetForRegressionTruncatedGene(args, args.valid_data_dir, act_rep_file=args.act_rep_valid_file)
            else:
                valid_spliceai_dataset = None
            logger.info(" Training more_layers_baseline ")
            global_step, tr_loss = train_spliceai(args, train_spliceai_dataset, spliceai_model, site_aux_model, ablation_finetune_model=spliceosome_model, valid_dataset=valid_spliceai_dataset, 
                                    do_train_spliceai=False, do_train_site_aux_model=False, eval_at_init=args.eval_before_training, train_epochs=args.num_train_ablation_more_layers_epochs, train_steps=args.ablation_more_layers_max_steps)


    # Save trained models
    if (args.do_train or args.save_model_weights_even_wo_train) and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir, exist_ok=True)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        if args.num_train_spliceai_epochs > 0 or args.num_train_full_spliceai_spliceosome_model_epochs > 0 or args.save_spliceai_model:

            model_to_save = (
                spliceai_model.module if hasattr(spliceai_model, "module") else spliceai_model
            )  # Take care of distributed/parallel training
            
            # Save model weights
            output_model_file = os.path.join(args.output_dir, "spliceai_pytorch_model.bin")

            torch.save(model_to_save.state_dict(), output_model_file)
            logger.info("Spliceai model weights saved in {}".format(output_model_file))

            if args.spliceai_model_training_type == 'spliceai_cls_reg' or args.num_train_only_spliceosome_model_epochs > 0 or args.spliceosome_max_steps > 0 or args.num_train_full_spliceai_spliceosome_model_epochs > 0 or args.full_spliceai_spliceosome_max_steps > 0:
                # Save site_aux_model
                model_to_save = (
                    site_aux_model.module if hasattr(site_aux_model, "module") else site_aux_model
                )  # Take care of distributed/parallel training                    
                output_model_file = os.path.join(args.output_dir, "site_aux_pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
                logger.info("Site aux model weights saved in {}".format(output_model_file))

            
            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        if args.num_train_only_spliceosome_model_epochs > 0 or args.num_train_full_spliceai_spliceosome_model_epochs > 0 or args.save_spliceosome_model:
            # Save spliceosome model
            model_to_save = (
                spliceosome_model.module if hasattr(spliceosome_model, "module") else spliceosome_model
            )  # Take care of distributed/parallel training

            # Save model weights
            output_model_file = os.path.join(args.output_dir, "spliceosomenet_pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)
            logger.info("Spliceosome model weights saved in {}".format(output_model_file))
 
            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        if args.spliceosome_model_type == 'more_layers_baseline' and (args.num_train_ablation_more_layers_epochs > 0 or args.ablation_more_layers_max_steps > 0):
            # Save ablation_finetune_model
            model_to_save = (
                spliceosome_model.module if hasattr(spliceosome_model, "module") else spliceosome_model
            )  # Take care of distributed/parallel training                    
            output_model_file = os.path.join(args.output_dir, "ablation_finetune_model_pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)
            logger.info("ablation_finetune_model weights saved in {}".format(output_model_file))

    # Evaluation: evaluate model on loss values
    all_results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if args.do_train==False and args.load_pretrained_spliceai_path is not None and args.load_pretrained_spliceosome_model_path is not None:
            raise ValueError(
                "--load_pretrained_spliceai_path and --load_pretrained_spliceosome_model_path are None while do_train==False, no trained models is available for evaluation"
            )

        checkpoints = [args.output_dir]

        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
        
            # Load saved model weights 
            spliceai_model_file = os.path.join(checkpoint, "spliceai_pytorch_model.bin")
            site_aux_model_file = os.path.join(checkpoint, "site_aux_pytorch_model.bin")
            spliceosome_model_file = os.path.join(checkpoint, "spliceosomenet_pytorch_model.bin")
            ablation_finetune_model_file = os.path.join(checkpoint, "ablation_finetune_model_pytorch_model.bin")

            if os.path.exists(spliceai_model_file):
                spliceai_state_dict = torch.load(spliceai_model_file)
                spliceai_model.load_state_dict(spliceai_state_dict)
                logger.info("Loaded spliceai_model weights from: %s", spliceai_model_file)
                spliceai_model.to(args.device)
                spliceai_model_file_exist = True
            else:
                spliceai_model_file_exist = False
                raise ValueError(
                    "Base spliceai model file does not exist for evaluation."
                )

            if os.path.exists(site_aux_model_file):
                site_aux_state_dict = torch.load(site_aux_model_file)
                site_aux_model.load_state_dict(site_aux_state_dict)
                logger.info("Loaded site_aux_model weights from: %s", site_aux_model_file)
                site_aux_model.to(args.device)
                site_aux_model_file_exist = True
            else:
                site_aux_model_file_exist = False

            if os.path.exists(spliceosome_model_file):
                spliceosome_state_dict = torch.load(spliceosome_model_file)
                spliceosome_model.load_state_dict(spliceosome_state_dict)
                logger.info("Loaded spliceosome_model weights from: %s", spliceosome_model_file)
                spliceosome_model.to(args.device)
                spliceosome_model_file_exist = True
            else:
                spliceosome_model_file_exist = False

            if os.path.exists(ablation_finetune_model_file) and args.spliceosome_model_type == 'more_layers_baseline':
                spliceosome_state_dict = torch.load(ablation_finetune_model_file)
                spliceosome_model.load_state_dict(spliceosome_state_dict)
                logger.info("Loaded spliceosome_model weights from: %s", ablation_finetune_model_file)
                spliceosome_model.to(args.device)
                ablation_finetune_model_file_exist = True
            else:
                ablation_finetune_model_file_exist = False

            # Evaluate spliceosome model
            if spliceosome_model_file_exist:
                eval_spliceosome_model_dataset = MultipleJsonlDatasetForRegression(args, args.eval_data_dir, act_rep_file=args.act_rep_eval_file, augment_transcript_data=False, return_sample_metadata=True, 
                                                    num_sampled_sub_per_acc_don=0, patient_list=args.eval_patient_list, chr_list=args.eval_chr_list)
                spliceosome_results = evaluate_spliceosome_model(args, spliceai_model, site_aux_model, spliceosome_model, eval_dataset=eval_spliceosome_model_dataset, eval_output_dir=checkpoint, prefix=args.eval_output_dir_prefix)
                spliceosome_results = dict((k + "_spliceosome{}".format(global_step), v) for k, v in spliceosome_results.items())
                all_results.update(spliceosome_results)
            else:
            # Evaluate spliceAI model
                if site_aux_model_file_exist and ablation_finetune_model_file_exist:
                    eval_spliceai_dataset = MultipleJsonlDatasetForRegression(args, args.eval_data_dir, act_rep_file=args.act_rep_eval_file, augment_transcript_data=False, return_sample_metadata=True, 
                                                        num_sampled_sub_per_acc_don=0, patient_list=args.eval_patient_list, chr_list=args.eval_chr_list)
                    spliceai_results = evaluate_spliceai_fast(args, spliceai_model, site_aux_model, ablation_finetune_model=spliceosome_model, eval_dataset=eval_spliceai_dataset, eval_output_dir=checkpoint, prefix=args.eval_output_dir_prefix)
                elif site_aux_model_file_exist:
                    eval_spliceai_dataset = MultipleJsonlDatasetForRegression(args, args.eval_data_dir, act_rep_file=args.act_rep_eval_file, augment_transcript_data=False, return_sample_metadata=True, 
                                                        num_sampled_sub_per_acc_don=0, patient_list=args.eval_patient_list, chr_list=args.eval_chr_list)
                    spliceai_results = evaluate_spliceai_fast(args, spliceai_model, site_aux_model, eval_dataset=eval_spliceai_dataset, eval_output_dir=checkpoint, prefix=args.eval_output_dir_prefix)
                else:
                    eval_spliceai_dataset = MultipleJsonlDatasetForRegression(args, args.eval_data_dir, act_rep_file=args.act_rep_eval_file, augment_transcript_data=False, return_sample_metadata=True, 
                                                        num_sampled_sub_per_acc_don=0, patient_list=args.eval_patient_list, chr_list=args.eval_chr_list)
                    spliceai_results = evaluate_spliceai_fast(args, spliceai_model, site_aux_model, eval_cor_with_spliceaicls=True, eval_dataset=eval_spliceai_dataset, eval_output_dir=checkpoint, prefix=args.eval_output_dir_prefix)


                spliceai_results = dict((k + "_spliceai{}".format(global_step), v) for k, v in spliceai_results.items())
                all_results.update(spliceai_results)

    # Inference: infer splice site and transcript probability prediction with model
    all_results = {}
    if args.do_infer and args.local_rank in [-1, 0]:
        logger.info("Doing inference..")
        if args.do_train==False and args.load_pretrained_spliceai_path is not None and args.load_pretrained_spliceosome_model_path is not None:
            raise ValueError(
                "--load_pretrained_spliceai_path and --load_pretrained_spliceosome_model_path are None while do_train==False, no trained models is available for evaluation"
            )

        checkpoints = [args.output_dir]

        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
        
            # Load saved model weights 
            spliceai_model_file = os.path.join(checkpoint, "spliceai_pytorch_model.bin")
            site_aux_model_file = os.path.join(checkpoint, "site_aux_pytorch_model.bin")
            spliceosome_model_file = os.path.join(checkpoint, "spliceosomenet_pytorch_model.bin")
            ablation_finetune_model_file = os.path.join(checkpoint, "ablation_finetune_model_pytorch_model.bin")

            if os.path.exists(spliceai_model_file):
                spliceai_state_dict = torch.load(spliceai_model_file)
                spliceai_model.load_state_dict(spliceai_state_dict)
                logger.info("Loaded spliceai_model weights from: %s", spliceai_model_file)
                spliceai_model.to(args.device)
                spliceai_model_file_exist = True
            else:
                spliceai_model_file_exist = False
                raise ValueError(
                    "Base spliceai model file does not exist for evaluation."
                )

            if os.path.exists(site_aux_model_file):
                site_aux_state_dict = torch.load(site_aux_model_file)
                site_aux_model.load_state_dict(site_aux_state_dict)
                logger.info("Loaded site_aux_model weights from: %s", site_aux_model_file)
                site_aux_model.to(args.device)
                site_aux_model_file_exist = True
            else:
                site_aux_model_file_exist = False

            if os.path.exists(spliceosome_model_file):
                spliceosome_state_dict = torch.load(spliceosome_model_file)
                spliceosome_model.load_state_dict(spliceosome_state_dict)
                logger.info("Loaded spliceosome_model weights from: %s", spliceosome_model_file)
                spliceosome_model.to(args.device)
                spliceosome_model_file_exist = True
            else:
                spliceosome_model_file_exist = False

            if os.path.exists(ablation_finetune_model_file) and args.spliceosome_model_type == 'more_layers_baseline':
                spliceosome_state_dict = torch.load(ablation_finetune_model_file)
                spliceosome_model.load_state_dict(spliceosome_state_dict)
                logger.info("Loaded spliceosome_model weights from: %s", ablation_finetune_model_file)
                spliceosome_model.to(args.device)
                ablation_finetune_model_file_exist = True
            else:
                ablation_finetune_model_file_exist = False

            # Evaluate spliceosome model
            if spliceosome_model_file_exist:
                eval_spliceosome_model_dataset = MultipleJsonlDatasetForRegression(args, args.eval_data_dir, act_rep_file=args.act_rep_eval_file, augment_transcript_data=False, return_sample_metadata=True, 
                                                    num_sampled_sub_per_acc_don=0, patient_list=args.eval_patient_list, chr_list=args.eval_chr_list)
                
                _ = predict_ss_transcript_prob_spliceosome_model(args, spliceai_model, site_aux_model, spliceosome_model, pred_dataset=eval_spliceosome_model_dataset, pred_output_dir=checkpoint, prefix="", gene_list=args.eval_gene_list)


    return all_results


if __name__ == "__main__":
    main()
