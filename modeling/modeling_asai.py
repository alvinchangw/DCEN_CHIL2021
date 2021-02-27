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


import logging
import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

import itertools


logger = logging.getLogger(__name__)


class ResidualUnit(nn.Module):
    def __init__(self, config, out_channels, kernel_size, dilation_rate):
        super().__init__()
        # l, out_channels 
        # w, kernel_size
        # ar, dilation_rate
        self.config = config

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        pad_size = math.ceil(dilation_rate * (kernel_size - 1) /2)
        self.conv1 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=pad_size, dilation=dilation_rate)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=pad_size, dilation=dilation_rate)

        self.init_weights()

    def forward(self, x):
        bn1 = self.bn1(x)
        act1 = self.act(bn1)
        conv1 = self.conv1(act1)
        bn2 = self.bn2(conv1)
        act2 = self.act(bn2)
        conv2 = self.conv2(act2)

        output = conv2 + x

        return output


    def init_weights(self):
        """ Initialize weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm) and module.bias is not None:
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class SpliceAINew(nn.Module):
    def __init__(self, config, out_channels, kernel_sizes, dilation_rates):
        # L, out_channels: Number of convolution kernels, scalar
        # W, kernel_sizes: Convolution window size in each residual unit, array of int
        # AR, dilation_rates: Atrous rate in each residual unit, array of int
        super().__init__()
        
        assert len(kernel_sizes) == len(dilation_rates)

        self.config = config

        self.reg_include_nonss_prob = config.reg_include_nonss_prob

        self.relu = nn.ReLU()
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.dilation_rates = dilation_rates

        self.CL = int(2 * np.sum(dilation_rates*(kernel_sizes-1)))
        self.conv = nn.Conv1d(config.seq_input_dim, out_channels, kernel_size=1)
        self.skip = nn.Conv1d(out_channels, out_channels, kernel_size=1)

        residual_units = []
        dense_layers = []
        for i in range(len(kernel_sizes)):
            residual_unit = ResidualUnit(config, out_channels, kernel_sizes[i], dilation_rates[i])
            residual_units.append(residual_unit)

            if (((i+1) % 4 == 0) or ((i+1) == len(kernel_sizes))):
                dense_layer = nn.Conv1d(out_channels, out_channels, kernel_size=1)
                dense_layers.append(dense_layer)
                
        self.residual_units = nn.ModuleList(residual_units)
        self.dense_layers = nn.ModuleList(dense_layers)

        # Classification layer
        self.class_output_layer = nn.Conv1d(out_channels, 3, kernel_size=1) # 3 classes: acceptor, donor and none
        self.softmax = nn.Softmax(dim=1)

        self.init_weights()

    def forward(self, x, labels_class=None, label_masks=None, return_site_hidden=True, none_class_reweight_factor=None, dynamic_reweight_none_class=False, keep_none_cls_prob=None):
        conv = self.conv(x)
        skip = self.skip(conv)
        
        dense_ind = 0
        for i in range(len(self.kernel_sizes)):
            residual_unit = self.residual_units[i]
            conv = residual_unit(conv)
            
            if (((i+1) % 4 == 0) or ((i+1) == len(self.kernel_sizes))):
                # Skip connections to the output after every 4 residual units
                dense_layer = self.dense_layers[dense_ind]
                dense = dense_layer(conv)
                skip = skip + dense
                dense_ind += 1

        # Remove flanking ends
        skip = skip[:, :, int(self.CL/2):int(-self.CL/2 )] # replaces Keras' Cropping1D(CL/2)(skip)
        site_hidden = skip

        # Compute classification logits and loss
        logits_class = self.class_output_layer(site_hidden)
        probs_class = self.softmax(logits_class)
        outputs = (logits_class,)

        if labels_class is not None:
            if label_masks is not None:
                loss_fct_class = CrossEntropyLoss(reduction='none')
                loss_class = loss_fct_class(logits_class, labels_class)
                num_valid_labels = torch.sum(label_masks)
                
                if dynamic_reweight_none_class:
                    reweight_matrix = torch.ones_like(label_masks, dtype=torch.float64)
                    num_acc_don = torch.sum(labels_class != 0)
                    reweight_matrix[labels_class == 0] = max(num_acc_don.item(), 1) / (2*(num_valid_labels.item()-num_acc_don.item()))
                    label_masks = label_masks * reweight_matrix
                elif none_class_reweight_factor:
                    reweight_matrix = torch.ones_like(label_masks, dtype=torch.float64)
                    reweight_matrix[labels_class == 0] = none_class_reweight_factor
                    label_masks = label_masks * reweight_matrix
                elif keep_none_cls_prob:
                    keep_none_cls_prob_mask = torch.full_like(label_masks, keep_none_cls_prob, dtype=torch.float64)
                    keep_mask = torch.bernoulli(keep_none_cls_prob_mask)
                    keep_mask[labels_class != 0] = 1
                    label_masks = label_masks * keep_mask

                total_loss_class = loss_class * label_masks
                reduced_loss_class = torch.sum(total_loss_class) / num_valid_labels
                outputs =  (reduced_loss_class,) + outputs
            else:
                loss_fct_class = CrossEntropyLoss()
                loss_class = loss_fct_class(logits_class, labels_class)
                outputs =  (loss_class,) + outputs

        if return_site_hidden:
            outputs = outputs + (site_hidden,)

        return outputs # (loss_class), logits_class, (site_hidden)

    def init_weights(self):
        """ Initialize weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm) and module.bias is not None:
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)



class SiteAuxNet(nn.Module):
    def __init__(self, config, spliceai_out_channels, act_rep_dim, reg_act='sigmoid', act_rep_input_dropout_prob=0, act_rep_activation_dropout_prob=0):
        super().__init__()
        # site + activator/repressor module
        
        self.config = config

        self.reg_include_nonss_prob = config.reg_include_nonss_prob
        self.act_rep_dim = act_rep_dim
        self.relu = nn.ReLU()
        self.spliceai_out_channels = spliceai_out_channels

        act_rep_layers = []
        for i in range(config.num_act_rep_layers):
            if i == 0 :
                act_rep_layer = nn.Linear(act_rep_dim, config.act_rep_channels)
            else:
                act_rep_layer = nn.Linear(config.act_rep_channels, config.act_rep_channels)
            act_rep_layers.append(act_rep_layer)

        self.act_rep_layers = nn.ModuleList(act_rep_layers)

        # acc/don act/rep combined modules
        site_act_rep_dim = spliceai_out_channels + config.act_rep_channels
        site_act_rep_layers = []
        for i in range(config.num_site_act_rep_layers):
            if i == 0 :
                site_act_rep_layer = nn.Conv1d(site_act_rep_dim, config.site_act_rep_channels, kernel_size=1)
            else:
                site_act_rep_layer = nn.Conv1d(config.site_act_rep_channels, config.site_act_rep_channels, kernel_size=1)
            site_act_rep_layers.append(site_act_rep_layer)

        self.site_act_rep_layers = nn.ModuleList(site_act_rep_layers)

        # Regression layer
        self.reg_output_layer = nn.Conv1d(config.site_act_rep_channels, 1, kernel_size=1)
        if reg_act == 'sigmoid':
            self.reg_act = nn.Sigmoid()
        else:
            self.reg_act = nn.ReLU()

        self.act_rep_input_dropout_prob = act_rep_input_dropout_prob
        if act_rep_input_dropout_prob != 0:
            self.act_rep_input_dropout = nn.Dropout(p=self.act_rep_input_dropout_prob)

        self.act_rep_activation_dropout_prob = act_rep_activation_dropout_prob
        if act_rep_activation_dropout_prob != 0:
            self.act_rep_activation_dropout = nn.Dropout(p=self.act_rep_activation_dropout_prob)

        self.init_weights()

    def forward(self, site_hidden, act_rep_input, labels_class=None, labels_reg=None, label_masks=None, reg_include_nonss_prob=None, 
                return_final_hidden=True, squeeze_site_act_rep_final_hidden=True):

        # Compute regression logits and loss
        # compute act_rep hidden states
        for i in range(len(self.act_rep_layers)):
            act_rep_layer = self.act_rep_layers[i]
            if i == 0:
                if self.act_rep_input_dropout_prob != 0:
                    act_rep_input = self.act_rep_input_dropout(act_rep_input)
                act_rep_hidden = act_rep_layer(act_rep_input)
            else:
                if self.act_rep_activation_dropout_prob != 0:
                    act_rep_hidden = self.act_rep_activation_dropout(act_rep_hidden)
                act_rep_hidden = act_rep_layer(act_rep_hidden)
            act_rep_hidden = self.relu(act_rep_hidden)
        
        # combine with site hidden states
        act_rep_hidden_expand = torch.unsqueeze(act_rep_hidden, dim=-1).expand(-1,-1,site_hidden.shape[-1]) # expand to match the length of acc/don hidden states
        if act_rep_hidden_expand.shape[0] == 1 and site_hidden.shape[0] != 1:
            # handle spliceosome training forward step
            act_rep_hidden_expand = act_rep_hidden_expand.expand(site_hidden.shape[0],-1,-1)
        spliceosome_hidden = torch.cat([site_hidden, act_rep_hidden_expand], dim=1)

        # compute site_act_rep hidden states
        for i in range(len(self.site_act_rep_layers)):
            site_act_rep_layer = self.site_act_rep_layers[i]
            spliceosome_hidden = site_act_rep_layer(spliceosome_hidden)
            spliceosome_hidden = self.relu(spliceosome_hidden)

        site_act_rep_final_hidden = spliceosome_hidden

        # compute regression logits
        logit_reg = self.reg_output_layer(site_act_rep_final_hidden)
        psi_reg = self.reg_act(logit_reg)
        psi_reg = torch.squeeze(psi_reg, dim=1)

        outputs = (psi_reg,)

        if labels_reg is not None:
            if reg_include_nonss_prob is None:
                reg_include_nonss_prob = self.reg_include_nonss_prob
            if reg_include_nonss_prob != 1:
                ss_prob_mask = torch.full_like(labels_reg, reg_include_nonss_prob)
                # set acc and don prob to 1
                ss_prob_mask[labels_class==1] = 1
                ss_prob_mask[labels_class==2] = 1

                # sample mask from prob mask
                ss_mask = torch.bernoulli(ss_prob_mask)
                
                if label_masks is not None:
                    label_masks = label_masks * ss_mask
                else:
                    label_masks = ss_mask
    
            if label_masks is not None:
                loss_fct_reg = MSELoss(reduction='none')
                loss_reg = loss_fct_reg(psi_reg, labels_reg)
                total_loss_reg = loss_reg * label_masks
                num_valid_labels = torch.sum(label_masks)
                reduced_loss_reg = torch.sum(total_loss_reg) / num_valid_labels
                outputs = (reduced_loss_reg,) + outputs
            else:
                loss_fct_reg = MSELoss()
                loss_reg = loss_fct_reg(psi_reg, labels_reg)
                outputs = (loss_reg,) + outputs

        if return_final_hidden:
            if squeeze_site_act_rep_final_hidden:
                site_act_rep_final_hidden = torch.squeeze(site_act_rep_final_hidden, dim=-1)
            outputs = outputs + (site_act_rep_final_hidden,)

        return outputs # (loss_reg), psi_reg, (site_act_rep_final_hidden)

    def init_weights(self):
        """ Initialize weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm) and module.bias is not None:
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class SiteAuxMoreLayersExtension(nn.Module):
    def __init__(self, config, num_more_layers=1, reg_act='sigmoid'):
        super().__init__()
        # site + activator/repressor module
        
        self.config = config
        self.reg_include_nonss_prob = config.reg_include_nonss_prob
        self.relu = nn.ReLU()

        site_act_rep_layers = []
        for i in range(num_more_layers):
            if i == 0 :
                site_act_rep_layer = nn.Conv1d(2*config.site_act_rep_channels, config.site_act_rep_channels, kernel_size=1)
            else:
                site_act_rep_layer = nn.Conv1d(config.site_act_rep_channels, config.site_act_rep_channels, kernel_size=1)
            site_act_rep_layers.append(site_act_rep_layer)

        self.site_act_rep_layers = nn.ModuleList(site_act_rep_layers)

        # Regression layer
        self.reg_output_layer = nn.Conv1d(config.site_act_rep_channels, 1, kernel_size=1)
        if reg_act == 'sigmoid':
            self.reg_act = nn.Sigmoid()
        else:
            self.reg_act = nn.ReLU()

        self.init_weights()

    def forward(self, site_act_rep_final_hidden, labels_class=None, labels_reg=None, label_masks=None, reg_include_nonss_prob=None, 
                return_final_hidden=False, squeeze_site_act_rep_final_hidden=True):

        # match the input dim of spliceosome_model
        input_site_act_rep_final_hidden = torch.cat([site_act_rep_final_hidden, site_act_rep_final_hidden], dim=1)
        hidden = input_site_act_rep_final_hidden

        # Compute site_act_rep hidden states
        for i in range(len(self.site_act_rep_layers)):
            site_act_rep_layer = self.site_act_rep_layers[i]
            hidden = site_act_rep_layer(hidden)
            hidden = self.relu(hidden)

        output_site_act_rep_final_hidden = hidden

        # Compute regression logits
        logit_reg = self.reg_output_layer(output_site_act_rep_final_hidden)
        psi_reg = self.reg_act(logit_reg)
        psi_reg = torch.squeeze(psi_reg, dim=1)

        outputs = (psi_reg,)

        if labels_reg is not None:
            if reg_include_nonss_prob is None:
                reg_include_nonss_prob = self.reg_include_nonss_prob
            if reg_include_nonss_prob != 1:
                ss_prob_mask = torch.full_like(labels_reg, reg_include_nonss_prob)
                # set acc and don prob to 1
                ss_prob_mask[labels_class==1] = 1
                ss_prob_mask[labels_class==2] = 1

                # sample mask from prob mask
                ss_mask = torch.bernoulli(ss_prob_mask)
                
                if label_masks is not None:
                    label_masks = label_masks * ss_mask
                else:
                    label_masks = ss_mask
    
            if label_masks is not None:
                loss_fct_reg = MSELoss(reduction='none')
                loss_reg = loss_fct_reg(psi_reg, labels_reg)
                total_loss_reg = loss_reg * label_masks
                num_valid_labels = torch.sum(label_masks)
                reduced_loss_reg = torch.sum(total_loss_reg) / num_valid_labels
                outputs = (reduced_loss_reg,) + outputs
            else:
                loss_fct_reg = MSELoss()
                loss_reg = loss_fct_reg(psi_reg, labels_reg)
                outputs = (loss_reg,) + outputs

        if return_final_hidden:
            if squeeze_site_act_rep_final_hidden:
                output_site_act_rep_final_hidden = torch.squeeze(output_site_act_rep_final_hidden, dim=-1)
            outputs = outputs + (output_site_act_rep_final_hidden,)

        return outputs # (loss_reg), psi_reg, (output_site_act_rep_final_hidden)

    def init_weights(self):
        """ Initialize weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm) and module.bias is not None:
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class SpliceosomeNet(nn.Module):
    def __init__(self, config, in_channels, layer_channels, num_layer):
        super().__init__()
        self.config = config

        self.in_channels = in_channels
        self.layer_channels = layer_channels
        self.num_layer = num_layer

        self.relu = nn.ReLU()

        dense_layers = []
        if num_layer > 1:
            for i in range(num_layer-1):
                if i == 0:
                    dense_layer = nn.Linear(in_channels, layer_channels)
                else:
                    dense_layer = nn.Linear(layer_channels, layer_channels)
                dense_layers.append(dense_layer)

            dense_layer = nn.Linear(layer_channels, 1) 
            dense_layers.append(dense_layer)
        else:
            dense_layer = nn.Linear(in_channels, 1) 
            dense_layers.append(dense_layer)
        self.dense_layers = nn.ModuleList(dense_layers)

        self.init_weights()

    def forward(self, spliceosome_input):
        for i in range(len(self.dense_layers)):
            if i == 0:
                spliceosome_hidden = spliceosome_input
            dense_layer = self.dense_layers[i]
            spliceosome_hidden = dense_layer(spliceosome_hidden)
            # omit relu act for last layer
            if i != len(self.dense_layers) -1:
                spliceosome_hidden = self.relu(spliceosome_hidden)

        spliceosome_potential = spliceosome_hidden
        
        return spliceosome_potential 

    def init_weights(self):
        """ Initialize weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm) and module.bias is not None:
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


def stringify_junction(junction_tuple):
    return '^'.join(list(map(str, junction_tuple)))

def stringify_transcript(transcript_list):
    return list(map( stringify_junction, transcript_list))

def stringify_gene(gene_list):
    return list(map(stringify_transcript, gene_list))

# convert 
# [ [ [(-1, 0), (857, 1904), (2031, 9999999)], [(-1, 0), (857, 2433), (2578, 9999999)], .. ], ..] 
# to
# [ [ ['-1^0', '857^1904', '2031^9999999'], ['-1^0', '857^2433', '2578^9999999'], .. ], ..]
def stringify_gene_batch(gene_batch):
    return list(map(stringify_gene, gene_batch))

def unique_elements(lists):
    ele_list = list(set(itertools.chain.from_iterable(lists)))
    ele_list.sort()
    return ele_list

def batch_unique_elements(batch_lists):
    return list(map(unique_elements, batch_lists))

class SpliceosomeModel(nn.Module):
    def __init__(self, config, in_channels, layer_channels, num_layer, use_ref_potential=True):
        super().__init__()
        self.config = config
        
        self.gene_start_rep = nn.Parameter(torch.zeros(config.site_act_rep_channels))
        self.use_ref_potential = use_ref_potential
        self.ref_potential = nn.Parameter(torch.zeros(1))
        self.gene_start_token_position = config.gene_start_token_position
        self.gene_end_rep =  nn.Parameter(torch.zeros(config.site_act_rep_channels))
        self.gene_end_token_position = config.gene_end_token_position

        self.spliceosome_net = SpliceosomeNet(config, in_channels, layer_channels, num_layer)

        self.loss_fct = MSELoss()

        self.init_weights()

    def forward(self, splice_site_reps, transcripts_splice_junctions, fake_transcripts_splice_junctions=None, alabels=None, dlabels=None, return_transcript_probs=False, output_np_prob=False):
        # splice_site_reps: [ {pos1: Tensor, pos2: Tensor }, {..}, .. ], array of dict where each dict stores splice sites' embedding
        # transcripts_splice_junctions: example [ [ [(-1, 0), (857, 1904), (2031, 9999999)], [(-1, 0), (857, 2433), (2578, 9999999)], .. ], ..] (batch) array of (gene) array of (transcript) array of (junction) tuple where each tuple stores donor and acceptors' positions
        # fake_transcripts_splice_junctions: example [ [ [(-1, 2), (857, 1904), (2031, 9999999)], [(-1, 0), (857, 1905), (2031, 9999999)], .. ], ..] (batch) array of (gene) array of (transcript) array of (junction) tuple where each tuple stores donor and acceptors' positions

        if alabels is not None and dlabels is not None:
            loss_reg = 0

        # compute potentials, transcript potentials and probabilities of each splice site
        transcripts_splice_junctions_str = stringify_gene_batch(transcripts_splice_junctions)
        # [ [ ['-1^0', '857^1904', '2031^9999999'], ['-1^0', '857^2433', '2578^9999999'], .. ], ..]
        if fake_transcripts_splice_junctions is not None:
            fake_transcripts_splice_junctions_str = stringify_gene_batch(fake_transcripts_splice_junctions)

            # combine splice junctions from real and fake transcripts
            all_transcripts_splice_junctions_str = [real + fake for real, fake in zip(transcripts_splice_junctions_str, fake_transcripts_splice_junctions_str)]
        else:
            all_transcripts_splice_junctions_str = transcripts_splice_junctions_str
        
        # find unique splice junctions in each gene
        unique_junctions = batch_unique_elements(all_transcripts_splice_junctions_str)

        # construct junction embs for genes
        junction_embs = []
        num_reg_labels = 0
        for gene_unique_junctions, gene_splice_site_reps in zip(unique_junctions, splice_site_reps):
            gene_junction_embs = []
            for gene_unique_junction in gene_unique_junctions:
                don, acc = gene_unique_junction.split('^')
                don = int(don)
                acc = int(acc)

                if don == self.gene_start_token_position:
                    don_emb = self.gene_start_rep
                else:
                    don_emb = gene_splice_site_reps[don]

                if acc == self.gene_end_token_position:
                    acc_emb = self.gene_end_rep
                else:
                    acc_emb = gene_splice_site_reps[acc]

                junction_emb = torch.cat([don_emb, acc_emb], dim=-1)
                gene_junction_embs.append(junction_emb)

            gene_junction_embs = torch.stack(gene_junction_embs, dim=0)
            junction_embs.append(gene_junction_embs)


        # compute junctions' potentials
        junction_potentials = []
        for gene_junction_embs in junction_embs:
            gene_junction_embs = gene_junction_embs
            gene_junction_potentials = self.spliceosome_net(gene_junction_embs)
            junction_potentials.append(gene_junction_potentials)

        # construct junction potential dict
        junction_potentials_dict = []
        for gene_unique_junctions, gene_junction_potentials in zip(unique_junctions, junction_potentials):
            gene_junction_potentials_dict = {}
            for junction_ind, gene_unique_junction in enumerate(gene_unique_junctions):
                gene_junction_potential = gene_junction_potentials[junction_ind]
                gene_junction_potentials_dict[gene_unique_junction] = gene_junction_potential

            junction_potentials_dict.append(gene_junction_potentials_dict)

        # compute transcripts' potentials
        splice_sites_prob = []
        transcripts_prob = []
        for gene_ind, (gene_all_transcripts_splice_junctions_str, gene_junction_potentials_dict) in enumerate(zip(all_transcripts_splice_junctions_str, junction_potentials_dict)):
            gene_transcripts_potential = []

            for gene_transcript_splice_junctions_str in gene_all_transcripts_splice_junctions_str:
                transcript_junction_potentials = []

                for splice_junction_str in gene_transcript_splice_junctions_str:
                    splice_junction_potential = gene_junction_potentials_dict[splice_junction_str]
                    transcript_junction_potentials.append(splice_junction_potential)
                transcript_total_potential = torch.sum(torch.stack(transcript_junction_potentials), dim=0)
                gene_transcripts_potential.append(transcript_total_potential)
            
            if self.use_ref_potential:
                gene_transcripts_potential.append(self.ref_potential)

            gene_transcripts_potential = torch.cat(gene_transcripts_potential, dim=0)

            # compute transcripts' prob
            gene_transcripts_prob = nn.functional.softmax(gene_transcripts_potential, dim=-1)

            # compute splice site prob
            gene_splice_site_prob_dict = {}
            gene_transcript_prob_dict = {}
            for transcript_ind, gene_transcript_splice_junctions_str in enumerate(gene_all_transcripts_splice_junctions_str):
                transcript_str = None
                for splice_junction_str in gene_transcript_splice_junctions_str:
                    if transcript_str is None:
                        transcript_str = splice_junction_str
                    else:
                        transcript_str = transcript_str + "_" + splice_junction_str
                    don, acc = splice_junction_str.split('^')
                    don = int(don)
                    acc = int(acc)

                    if don != self.gene_start_token_position:
                        if don in gene_splice_site_prob_dict:
                            gene_splice_site_prob_dict[don] = gene_splice_site_prob_dict[don] + gene_transcripts_prob[transcript_ind]
                        else:
                            gene_splice_site_prob_dict[don] = gene_transcripts_prob[transcript_ind]
                        
                    if acc != self.gene_end_token_position:
                        if acc in gene_splice_site_prob_dict:
                            gene_splice_site_prob_dict[acc] = gene_splice_site_prob_dict[acc] + gene_transcripts_prob[transcript_ind]
                        else:
                            gene_splice_site_prob_dict[acc] = gene_transcripts_prob[transcript_ind]

                # save transcript prob in gene_transcript_prob_dict
                gene_transcript_prob_dict[transcript_str] = gene_transcripts_prob[transcript_ind]

            splice_sites_prob.append(gene_splice_site_prob_dict)
            transcripts_prob.append(gene_transcript_prob_dict)

            # compute reg loss
            if alabels is not None and dlabels is not None:
                gene_alabels = alabels[gene_ind]
                gene_dlabels = dlabels[gene_ind]

                loss_reg_don = 0
                loss_reg_acc = 0
                for don_pos in gene_dlabels.keys():
                    label = torch.tensor(gene_dlabels[don_pos]['psi'], dtype=torch.float).to(self.config.device)
                    pred = gene_splice_site_prob_dict[don_pos]
                    loss_reg_don = loss_reg_don + self.loss_fct(pred, label)
                for acc_pos in gene_alabels.keys():
                    label = torch.tensor(gene_alabels[acc_pos]['psi'], dtype=torch.float).to(self.config.device)
                    pred = gene_splice_site_prob_dict[acc_pos]
                    loss_reg_acc = loss_reg_acc + self.loss_fct(pred, label)

                num_reg_labels = num_reg_labels + len(gene_dlabels.keys()) + len(gene_alabels.keys())

                gene_loss_reg = loss_reg_don + loss_reg_acc
                loss_reg = loss_reg + gene_loss_reg
        
        # Divide the total loss value by num_reg_labels to output mean reg loss value
        mean_reduced_loss_reg = loss_reg / num_reg_labels

        if output_np_prob:
            for gene_ind, gene_splice_site_prob_dict in enumerate(splice_sites_prob):
                for site in splice_sites_prob[gene_ind]:
                    splice_sites_prob[gene_ind][site] = splice_sites_prob[gene_ind][site].cpu().numpy()
                for transcript in transcripts_prob[gene_ind]:
                    transcripts_prob[gene_ind][transcript] = transcripts_prob[gene_ind][transcript].cpu().numpy()

        if alabels is not None and dlabels is not None:
            if return_transcript_probs:
                return mean_reduced_loss_reg, splice_sites_prob, transcripts_prob
            else:
                return mean_reduced_loss_reg, splice_sites_prob
        else:
            if return_transcript_probs:
                return splice_sites_prob, transcripts_prob
            else:
                return splice_sites_prob 

    def init_weights(self):
        """ Initialize weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm) and module.bias is not None:
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class SpliceosomeModelWithTranscriptProbLoss(nn.Module):
    def __init__(self, config, in_channels, layer_channels, num_layer, use_ref_potential=True):
        super().__init__()

        self.config = config
        
        self.gene_start_rep = nn.Parameter(torch.zeros(config.site_act_rep_channels))
        self.use_ref_potential = use_ref_potential
        self.ref_potential = nn.Parameter(torch.zeros(1))
        self.gene_start_token_position = config.gene_start_token_position
        self.gene_end_rep =  nn.Parameter(torch.zeros(config.site_act_rep_channels))
        self.gene_end_token_position = config.gene_end_token_position

        self.spliceosome_net = SpliceosomeNet(config, in_channels, layer_channels, num_layer)

        self.loss_fct = MSELoss()

        self.init_weights()

    def forward(self, splice_site_reps, transcripts_splice_junctions, fake_transcripts_splice_junctions=None, alabels=None, dlabels=None, tlabels=None, return_transcript_probs=False, output_np_prob=False):
        # splice_site_reps: [ {pos1: Tensor, pos2: Tensor }, {..}, .. ], array of dict where each dict stores splice sites' embedding
        # transcripts_splice_junctions: example [ [ [(-1, 0), (857, 1904), (2031, 9999999)], [(-1, 0), (857, 2433), (2578, 9999999)], .. ], ..] (batch) array of (gene) array of (transcript) array of (junction) tuple where each tuple stores donor and acceptors' positions
        # fake_transcripts_splice_junctions: example [ [ [(-1, 2), (857, 1904), (2031, 9999999)], [(-1, 0), (857, 1905), (2031, 9999999)], .. ], ..] (batch) array of (gene) array of (transcript) array of (junction) tuple where each tuple stores donor and acceptors' positions

        if alabels is not None and dlabels is not None:
            loss_reg = 0

        if tlabels is not None:
            loss_reg_transcript = 0

        # compute potentials, transcript potentials and probabilities of each splice site
        transcripts_splice_junctions_str = stringify_gene_batch(transcripts_splice_junctions)
        # [ [ ['-1^0', '857^1904', '2031^9999999'], ['-1^0', '857^2433', '2578^9999999'], .. ], ..]
        if fake_transcripts_splice_junctions is not None:
            fake_transcripts_splice_junctions_str = stringify_gene_batch(fake_transcripts_splice_junctions)

            # combine splice junctions from real and fake transcripts
            all_transcripts_splice_junctions_str = [real + fake for real, fake in zip(transcripts_splice_junctions_str, fake_transcripts_splice_junctions_str)]
        else:
            all_transcripts_splice_junctions_str = transcripts_splice_junctions_str
        
        # find unique splice junctions in each gene
        unique_junctions = batch_unique_elements(all_transcripts_splice_junctions_str)

        # construct junction embs for genes
        junction_embs = []
        num_reg_labels = 0
        num_reg_transcript_labels = 0
        for gene_unique_junctions, gene_splice_site_reps in zip(unique_junctions, splice_site_reps):
            gene_junction_embs = []
            for gene_unique_junction in gene_unique_junctions:
                don, acc = gene_unique_junction.split('^')
                don = int(don)
                acc = int(acc)

                if don == self.gene_start_token_position:
                    don_emb = self.gene_start_rep
                else:
                    don_emb = gene_splice_site_reps[don]

                if acc == self.gene_end_token_position:
                    acc_emb = self.gene_end_rep
                else:
                    acc_emb = gene_splice_site_reps[acc]

                junction_emb = torch.cat([don_emb, acc_emb], dim=-1)
                gene_junction_embs.append(junction_emb)

            gene_junction_embs = torch.stack(gene_junction_embs, dim=0)
            junction_embs.append(gene_junction_embs)


        # compute junctions' potentials
        junction_potentials = []
        for gene_junction_embs in junction_embs:
            gene_junction_embs = gene_junction_embs
            gene_junction_potentials = self.spliceosome_net(gene_junction_embs)
            junction_potentials.append(gene_junction_potentials)

        # construct junction potential dict
        junction_potentials_dict = []
        for gene_unique_junctions, gene_junction_potentials in zip(unique_junctions, junction_potentials):
            gene_junction_potentials_dict = {}
            for junction_ind, gene_unique_junction in enumerate(gene_unique_junctions):
                gene_junction_potential = gene_junction_potentials[junction_ind]
                gene_junction_potentials_dict[gene_unique_junction] = gene_junction_potential

            junction_potentials_dict.append(gene_junction_potentials_dict)

        # compute transcripts' potentials
        splice_sites_prob = []
        transcripts_prob = []
        for gene_ind, (gene_all_transcripts_splice_junctions_str, gene_junction_potentials_dict) in enumerate(zip(all_transcripts_splice_junctions_str, junction_potentials_dict)):
            gene_transcripts_potential = []

            for gene_transcript_splice_junctions_str in gene_all_transcripts_splice_junctions_str:
                transcript_junction_potentials = []

                for splice_junction_str in gene_transcript_splice_junctions_str:
                    splice_junction_potential = gene_junction_potentials_dict[splice_junction_str]
                    transcript_junction_potentials.append(splice_junction_potential)
                transcript_total_potential = torch.sum(torch.stack(transcript_junction_potentials), dim=0)
                gene_transcripts_potential.append(transcript_total_potential)
            
            if self.use_ref_potential:
                gene_transcripts_potential.append(self.ref_potential)

            gene_transcripts_potential = torch.cat(gene_transcripts_potential, dim=0)

            # compute transcripts' prob
            gene_transcripts_prob = nn.functional.softmax(gene_transcripts_potential, dim=-1)

            # compute splice site prob
            gene_splice_site_prob_dict = {}
            gene_transcript_prob_dict = {}
            for transcript_ind, gene_transcript_splice_junctions_str in enumerate(gene_all_transcripts_splice_junctions_str):
                transcript_str = None
                for splice_junction_str in gene_transcript_splice_junctions_str:
                    if transcript_str is None:
                        transcript_str = splice_junction_str
                    else:
                        transcript_str = transcript_str + "_" + splice_junction_str
                    don, acc = splice_junction_str.split('^')
                    don = int(don)
                    acc = int(acc)

                    if don != self.gene_start_token_position:
                        if don in gene_splice_site_prob_dict:
                            gene_splice_site_prob_dict[don] = gene_splice_site_prob_dict[don] + gene_transcripts_prob[transcript_ind]
                        else:
                            gene_splice_site_prob_dict[don] = gene_transcripts_prob[transcript_ind]
                        
                    if acc != self.gene_end_token_position:
                        if acc in gene_splice_site_prob_dict:
                            gene_splice_site_prob_dict[acc] = gene_splice_site_prob_dict[acc] + gene_transcripts_prob[transcript_ind]
                        else:
                            gene_splice_site_prob_dict[acc] = gene_transcripts_prob[transcript_ind]

                # save transcript prob in gene_transcript_prob_dict
                gene_transcript_prob_dict[transcript_str] = gene_transcripts_prob[transcript_ind]

            splice_sites_prob.append(gene_splice_site_prob_dict)
            transcripts_prob.append(gene_transcript_prob_dict)

            # compute acceptor & donor reg loss
            if alabels is not None and dlabels is not None:
                gene_alabels = alabels[gene_ind]
                gene_dlabels = dlabels[gene_ind]

                loss_reg_don = 0
                loss_reg_acc = 0
                for don_pos in gene_dlabels.keys():
                    label = torch.tensor(gene_dlabels[don_pos]['psi'], dtype=torch.float).to(self.config.device)
                    pred = gene_splice_site_prob_dict[don_pos]
                    loss_reg_don = loss_reg_don + self.loss_fct(pred, label)
                for acc_pos in gene_alabels.keys():
                    label = torch.tensor(gene_alabels[acc_pos]['psi'], dtype=torch.float).to(self.config.device)
                    pred = gene_splice_site_prob_dict[acc_pos]
                    loss_reg_acc = loss_reg_acc + self.loss_fct(pred, label)

                num_reg_labels = num_reg_labels + len(gene_dlabels.keys()) + len(gene_alabels.keys())

                gene_loss_reg = loss_reg_don + loss_reg_acc
                loss_reg = loss_reg + gene_loss_reg

            # compute transcript reg loss
            if tlabels is not None:
                gene_tlabels = tlabels[gene_ind]

                gene_loss_reg_transcript = 0
                for transcript in gene_tlabels.keys():
                    label = torch.tensor(gene_tlabels[transcript], dtype=torch.float).to(self.config.device)
                    pred = gene_transcript_prob_dict[transcript]
                    gene_loss_reg_transcript = gene_loss_reg_transcript + self.loss_fct(pred, label)

                num_reg_transcript_labels = num_reg_transcript_labels + len(gene_tlabels.keys())

                loss_reg_transcript = loss_reg_transcript + gene_loss_reg_transcript
        
        # Divide the total loss value by num_reg_labels to output mean reg loss value
        mean_reduced_loss_reg = loss_reg / num_reg_labels
        mean_reduced_loss_reg_transcript = loss_reg_transcript / num_reg_transcript_labels

        if output_np_prob:
            for gene_ind, gene_splice_site_prob_dict in enumerate(splice_sites_prob):
                for site in splice_sites_prob[gene_ind]:
                    splice_sites_prob[gene_ind][site] = splice_sites_prob[gene_ind][site].cpu().numpy()
                for transcript in transcripts_prob[gene_ind]:
                    transcripts_prob[gene_ind][transcript] = transcripts_prob[gene_ind][transcript].cpu().numpy()

        if alabels is not None and dlabels is not None and tlabels is not None:
            return mean_reduced_loss_reg, mean_reduced_loss_reg_transcript, splice_sites_prob, transcripts_prob
        elif alabels is not None and dlabels is not None:
            if return_transcript_probs:
                return mean_reduced_loss_reg, splice_sites_prob, transcripts_prob
            else:
                return mean_reduced_loss_reg, splice_sites_prob
        else:
            if return_transcript_probs:
                return splice_sites_prob, transcripts_prob
            else:
                return splice_sites_prob 

    def init_weights(self):
        """ Initialize weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm) and module.bias is not None:
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class SpliceosomeModelJunctionBaseline(nn.Module):
    def __init__(self, config, in_channels, layer_channels, num_layer, normalize_probs=False):
        super().__init__()
        self.config = config
        
        self.gene_start_rep = nn.Parameter(torch.zeros(config.site_act_rep_channels))
        self.gene_start_token_position = config.gene_start_token_position
        self.gene_end_rep =  nn.Parameter(torch.zeros(config.site_act_rep_channels))
        self.gene_end_token_position = config.gene_end_token_position

        self.spliceosome_net = SpliceosomeNet(config, in_channels, layer_channels, num_layer)
        self.sigmoid = nn.Sigmoid()

        self.loss_fct = MSELoss()
        self.normalize_probs = normalize_probs

        self.init_weights()

    def forward(self, splice_site_reps, transcripts_splice_junctions, fake_transcripts_splice_junctions=None, alabels=None, dlabels=None):
        # splice_site_reps: [ {pos1: Tensor, pos2: Tensor }, {..}, .. ], array of dict where each dict stores splice sites' embedding
        # transcripts_splice_junctions: example [ [ [(-1, 0), (857, 1904), (2031, 9999999)], [(-1, 0), (857, 2433), (2578, 9999999)], .. ], ..] (batch) array of (gene) array of (transcript) array of (junction) tuple where each tuple stores donor and acceptors' positions
        # fake_transcripts_splice_junctions: example [ [ [(-1, 2), (857, 1904), (2031, 9999999)], [(-1, 0), (857, 1905), (2031, 9999999)], .. ], ..] (batch) array of (gene) array of (transcript) array of (junction) tuple where each tuple stores donor and acceptors' positions

        if alabels is not None and dlabels is not None:
            loss_reg = 0

        # compute potentials, transcript potentials and probabilities of each splice site
        transcripts_splice_junctions_str = stringify_gene_batch(transcripts_splice_junctions)
        # [ [ ['-1^0', '857^1904', '2031^9999999'], ['-1^0', '857^2433', '2578^9999999'], .. ], ..]
        if fake_transcripts_splice_junctions is not None:
            fake_transcripts_splice_junctions_str = stringify_gene_batch(fake_transcripts_splice_junctions)

            # combine splice junctions from real and fake transcripts
            all_transcripts_splice_junctions_str = [real + fake for real, fake in zip(transcripts_splice_junctions_str, fake_transcripts_splice_junctions_str)]
        else:
            all_transcripts_splice_junctions_str = transcripts_splice_junctions_str
        
        # find unique splice junctions in each gene
        unique_junctions = batch_unique_elements(all_transcripts_splice_junctions_str)

        # construct junction embs for genes
        junction_embs = []
        for gene_unique_junctions, gene_splice_site_reps in zip(unique_junctions, splice_site_reps):
            gene_junction_embs = []
            for gene_unique_junction in gene_unique_junctions:
                don, acc = gene_unique_junction.split('^')
                don = int(don)
                acc = int(acc)

                if don == self.gene_start_token_position:
                    don_emb = self.gene_start_rep
                else:
                    don_emb = gene_splice_site_reps[don]

                if acc == self.gene_end_token_position:
                    acc_emb = self.gene_end_rep
                else:
                    acc_emb = gene_splice_site_reps[acc]

                junction_emb = torch.cat([don_emb, acc_emb], dim=-1)
                gene_junction_embs.append(junction_emb)

            gene_junction_embs = torch.stack(gene_junction_embs, dim=0)
            junction_embs.append(gene_junction_embs)


        # compute junctions' probs
        junction_probs = []
        for gene_junction_embs in junction_embs:
            gene_junction_embs = gene_junction_embs
            gene_junction_potentials = self.spliceosome_net(gene_junction_embs)
            gene_junction_probs = self.sigmoid(gene_junction_potentials)
            gene_junction_probs = torch.squeeze(gene_junction_probs, dim=-1)
            junction_probs.append(gene_junction_probs)

        # construct junction probs dict
        junction_probs_dict = []
        for gene_unique_junctions, gene_junction_probs in zip(unique_junctions, junction_probs):
            gene_junction_probs_dict = {}
            for junction_ind, gene_unique_junction in enumerate(gene_unique_junctions):
                gene_junction_prob = gene_junction_probs[junction_ind]
                gene_junction_probs_dict[gene_unique_junction] = gene_junction_prob

            junction_probs_dict.append(gene_junction_probs_dict)

        # compute splice site prob
        splice_sites_prob = []
        num_reg_labels = 0
        for gene_ind, gene_junction_probs_dict in enumerate(junction_probs_dict):  
            
            # allocate splice site prob
            gene_splice_site_prob_dict = {}
            for splice_junction_str in gene_junction_probs_dict.keys():
                don, acc = splice_junction_str.split('^')
                don = int(don)
                acc = int(acc)

                if don != self.gene_start_token_position:
                    if don in gene_splice_site_prob_dict:
                        gene_splice_site_prob_dict[don] = gene_splice_site_prob_dict[don] + gene_junction_probs_dict[splice_junction_str]
                    else:
                        gene_splice_site_prob_dict[don] = gene_junction_probs_dict[splice_junction_str]
                    
                if acc != self.gene_end_token_position:
                    if acc in gene_splice_site_prob_dict:
                        gene_splice_site_prob_dict[acc] = gene_splice_site_prob_dict[acc] + gene_junction_probs_dict[splice_junction_str]
                    else:
                        gene_splice_site_prob_dict[acc] = gene_junction_probs_dict[splice_junction_str]

            if self.normalize_probs:
                # normalize prob to [0,1]
                gene_max_ss_prob = 0
                # find max prob value
                for ss in gene_splice_site_prob_dict.keys():
                    ss_prob = gene_splice_site_prob_dict[ss]
                    if ss_prob > gene_max_ss_prob:
                        gene_max_ss_prob = ss_prob

                # normalize all probs with max prob value
                for ss in gene_splice_site_prob_dict.keys():
                    ss_prob = gene_splice_site_prob_dict[ss]
                    ss_prob_normalized = ss_prob / gene_max_ss_prob
                    gene_splice_site_prob_dict[ss] = ss_prob_normalized

            splice_sites_prob.append(gene_splice_site_prob_dict)

            # compute reg loss
            if alabels is not None and dlabels is not None:
                gene_alabels = alabels[gene_ind]
                gene_dlabels = dlabels[gene_ind]

                loss_reg_don = 0
                loss_reg_acc = 0
                for don_pos in gene_dlabels.keys():
                    label = torch.tensor(gene_dlabels[don_pos]['psi'], dtype=torch.float).to(self.config.device)
                    pred = gene_splice_site_prob_dict[don_pos]
                    loss_reg_don = loss_reg_don + self.loss_fct(pred, label)
                for acc_pos in gene_alabels.keys():
                    label = torch.tensor(gene_alabels[acc_pos]['psi'], dtype=torch.float).to(self.config.device)
                    pred = gene_splice_site_prob_dict[acc_pos]
                    loss_reg_acc = loss_reg_acc + self.loss_fct(pred, label)

                num_reg_labels = num_reg_labels + len(gene_dlabels.keys()) + len(gene_alabels.keys())

                gene_loss_reg = loss_reg_don + loss_reg_acc
                loss_reg = loss_reg + gene_loss_reg

        # Divide the total loss value by num_reg_labels to output mean reg loss value
        mean_reduced_loss_reg = loss_reg / num_reg_labels

        if alabels is not None and dlabels is not None:
            return mean_reduced_loss_reg, splice_sites_prob
        else:
            return splice_sites_prob 

    def init_weights(self):
        """ Initialize weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm) and module.bias is not None:
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)