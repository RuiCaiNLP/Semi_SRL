from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
from nnet.util import *

import numpy as np
import torch
import math
import torch.nn as nn
import torch.autograd
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import torch.nn.init as init
from numpy import random as nr
from operator import itemgetter

_BIG_NUMBER = 10. ** 6.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def cat(l, dimension=-1):
    valid_l = l
    if dimension < 0:
        dimension += len(valid_l[0].size())
    return torch.cat(valid_l, dimension)


class MLP(nn.Linear):
    def __init__(self, in_features, out_features, activation=None, dropout=0.0, bias=True):
        super(MLP, self).__init__(in_features, out_features, bias)
        if activation is None:
            self._activate = lambda x: x
        else:
            if not callable(activation):
                raise ValueError("activation must be callable, but got {}".format(type(activation)))
            self._activate = activation
        assert dropout == 0 or type(dropout) == float
        self._dropout_ratio = dropout
        if dropout > 0:
            self._dropout = nn.Dropout(p=self._dropout_ratio)

    def forward(self, x):
        size = x.size()
        if len(size) > 2:
            y = super(MLP, self).forward(
                x.contiguous().view(-1, size[-1]))
            y = y.view(size[0:-1] + (-1,))
        else:
            y = super(MLP, self).forward(x)
        if self._dropout_ratio > 0:
            return self._dropout(self._activate(y))
        else:
            return self._activate(y)


class BiLSTMTagger(nn.Module):

    # def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
    def __init__(self, hps, *_):
        super(BiLSTMTagger, self).__init__()

        batch_size = hps['batch_size']
        lstm_hidden_dim = hps['sent_hdim']
        sent_embedding_dim_DEP = 2 * hps['sent_edim']
        sent_embedding_dim_SRL = 3 * hps['sent_edim'] + 1 * hps['pos_edim'] + 16

        self.sent_embedding_dim_DEP = sent_embedding_dim_DEP
        ## for the region mark
        role_embedding_dim = hps['role_edim']
        frame_embedding_dim = role_embedding_dim
        vocab_size = hps['vword']

        self.tagset_size = hps['vbio']
        self.pos_size = hps['vpos']
        self.dep_size = hps['vdep']
        self.frameset_size = hps['vframe']
        self.num_layers = hps['rec_layers']
        self.batch_size = batch_size
        self.hidden_dim = lstm_hidden_dim
        self.word_emb_dim = hps['sent_edim']
        self.specific_dep_size = hps['svdep']

        self.word_embeddings_SRL = nn.Embedding(vocab_size, hps['sent_edim'])
        self.word_embeddings_DEP = nn.Embedding(vocab_size, hps['sent_edim'])
        self.pos_embeddings = nn.Embedding(self.pos_size, hps['pos_edim'])
        self.pos_embeddings_DEP = nn.Embedding(self.pos_size, hps['pos_edim'])
        self.p_lemma_embeddings = nn.Embedding(self.frameset_size, hps['sent_edim'])
        self.dep_embeddings = nn.Embedding(self.dep_size, self.pos_size)
        self.region_embeddings = nn.Embedding(2, 16)
        # self.lr_dep_embeddings = nn.Embedding(self.lr_dep_size, hps[])

        self.word_fixed_embeddings = nn.Embedding(vocab_size, hps['sent_edim'])
        self.word_fixed_embeddings.weight.data.copy_(torch.from_numpy(hps['word_embeddings']))

        self.word_fixed_embeddings_DEP = nn.Embedding(vocab_size, hps['sent_edim'])
        self.word_fixed_embeddings_DEP.weight.data.copy_(torch.from_numpy(hps['word_embeddings']))

        self.role_embeddings = nn.Embedding(self.tagset_size, role_embedding_dim)
        self.frame_embeddings = nn.Embedding(self.frameset_size, frame_embedding_dim)

        self.hidden2tag = nn.Linear(4 * lstm_hidden_dim, 2 * lstm_hidden_dim)
        self.MLP = nn.Linear(2 * lstm_hidden_dim, self.dep_size)
        self.tag2hidden = nn.Linear(self.dep_size, self.pos_size)

        self.hidden2tag_spe = nn.Linear(2 * lstm_hidden_dim, 2 * lstm_hidden_dim)
        self.MLP_spe = nn.Linear(2 * lstm_hidden_dim, 4)
        self.tag2hidden_spe = nn.Linear(4, self.pos_size)

        # self.elmo_embeddings_0 = nn.Embedding(vocab_size, 1024)
        # self.elmo_embeddings_0.weight.data.copy_(torch.from_numpy(hps['elmo_embeddings_0']))

        # self.elmo_embeddings_1 = nn.Embedding(vocab_size, 1024)
        # self.elmo_embeddings_1.weight.data.copy_(torch.from_numpy(hps['elmo_embeddings_1']))

        self.elmo_emb_size = 200
        self.elmo_mlp_word = nn.Sequential(nn.Linear(1024, self.elmo_emb_size), nn.ReLU())
        self.elmo_word = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.elmo_gamma_word = nn.Parameter(torch.ones(1))

        self.elmo_mlp = nn.Sequential(nn.Linear(2 * lstm_hidden_dim, self.elmo_emb_size), nn.ReLU())
        self.elmo_w = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.elmo_gamma = nn.Parameter(torch.ones(1))

        self.SRL_input_dropout = nn.Dropout(p=0.3)
        self.DEP_input_dropout = nn.Dropout(p=0.3)
        self.hidden_state_dropout_DEP = nn.Dropout(p=0.3)
        self.hidden_state_dropout_1 = nn.Dropout(p=0.3)
        self.hidden_state_dropout_2 = nn.Dropout(p=0.3)
        self.word_dropout = nn.Dropout(p=0.0)
        self.predicate_dropout = nn.Dropout(p=0.0)
        self.label_dropout = nn.Dropout(p=0.5)
        self.link_dropout = nn.Dropout(p=0.5)
        # self.use_dropout = nn.Dropout(p=0.2)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.num_layers = 1
        self.BiLSTM_0 = nn.LSTM(input_size=sent_embedding_dim_DEP, hidden_size=lstm_hidden_dim, batch_first=True,
                                bidirectional=True, num_layers=self.num_layers)

        init.orthogonal_(self.BiLSTM_0.all_weights[0][0])
        init.orthogonal_(self.BiLSTM_0.all_weights[0][1])
        init.orthogonal_(self.BiLSTM_0.all_weights[1][0])
        init.orthogonal_(self.BiLSTM_0.all_weights[1][1])

        self.num_layers = 1
        self.BiLSTM_1 = nn.LSTM(input_size=lstm_hidden_dim * 2, hidden_size=lstm_hidden_dim, batch_first=True,
                                bidirectional=True, num_layers=self.num_layers)

        init.orthogonal_(self.BiLSTM_1.all_weights[0][0])
        init.orthogonal_(self.BiLSTM_1.all_weights[0][1])
        init.orthogonal_(self.BiLSTM_1.all_weights[1][0])
        init.orthogonal_(self.BiLSTM_1.all_weights[1][1])

        self.num_layers = 3
        self.BiLSTM_SRL = nn.LSTM(input_size=sent_embedding_dim_SRL + 200,
                                  hidden_size=lstm_hidden_dim, batch_first=True,
                                  bidirectional=True, num_layers=self.num_layers)

        init.orthogonal_(self.BiLSTM_SRL.all_weights[0][0])
        init.orthogonal_(self.BiLSTM_SRL.all_weights[0][1])
        init.orthogonal_(self.BiLSTM_SRL.all_weights[1][0])
        init.orthogonal_(self.BiLSTM_SRL.all_weights[1][1])

        # non-linear map to role embedding
        self.role_map = nn.Linear(in_features=role_embedding_dim * 2, out_features=self.hidden_dim * 4)

        self.map_dim = lstm_hidden_dim

        self.ldims = lstm_hidden_dim
        self.hidLayerFOH = nn.Linear(self.ldims * 2, self.ldims)
        self.hidLayerFOM = nn.Linear(self.ldims * 2, self.ldims)
        self.W_R_link = nn.Parameter(torch.rand(lstm_hidden_dim + 1, lstm_hidden_dim))

        self.hidLayerFOH_FF = nn.Linear(self.ldims, self.ldims)
        self.hidLayerFOM_FF = nn.Linear(self.ldims, self.ldims)
        self.W_R_link_FF = nn.Parameter(torch.rand(lstm_hidden_dim + 1, lstm_hidden_dim))
        self.hidLayerFOH_BB = nn.Linear(self.ldims, self.ldims)
        self.hidLayerFOM_BB = nn.Linear(self.ldims, self.ldims)
        self.W_R_link_BB = nn.Parameter(torch.rand(lstm_hidden_dim + 1, lstm_hidden_dim))
        self.hidLayerFOH_BF = nn.Linear(self.ldims, self.ldims)
        self.hidLayerFOM_BF = nn.Linear(self.ldims, self.ldims)
        self.W_R_link_BF = nn.Parameter(torch.rand(lstm_hidden_dim + 1, lstm_hidden_dim))
        self.hidLayerFOH_FB = nn.Linear(self.ldims, self.ldims)
        self.hidLayerFOM_FB = nn.Linear(self.ldims, self.ldims)
        self.W_R_link_FB = nn.Parameter(torch.rand(lstm_hidden_dim + 1, lstm_hidden_dim))


        self.hidLayerFOH_tag = nn.Linear(self.ldims * 2, self.ldims)
        self.hidLayerFOM_tag = nn.Linear(self.ldims * 2, self.ldims)
        self.W_R_tag = nn.Parameter(torch.rand(lstm_hidden_dim + 1, self.dep_size*(1 + lstm_hidden_dim)))

        self.Non_Predicate_Proj = nn.Linear(2 * lstm_hidden_dim, lstm_hidden_dim)
        self.Predicate_Proj = nn.Linear(2 * lstm_hidden_dim, lstm_hidden_dim)
        self.W_R = nn.Parameter(torch.rand(self.map_dim + 1, self.tagset_size * (self.map_dim + 1)))

        self.VR_embedding = nn.Parameter(
            torch.from_numpy(np.ones((1, sent_embedding_dim_DEP), dtype='float32')))

        self.mid_hidden = lstm_hidden_dim
        self.POS_MLP = nn.Sequential(nn.Linear(2 * lstm_hidden_dim, lstm_hidden_dim), nn.ReLU(),
                                     nn.Linear(lstm_hidden_dim, self.pos_size))

        # Init hidden state
        self.hidden = self.init_hidden_spe()
        self.hidden_2 = self.init_hidden_spe()
        self.hidden_3 = self.init_hidden_spe()
        self.hidden_4 = self.init_hidden_share()

    def init_hidden_share(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        # return (Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
        #        Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))
        return (torch.zeros(3 * 2, self.batch_size, self.hidden_dim, requires_grad=False).to(device),
                torch.zeros(3 * 2, self.batch_size, self.hidden_dim, requires_grad=False).to(device))

    def init_hidden_spe(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        # return (Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
        #        Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))
        return (torch.zeros(1 * 2, self.batch_size, self.hidden_dim, requires_grad=False).to(device),
                torch.zeros(1 * 2, self.batch_size, self.hidden_dim, requires_grad=False).to(device))

    def mask_loss(self, Semi_loss, lengths):

        for i in range(Semi_loss.size()[0]):
            for j in range(Semi_loss.size()[1]):
                if j >= lengths[i]:
                    Semi_loss[i][j] = 0 * Semi_loss[i][j]
        return Semi_loss



    def Semi_DEP_Loss(self, hidden_forward, hidden_backward, TagProbs_use, sentence,  lengths):

        tag_mask = np.zeros(TagProbs_use.size(), dtype='float32')
        for i in range(self.batch_size):
            for j in range(len(sentence[0]) + 1):
                if j == 0 or j > lengths[i]:
                    tag_mask[i][j] -= _BIG_NUMBER
                    continue
                for k in range(len(sentence[0]) + 1):
                    if k > lengths[i]:
                        tag_mask[i, j, k] -= _BIG_NUMBER
        tag_mask = torch.from_numpy(tag_mask).to(device)


        TagProbs_use_softmax = F.softmax(TagProbs_use, dim=2).detach()
        sample_nums = lengths.sum()
        log(lengths)
        log(sample_nums)
        unlabeled_loss_function = nn.KLDivLoss(reduce = False)
        ## Dependency Extractor FF
        Head_hidden = F.relu(self.hidLayerFOH_FF(hidden_forward))
        Dependent_hidden = F.relu(self.hidLayerFOM_FF(hidden_forward))

        bias_one = torch.ones((self.batch_size, len(sentence[0]) + 1, 1)).to(device)
        Dependent_hidden = torch.cat((Dependent_hidden, Variable(bias_one)), 2)
        left_part = torch.mm(Dependent_hidden.view(self.batch_size * (len(sentence[0]) + 1), -1), self.W_R_link_FF)
        left_part = left_part.view(self.batch_size, (len(sentence[0]) + 1), -1)
        Head_hidden = Head_hidden.view(self.batch_size, (len(sentence[0]) + 1), -1).transpose(1, 2)
        tag_space = torch.bmm(left_part, Head_hidden).view(self.batch_size, len(sentence[0]) + 1, len(sentence[0]) + 1)
        tag_space = tag_space + tag_mask
        dep_tag_space = tag_space
        log(dep_tag_space)
        DEPprobs_student = F.log_softmax(dep_tag_space, dim=2)
        log(DEPprobs_student)
        DEP_FF_loss = unlabeled_loss_function(DEPprobs_student, TagProbs_use_softmax)
        log(TagProbs_use_softmax)

        ## Dependency Extractor BB
        Head_hidden = F.relu(self.hidLayerFOH_BB(hidden_backward))
        Dependent_hidden = F.relu(self.hidLayerFOM_BB(hidden_backward))
        bias_one = torch.ones((self.batch_size, len(sentence[0]) + 1, 1)).to(device)
        Dependent_hidden = torch.cat((Dependent_hidden, Variable(bias_one)), 2)
        left_part = torch.mm(Dependent_hidden.view(self.batch_size * (len(sentence[0]) + 1), -1), self.W_R_link_BB)
        left_part = left_part.view(self.batch_size, (len(sentence[0]) + 1), -1)
        Head_hidden = Head_hidden.view(self.batch_size, (len(sentence[0]) + 1), -1).transpose(1, 2)
        tag_space = torch.bmm(left_part, Head_hidden).view(self.batch_size, len(sentence[0]) + 1, len(sentence[0]) + 1)
        tag_space = tag_space + tag_mask
        dep_tag_space = tag_space
        DEPprobs_student = F.log_softmax(dep_tag_space, dim=2)
        DEP_BB_loss = unlabeled_loss_function(DEPprobs_student, TagProbs_use_softmax)

        ## Dependency Extractor FB
        Head_hidden = F.relu(self.hidLayerFOH_FB(hidden_forward))
        Dependent_hidden = F.relu(self.hidLayerFOM_FB(hidden_backward))
        bias_one = torch.ones((self.batch_size, len(sentence[0]) + 1, 1)).to(device)
        Dependent_hidden = torch.cat((Dependent_hidden, Variable(bias_one)), 2)
        left_part = torch.mm(Dependent_hidden.view(self.batch_size * (len(sentence[0]) + 1), -1), self.W_R_link_FB)
        left_part = left_part.view(self.batch_size, (len(sentence[0]) + 1), -1)
        Head_hidden = Head_hidden.view(self.batch_size, (len(sentence[0]) + 1), -1).transpose(1, 2)
        tag_space = torch.bmm(left_part, Head_hidden).view(self.batch_size, len(sentence[0]) + 1, len(sentence[0]) + 1)
        tag_space = tag_space + tag_mask
        dep_tag_space = tag_space
        DEPprobs_student = F.log_softmax(dep_tag_space, dim=2)
        DEP_FB_loss = unlabeled_loss_function(DEPprobs_student, TagProbs_use_softmax)

        ## Dependency Extractor BF
        Head_hidden = F.relu(self.hidLayerFOH_BF(hidden_backward))
        Dependent_hidden = F.relu(self.hidLayerFOM_BF(hidden_forward))
        bias_one = torch.ones((self.batch_size, len(sentence[0]) + 1, 1)).to(device)
        Dependent_hidden = torch.cat((Dependent_hidden, Variable(bias_one)), 2)
        left_part = torch.mm(Dependent_hidden.view(self.batch_size * (len(sentence[0]) + 1), -1), self.W_R_link_BF)
        left_part = left_part.view(self.batch_size, (len(sentence[0]) + 1), -1)
        Head_hidden = Head_hidden.view(self.batch_size, (len(sentence[0]) + 1), -1).transpose(1, 2)
        tag_space = torch.bmm(left_part, Head_hidden).view(self.batch_size, len(sentence[0]) + 1, len(sentence[0]) + 1)
        tag_space = tag_space + tag_mask
        dep_tag_space = tag_space
        DEPprobs_student = F.log_softmax(dep_tag_space, dim=2)
        DEP_BF_loss = unlabeled_loss_function(DEPprobs_student, TagProbs_use_softmax)

        DEP_Semi_loss = DEP_FF_loss + DEP_BB_loss + DEP_BF_loss + DEP_FB_loss

        loss_mask = np.ones(TagProbs_use.size(), dtype='float32')
        for i in range(self.batch_size):
            for j in range(len(sentence[0]) + 1):
                if j == 0 or j > lengths[i]:
                    loss_mask[i][j] = 0.0
                    continue
                for k in range(len(sentence[0]) + 1):
                    if k > lengths[i]:
                        loss_mask[i, j, k] = 0.0
        loss_mask = torch.from_numpy(loss_mask).to(device)
        DEP_Semi_loss = DEP_Semi_loss * loss_mask
        DEP_Semi_loss = torch.sum(DEP_Semi_loss)
        return DEP_Semi_loss/sample_nums

    def find_predicate_embeds(self, hidden_states, target_idx_in):
        Label_composer = hidden_states
        predicate_embeds = Label_composer[np.arange(0, Label_composer.size()[0]), target_idx_in]
        # T * B * H
        added_embeds = torch.zeros(Label_composer.size()[1], Label_composer.size()[0],
                                   Label_composer.size()[2]).to(device)
        concat_embeds = (added_embeds + predicate_embeds).transpose(0, 1)
        return concat_embeds



    def CVT_train(self, sentence, p_sentence, sent_mask, lengths):
        ## start unlabeled training:

        embeds_DEP = self.word_embeddings_DEP(sentence)
        fixed_embeds_DEP = self.word_fixed_embeddings(p_sentence)
        fixed_embeds_DEP = fixed_embeds_DEP.view(self.batch_size, len(sentence[0]), self.word_emb_dim)
        embeds_forDEP = torch.cat((embeds_DEP, fixed_embeds_DEP), 2)
        add_zero = torch.zeros((self.batch_size, 1, self.sent_embedding_dim_DEP)).to(device)
        embeds_forDEP_cat = torch.cat((self.VR_embedding + add_zero, embeds_forDEP), 1)
        embeds_forDEP = self.DEP_input_dropout(embeds_forDEP_cat)

        # first layer
        embeds_sort, lengths_sort, unsort_idx = self.sort_batch(embeds_forDEP, lengths + 1)
        embeds_sort = rnn.pack_padded_sequence(embeds_sort, lengths_sort, batch_first=True)
        # hidden states [time_steps * batch_size * hidden_units]
        hidden_states, self.hidden = self.BiLSTM_0(embeds_sort, self.hidden)
        # it seems that hidden states is already batch first, we don't need swap the dims
        # hidden_states = hidden_states.permute(1, 2, 0).contiguous().view(self.batch_size, -1, )
        hidden_states, lens = rnn.pad_packed_sequence(hidden_states, batch_first=True)
        # hidden_states = hidden_states.transpose(0, 1)
        hidden_states_0 = hidden_states[unsort_idx]
        hidden_states_0 = self.hidden_state_dropout_1(hidden_states_0)

        hidden_forward, hidden_backward = hidden_states_0.split(self.hidden_dim, 2)

        # second_layer
        embeds_sort, lengths_sort, unsort_idx = self.sort_batch(hidden_states_0, lengths + 1)
        embeds_sort = rnn.pack_padded_sequence(embeds_sort, lengths_sort, batch_first=True)
        # hidden states [time_steps * batch_size * hidden_units]
        hidden_states, self.hidden_2 = self.BiLSTM_1(embeds_sort, self.hidden_2)
        # it seems that hidden states is already batch first, we don't need swap the dims
        # hidden_states = hidden_states.permute(1, 2, 0).contiguous().view(self.batch_size, -1, )
        hidden_states, lens = rnn.pad_packed_sequence(hidden_states, batch_first=True)
        # hidden_states = hidden_states.transpose(0, 1)
        hidden_states_1 = hidden_states[unsort_idx]
        hidden_states_1 = self.hidden_state_dropout_2(hidden_states_1)

        Head_hidden = F.relu(self.hidLayerFOH(hidden_states_1))
        Dependent_hidden = F.relu(self.hidLayerFOM(hidden_states_1))
        bias_one = torch.ones((self.batch_size, len(sentence[0]) + 1, 1)).to(device)
        Dependent_hidden = torch.cat((Dependent_hidden, Variable(bias_one)), 2)

        left_part = torch.mm(Dependent_hidden.view(self.batch_size * (len(sentence[0]) + 1), -1), self.W_R_link)
        left_part = left_part.view(self.batch_size, (len(sentence[0]) + 1), -1)
        Head_hidden = Head_hidden.view(self.batch_size, (len(sentence[0]) + 1), -1).transpose(1, 2)
        tag_space = torch.bmm(left_part, Head_hidden).view(self.batch_size, len(sentence[0]) + 1, len(sentence[0]) + 1)

        tag_mask = np.zeros(tag_space.size(), dtype='float32')
        for i in range(self.batch_size):
            for j in range(len(sentence[0]) + 1):
                if j == 0 or j > lengths[i]:
                    tag_mask[i][j] -= _BIG_NUMBER
                    continue
                for k in range(len(sentence[0]) + 1):
                    if k > lengths[i]:
                        tag_mask[i, j, k] -= _BIG_NUMBER
        tag_mask = torch.from_numpy(tag_mask).to(device)

        tag_space = tag_mask + tag_space

        TagProbs_use = tag_space.view(self.batch_size, len(sentence[0])+1, -1).detach()
        CVT_DEP_Loss = self.Semi_DEP_Loss(hidden_forward, hidden_backward, TagProbs_use, sentence, lengths)

        return CVT_DEP_Loss



    def forward(self, sentence, p_sentence, pos_tags, sent_mask, lengths, target_idx_in, region_marks,
                local_roles_voc, frames, local_roles_mask,
                sent_pred_lemmas_idx, dep_tags, dep_heads, targets, gold_pos_tag, specific_dep_relations,
                Chars=None, Predicate_indicator = None, test=False,
                unlabeled_sentence=None, p_unlabeled_sentence=None, unlabeled_sent_mask=None, unlabeled_lengths=None, cvt_train=False):

        if cvt_train:
            CVT_SRL_Loss = self.CVT_train(unlabeled_sentence, p_unlabeled_sentence, unlabeled_sent_mask, unlabeled_lengths)
            return CVT_SRL_Loss

        """
        DEP_learning
        """
        embeds_DEP = self.word_embeddings_DEP(sentence)
        fixed_embeds_DEP = self.word_fixed_embeddings(p_sentence)
        fixed_embeds_DEP = fixed_embeds_DEP.view(self.batch_size, len(sentence[0]), self.word_emb_dim)
        embeds_forDEP = torch.cat((embeds_DEP, fixed_embeds_DEP), 2)
        add_zero = torch.zeros((self.batch_size, 1, self.sent_embedding_dim_DEP)).to(device)
        embeds_forDEP_cat = torch.cat((self.VR_embedding + add_zero, embeds_forDEP), 1)
        embeds_forDEP = self.DEP_input_dropout(embeds_forDEP_cat)

        # first layer
        embeds_sort, lengths_sort, unsort_idx = self.sort_batch(embeds_forDEP, lengths+1)
        embeds_sort = rnn.pack_padded_sequence(embeds_sort, lengths_sort, batch_first=True)
        # hidden states [time_steps * batch_size * hidden_units]
        hidden_states, self.hidden = self.BiLSTM_0(embeds_sort, self.hidden)
        # it seems that hidden states is already batch first, we don't need swap the dims
        # hidden_states = hidden_states.permute(1, 2, 0).contiguous().view(self.batch_size, -1, )
        hidden_states, lens = rnn.pad_packed_sequence(hidden_states, batch_first=True)
        # hidden_states = hidden_states.transpose(0, 1)
        hidden_states_0 = hidden_states[unsort_idx]
        hidden_states_0 = self.hidden_state_dropout_1(hidden_states_0)

        # second_layer
        embeds_sort, lengths_sort, unsort_idx = self.sort_batch(hidden_states_0, lengths+1)
        embeds_sort = rnn.pack_padded_sequence(embeds_sort, lengths_sort, batch_first=True)
        # hidden states [time_steps * batch_size * hidden_units]
        hidden_states, self.hidden_2 = self.BiLSTM_1(embeds_sort, self.hidden_2)
        # it seems that hidden states is already batch first, we don't need swap the dims
        # hidden_states = hidden_states.permute(1, 2, 0).contiguous().view(self.batch_size, -1, )
        hidden_states, lens = rnn.pad_packed_sequence(hidden_states, batch_first=True)
        # hidden_states = hidden_states.transpose(0, 1)
        hidden_states_1 = hidden_states[unsort_idx]
        hidden_states_1 = self.hidden_state_dropout_2(hidden_states_1)

        #hidden_states_1 = torch.cat((hidden_states_0, hidden_states_1), 2)
        Head_hidden = F.relu(self.hidLayerFOH(hidden_states_1))
        Dependent_hidden = F.relu(self.hidLayerFOM(hidden_states_1))


        #bias_one = torch.ones((self.batch_size, len(sentence[0]) + 1, 1)).to(device)
        #Head_hidden = torch.cat((Head_hidden, Variable(bias_one)), 2)

        bias_one = torch.ones((self.batch_size, len(sentence[0]) + 1, 1)).to(device)
        Dependent_hidden = torch.cat((Dependent_hidden, Variable(bias_one)), 2)

        left_part = torch.mm(Dependent_hidden.view(self.batch_size * (len(sentence[0]) + 1), -1), self.W_R_link)
        left_part = left_part.view(self.batch_size, (len(sentence[0]) + 1), -1)
        Head_hidden = Head_hidden.view(self.batch_size, (len(sentence[0]) + 1), -1).transpose(1, 2)
        tag_space = torch.bmm(left_part, Head_hidden).view(self.batch_size, len(sentence[0]) + 1, len(sentence[0]) + 1)

        tag_mask = np.zeros(tag_space.size(), dtype='float32')
        for i in range(self.batch_size):
            for j in range(len(sentence[0]) + 1):
                if j == 0 or j > lengths[i]:
                    tag_mask[i][j] -= _BIG_NUMBER
                    continue
                for k in range(len(sentence[0]) + 1):
                    if k > lengths[i]:
                        tag_mask[i, j, k] -= _BIG_NUMBER
        tag_mask = torch.from_numpy(tag_mask).to(device)

        tag_space = tag_space + tag_mask

        tag_space = tag_space.contiguous().view(self.batch_size * (len(sentence[0])+1), len(sentence[0]) + 1)
        heads = np.argmax(tag_space.cpu().data.numpy(), axis=1)

        loss_function = nn.CrossEntropyLoss(ignore_index=-1)
        Link_DEPloss = loss_function(tag_space, torch.from_numpy(dep_heads).to(device).view(-1))

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++


        ##########################################
        tag_space = self.POS_MLP(hidden_states_1[:, 1:, :]).view(
            len(sentence[0]) * self.batch_size, -1)

        POS_label = np.argmax(tag_space.cpu().data.numpy(), axis=1)

        loss_function = nn.CrossEntropyLoss(ignore_index=0)
        POS_loss = loss_function(tag_space, gold_pos_tag.view(-1))

        ##########################################
        Link_right, Link_all, \
        POS_right, POS_all, PI_right, PI_all = 0., 0., 0., 0., 0., 0.

        for a, b in zip(heads, dep_heads.flatten()):
            if b == -1:
                continue
            Link_all += 1
            if a == b:
                Link_right += 1

        for a, b in zip(POS_label, gold_pos_tag.view(-1).cpu().data.numpy()):
            if b == 0:
                continue
            POS_all += 1
            if a == b:
                POS_right += 1

        SRLloss = 0
        PI_loss = 0
        SRLprobs = 0
        POS_loss = 0
        Tag_DEPloss = 0

        return SRLloss, Link_DEPloss, Tag_DEPloss, POS_loss, PI_loss, SRLprobs, Link_right, Link_all, \
               POS_right, POS_all, PI_right, PI_all \

    @staticmethod
    def sort_batch(x, l):
        l = torch.from_numpy(np.asarray(l))
        l_sorted, sidx = l.sort(0, descending=True)
        x_sorted = x[sidx]
        _, unsort_idx = sidx.sort()
        return x_sorted, l_sorted, unsort_idx