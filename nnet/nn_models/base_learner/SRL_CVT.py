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
        sent_embedding_dim_SRL = 2 * hps['sent_edim'] + 0 * hps['pos_edim'] + 16

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
        self.DEP_input_dropout = nn.Dropout(p=0.5)
        self.hidden_state_dropout_DEP = nn.Dropout(p=0.3)

        self.hidden_state_dropout_1 = nn.Dropout(p=0.5)
        self.hidden_state_dropout_2 = nn.Dropout(p=0.5)
        self.DEP_hidden_state_dropout_1 = nn.Dropout(p=0.5)
        self.DEP_hidden_state_dropout_2 = nn.Dropout(p=0.5)

        self.head_dropout = nn.Dropout(p=0.3)
        self.dep_dropout = nn.Dropout(p=0.3)

        self.SRL_input_dropout_unlabeled = nn.Dropout(p=0.1)
        self.DEP_input_dropout_unlabeled = nn.Dropout(p=0)
        self.hidden_state_dropout_1_unlabeled = nn.Dropout(p=0.2)
        self.hidden_state_dropout_2_unlabeled = nn.Dropout(p=0.2)
        self.DEP_hidden_state_dropout_1_unlabeled = nn.Dropout(p=0.2)
        self.DEP_hidden_state_dropout_2_unlabeled = nn.Dropout(p=0)
        self.head_dropout_unlabeled = nn.Dropout(p=0.2)
        self.dep_dropout_unlabeled = nn.Dropout(p=0.2)

        self.head_dropout_unlabeled_FF = nn.Dropout(p=0.1)
        self.dep_dropout_unlabeled_FF = nn.Dropout(p=0.1)
        self.head_dropout_unlabeled_BB = nn.Dropout(p=0.1)
        self.dep_dropout_unlabeled_BB = nn.Dropout(p=0.1)
        self.head_dropout_unlabeled_FB = nn.Dropout(p=0.1)
        self.dep_dropout_unlabeled_FB = nn.Dropout(p=0.1)
        self.head_dropout_unlabeled_BF = nn.Dropout(p=0.1)
        self.dep_dropout_unlabeled_BF = nn.Dropout(p=0.1)
        # self.use_dropout = nn.Dropout(p=0.2)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.SA_primary_num_layers = 1
        self.BiLSTM_SA_primary = nn.LSTM(input_size=sent_embedding_dim_DEP, hidden_size=lstm_hidden_dim, batch_first=True,
                                bidirectional=True, num_layers=self.SA_primary_num_layers)

        init.orthogonal_(self.BiLSTM_SA_primary.all_weights[0][0])
        init.orthogonal_(self.BiLSTM_SA_primary.all_weights[0][1])
        init.orthogonal_(self.BiLSTM_SA_primary.all_weights[1][0])
        init.orthogonal_(self.BiLSTM_SA_primary.all_weights[1][1])

        self.SA_high_num_layers = 1
        self.BiLSTM_SA_high = nn.LSTM(input_size=lstm_hidden_dim * 2, hidden_size=lstm_hidden_dim, batch_first=True,
                                bidirectional=True, num_layers=self.SA_high_num_layers)

        init.orthogonal_(self.BiLSTM_SA_high.all_weights[0][0])
        init.orthogonal_(self.BiLSTM_SA_high.all_weights[0][1])
        init.orthogonal_(self.BiLSTM_SA_high.all_weights[1][0])
        init.orthogonal_(self.BiLSTM_SA_high.all_weights[1][1])

        self.SRL_primary_num_layers = 1
        self.BiLSTM_SRL_primary = nn.LSTM(input_size=sent_embedding_dim_SRL, hidden_size=lstm_hidden_dim, batch_first=True,
                                bidirectional=True, num_layers=self.SRL_primary_num_layers)

        init.orthogonal_(self.BiLSTM_SRL_primary.all_weights[0][0])
        init.orthogonal_(self.BiLSTM_SRL_primary.all_weights[0][1])
        init.orthogonal_(self.BiLSTM_SRL_primary.all_weights[1][0])
        init.orthogonal_(self.BiLSTM_SRL_primary.all_weights[1][1])

        self.SRL_high_num_layers = 2
        self.BiLSTM_SRL_high = nn.LSTM(input_size=2*lstm_hidden_dim,
                                  hidden_size=lstm_hidden_dim, batch_first=True,
                                  bidirectional=True, num_layers=self.SRL_high_num_layers)

        init.orthogonal_(self.BiLSTM_SRL_high.all_weights[0][0])
        init.orthogonal_(self.BiLSTM_SRL_high.all_weights[0][1])
        init.orthogonal_(self.BiLSTM_SRL_high.all_weights[1][0])
        init.orthogonal_(self.BiLSTM_SRL_high.all_weights[1][1])

        # non-linear map to role embedding
        self.role_map = nn.Linear(in_features=role_embedding_dim * 2, out_features=self.hidden_dim * 4)

        self.map_dim = lstm_hidden_dim



        self.ldims = lstm_hidden_dim
        self.hidLayerFOH_SRL = nn.Linear(self.ldims * 2, self.ldims)
        self.hidLayerFOM_SRL = nn.Linear(self.ldims * 2, self.ldims)
        self.W_R_SRL = nn.Parameter(torch.rand(lstm_hidden_dim + 1, self.tagset_size*(lstm_hidden_dim+1)))

        self.biaffine_mid = int(self.ldims)

        self.hidLayerFOH_SRL_FF = nn.Linear(self.ldims, self.biaffine_mid)
        self.hidLayerFOM_SRL_FF = nn.Linear(self.ldims, self.biaffine_mid)
        self.W_R_SRL_FF = nn.Parameter(torch.rand(self.biaffine_mid + 1, self.tagset_size*(self.biaffine_mid+1)))

        self.hidLayerFOH_SRL_BB = nn.Linear(self.ldims, self.biaffine_mid)
        self.hidLayerFOM_SRL_BB = nn.Linear(self.ldims, self.biaffine_mid)
        self.W_R_SRL_BB = nn.Parameter(torch.rand(self.biaffine_mid + 1, self.tagset_size*(self.biaffine_mid+1)))

        self.hidLayerFOH_SRL_BF = nn.Linear(self.ldims, self.biaffine_mid)
        self.hidLayerFOM_SRL_BF = nn.Linear(self.ldims, self.biaffine_mid)
        self.W_R_SRL_BF = nn.Parameter(torch.rand(self.biaffine_mid + 1, self.tagset_size*(self.biaffine_mid+1)))

        self.hidLayerFOH_SRL_FB = nn.Linear(self.ldims, self.biaffine_mid)
        self.hidLayerFOM_SRL_FB = nn.Linear(self.ldims, self.biaffine_mid)
        self.W_R_SRL_FB = nn.Parameter(torch.rand(self.biaffine_mid + 1, self.tagset_size*(self.biaffine_mid+1)))




        self.VR_embedding = nn.Parameter(
            torch.from_numpy(np.ones((1, sent_embedding_dim_DEP), dtype='float32')))

        self.mid_hidden = lstm_hidden_dim
        self.POS_MLP = nn.Sequential(nn.Linear(2 * lstm_hidden_dim, lstm_hidden_dim), nn.ReLU(),
                                     nn.Linear(lstm_hidden_dim, self.pos_size))

        self.PI_MLP = nn.Sequential(nn.Linear(2 * lstm_hidden_dim, lstm_hidden_dim), nn.ReLU(),
                                    nn.Linear(lstm_hidden_dim, 2))

        self.SRL_primary_hidden = self.init_SRL_primary()
        self.SRL_high_hidden = self.init_SRL_high()
        self.SA_primary_hidden = self.init_SA_primary()
        self.SA_high_hidden = self.init_SA_high()

    def init_SA_primary(self):
        return (torch.zeros(self.SA_primary_num_layers * 2, self.batch_size, self.hidden_dim, requires_grad=False).to(device),
                torch.zeros(self.SA_primary_num_layers * 2, self.batch_size, self.hidden_dim, requires_grad=False).to(device))

    def init_SA_high(self):
        return (torch.zeros(self.SA_high_num_layers * 2, self.batch_size, self.hidden_dim, requires_grad=False).to(device),
                torch.zeros(self.SA_high_num_layers * 2, self.batch_size, self.hidden_dim, requires_grad=False).to(device))

    def init_SRL_primary(self):
        return (torch.zeros(self.SRL_primary_num_layers * 2, self.batch_size, self.hidden_dim, requires_grad=False).to(device),
                torch.zeros(self.SRL_primary_num_layers * 2, self.batch_size, self.hidden_dim, requires_grad=False).to(device))

    def init_SRL_high(self):
        return (torch.zeros(self.SRL_high_num_layers * 2, self.batch_size, self.hidden_dim, requires_grad=False).to(device),
                torch.zeros(self.SRL_high_num_layers * 2, self.batch_size, self.hidden_dim, requires_grad=False).to(device))



    def mask_loss(self, Semi_loss, lengths):

        for i in range(Semi_loss.size()[0]):
            for j in range(Semi_loss.size()[1]):
                if j >= lengths[i]:
                    Semi_loss[i][j] = 0 * Semi_loss[i][j]
        return Semi_loss



    def Semi_SRL_Loss(self, hidden_forward, hidden_backward, TagProbs_use, sentence,  lengths, target_idx_in):


        TagProbs_use_softmax = F.softmax(TagProbs_use, dim=2).detach()
        TagProbs_use_softmax_log = F.log_softmax(TagProbs_use, dim=2).detach()
        Entroy_Weights = 1-torch.sum(TagProbs_use_softmax_log * TagProbs_use_softmax, dim=2).detach()

        sample_nums = lengths.sum()
        unlabeled_loss_function = nn.KLDivLoss(reduce=False)


        ## Dependency Extractor FF
        predicate_embeds = hidden_forward[np.arange(0, self.batch_size), target_idx_in]
        Head_hidden = self.head_dropout_unlabeled_FF(F.relu(self.hidLayerFOH_SRL_FF(predicate_embeds)))
        Dependent_hidden = self.dep_dropout_unlabeled_FF(F.relu(self.hidLayerFOM_SRL_FF(hidden_forward)))
        bias_one = torch.ones((self.batch_size, len(sentence[0]), 1)).to(device)
        Dependent_hidden = torch.cat((Dependent_hidden, Variable(bias_one)), 2)
        bias_one = torch.ones((self.batch_size, 1)).to(device)
        Head_hidden = torch.cat((Head_hidden, Variable(bias_one)), 1)
        left_part = torch.mm(Dependent_hidden.view(self.batch_size * len(sentence[0]), -1), self.W_R_SRL_FF)
        left_part = left_part.view(self.batch_size, len(sentence[0]) * self.tagset_size, -1)
        Head_hidden = Head_hidden.view(self.batch_size, -1, 1)
        tag_space = torch.bmm(left_part, Head_hidden).view(self.batch_size, len(sentence[0]), self.tagset_size)
        tag_space = tag_space.view(self.batch_size, len(sentence[0]), -1)
        dep_tag_space = tag_space
        DEPprobs_student = F.log_softmax(dep_tag_space, dim=2)
        DEP_FF_loss = unlabeled_loss_function(DEPprobs_student, TagProbs_use_softmax)


        ## Dependency Extractor BB
        predicate_embeds = hidden_backward[np.arange(0, self.batch_size), target_idx_in]
        Head_hidden = self.head_dropout_unlabeled_BB(F.relu(self.hidLayerFOH_SRL_BB(predicate_embeds)))
        Dependent_hidden = self.dep_dropout_unlabeled_BB(F.relu(self.hidLayerFOM_SRL_BB(hidden_backward)))
        bias_one = torch.ones((self.batch_size, len(sentence[0]), 1)).to(device)
        Dependent_hidden = torch.cat((Dependent_hidden, Variable(bias_one)), 2)
        bias_one = torch.ones((self.batch_size, 1)).to(device)
        Head_hidden = torch.cat((Head_hidden, Variable(bias_one)), 1)
        left_part = torch.mm(Dependent_hidden.view(self.batch_size * len(sentence[0]), -1), self.W_R_SRL_BB)
        left_part = left_part.view(self.batch_size, len(sentence[0]) * self.tagset_size, -1)
        Head_hidden = Head_hidden.view(self.batch_size, -1, 1)
        tag_space = torch.bmm(left_part, Head_hidden).view(self.batch_size, len(sentence[0]), self.tagset_size)
        tag_space = tag_space.view(self.batch_size, len(sentence[0]), -1)
        dep_tag_space = tag_space
        DEPprobs_student = F.log_softmax(dep_tag_space, dim=2)
        DEP_BB_loss = unlabeled_loss_function(DEPprobs_student, TagProbs_use_softmax)

        ## Dependency Extractor FB
        predicate_embeds = hidden_forward[np.arange(0, self.batch_size), target_idx_in]
        Head_hidden = self.head_dropout_unlabeled_FB(F.relu(self.hidLayerFOH_SRL_FB(predicate_embeds)))
        Dependent_hidden = self.dep_dropout_unlabeled_FB(F.relu(self.hidLayerFOM_SRL_FB(hidden_backward)))
        bias_one = torch.ones((self.batch_size, len(sentence[0]), 1)).to(device)
        Dependent_hidden = torch.cat((Dependent_hidden, Variable(bias_one)), 2)
        bias_one = torch.ones((self.batch_size, 1)).to(device)
        Head_hidden = torch.cat((Head_hidden, Variable(bias_one)), 1)
        left_part = torch.mm(Dependent_hidden.view(self.batch_size * len(sentence[0]), -1), self.W_R_SRL_FB)
        left_part = left_part.view(self.batch_size, len(sentence[0]) * self.tagset_size, -1)
        Head_hidden = Head_hidden.view(self.batch_size, -1, 1)
        tag_space = torch.bmm(left_part, Head_hidden).view(self.batch_size, len(sentence[0]), self.tagset_size)
        tag_space = tag_space.view(self.batch_size, len(sentence[0]), -1)
        dep_tag_space = tag_space
        DEPprobs_student = F.log_softmax(dep_tag_space, dim=2)
        DEP_FB_loss = unlabeled_loss_function(DEPprobs_student, TagProbs_use_softmax)

        ## Dependency Extractor BF
        predicate_embeds = hidden_backward[np.arange(0, self.batch_size), target_idx_in]
        Head_hidden = self.head_dropout_unlabeled_BF(F.relu(self.hidLayerFOH_SRL_BF(predicate_embeds)))
        Dependent_hidden = self.dep_dropout_unlabeled_BF(F.relu(self.hidLayerFOM_SRL_BF(hidden_forward)))
        bias_one = torch.ones((self.batch_size, len(sentence[0]), 1)).to(device)
        Dependent_hidden = torch.cat((Dependent_hidden, Variable(bias_one)), 2)
        bias_one = torch.ones((self.batch_size, 1)).to(device)
        Head_hidden = torch.cat((Head_hidden, Variable(bias_one)), 1)
        left_part = torch.mm(Dependent_hidden.view(self.batch_size * len(sentence[0]), -1), self.W_R_SRL_BF)
        left_part = left_part.view(self.batch_size, len(sentence[0]) * self.tagset_size, -1)
        Head_hidden = Head_hidden.view(self.batch_size, -1, 1)
        tag_space = torch.bmm(left_part, Head_hidden).view(self.batch_size, len(sentence[0]), self.tagset_size)
        tag_space = tag_space.view(self.batch_size, len(sentence[0]), -1)
        dep_tag_space = tag_space
        DEPprobs_student = F.log_softmax(dep_tag_space, dim=2)
        DEP_BF_loss = unlabeled_loss_function(DEPprobs_student, TagProbs_use_softmax)

        DEP_Semi_loss = DEP_FF_loss + DEP_BB_loss + DEP_BF_loss + DEP_FB_loss


        DEP_Semi_loss = torch.sum(DEP_Semi_loss, dim=2) # / Entroy_Weights
        loss_mask = np.ones(DEP_Semi_loss.size(), dtype='float32')
        for i in range(self.batch_size):
            if target_idx_in[i] == -1:
                sample_nums -= lengths[i]
            for j in range(len(sentence[0])):
                if j >= lengths[i] or target_idx_in[i] == -1:
                    loss_mask[i][j] = 0.0
        loss_mask = torch.from_numpy(loss_mask).to(device)

        DEP_Semi_loss = DEP_Semi_loss * loss_mask
        DEP_Semi_loss = torch.sum(DEP_Semi_loss)
        if sample_nums == 0:
            log("shit")
            sample_nums = 1
            log(DEP_Semi_loss)
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

        """
        SA_learning
        """
        embeds_DEP = self.word_embeddings_DEP(sentence)
        fixed_embeds_DEP = self.word_fixed_embeddings_DEP(p_sentence)
        fixed_embeds_DEP = fixed_embeds_DEP.view(self.batch_size, len(sentence[0]), self.word_emb_dim)
        embeds_forDEP = torch.cat((embeds_DEP, fixed_embeds_DEP), 2)
        embeds_forDEP = self.DEP_input_dropout_unlabeled(embeds_forDEP)

        # first layer
        embeds_sort, lengths_sort, unsort_idx = self.sort_batch(embeds_forDEP, lengths)
        embeds_sort = rnn.pack_padded_sequence(embeds_sort, lengths_sort, batch_first=True)
        hidden_states, self.SA_primary_hidden = self.BiLSTM_SA_primary(embeds_sort, self.SA_primary_hidden)
        hidden_states, lens = rnn.pad_packed_sequence(hidden_states, batch_first=True)
        hidden_states_0 = hidden_states[unsort_idx]


        # second_layer
        embeds_sort, lengths_sort, unsort_idx = self.sort_batch(hidden_states_0, lengths)
        embeds_sort = rnn.pack_padded_sequence(embeds_sort, lengths_sort, batch_first=True)
        # hidden states [time_steps * batch_size * hidden_units]
        hidden_states, self.SA_high_hidden = self.BiLSTM_SA_high(embeds_sort, self.SA_high_hidden)
        # it seems that hidden states is already batch first, we don't need swap the dims
        # hidden_states = hidden_states.permute(1, 2, 0).contiguous().view(self.batch_size, -1, )
        hidden_states, lens = rnn.pad_packed_sequence(hidden_states, batch_first=True)
        # hidden_states = hidden_states.transpose(0, 1)
        hidden_states_1 = hidden_states[unsort_idx]
        hidden_states_1 = self.DEP_hidden_state_dropout_2_unlabeled(hidden_states_1)

        tag_space = self.PI_MLP(hidden_states_1).view(
              self.batch_size, len(sentence[0]), -1)
        Predicate_identification_space = F.softmax(tag_space, dim=2)
        Predicate_probs = Predicate_identification_space.cpu().data.numpy()
        Predicate_idx_batch = [-1] * self.batch_size
        for i in range(self.batch_size):
            candidate_set = []
            probs_set = []
            for j in range(len(sentence[0])):
                probs_set.append(Predicate_probs[i][j][1])
                if j >= lengths[i]:
                    break
                if Predicate_probs[i][j][1] > 2*Predicate_probs[i][j][0]:
                    candidate_set.append(j)
            if len(candidate_set) > 0:
                index = random.sample(candidate_set, 1)
                Predicate_idx_batch[i] = index[0]
            else:
                Predicate_idx_batch[i] = np.argmax(probs_set)


        #log(Predicate_idx_batch)

        unlabeled_region_mark = np.zeros(sentence.size(), dtype='int64')
        for i in range(30):
            unlabeled_region_mark[i][Predicate_idx_batch[i]] = 1


        unlabeled_region_mark_in = torch.from_numpy(unlabeled_region_mark).to(device)
        unlabeled_region_mark_embeds = self.region_embeddings(unlabeled_region_mark_in)


        tag_space = self.POS_MLP(hidden_states_1).view(
              self.batch_size, len(sentence[0]), -1)
        POS_label = torch.argmax(tag_space, dim=2)

        ######################################################

        """
        SRL_learning
        """
        #########################################################
        embeds_SRL = self.word_embeddings_SRL(sentence)
        fixed_embeds_SRL = self.word_fixed_embeddings(p_sentence)
        pos_embeds = self.pos_embeddings(POS_label)
        embeds_forSRL = torch.cat((embeds_SRL, fixed_embeds_SRL, unlabeled_region_mark_embeds), 2)
        embeds_forSRL = self.SRL_input_dropout_unlabeled(embeds_forSRL)

        # first layer
        embeds_sort, lengths_sort, unsort_idx = self.sort_batch(embeds_forSRL, lengths)
        embeds_sort = rnn.pack_padded_sequence(embeds_sort, lengths_sort, batch_first=True)
        hidden_states, self.SRL_primary_hidden = self.BiLSTM_SRL_primary(embeds_sort, self.SRL_primary_hidden)
        hidden_states, lens = rnn.pad_packed_sequence(hidden_states, batch_first=True)
        hidden_states_0 = hidden_states[unsort_idx]

        hidden_forward, hidden_backward = hidden_states_0.split(self.hidden_dim, 2)

        # second_layer
        embeds_sort, lengths_sort, unsort_idx = self.sort_batch(hidden_states_0, lengths)
        embeds_sort = rnn.pack_padded_sequence(embeds_sort, lengths_sort, batch_first=True)
        # hidden states [time_steps * batch_size * hidden_units]
        hidden_states, self.SRL_high_hidden = self.BiLSTM_SRL_high(embeds_sort, self.SRL_high_hidden)
        # it seems that hidden states is already batch first, we don't need swap the dims
        # hidden_states = hidden_states.permute(1, 2, 0).contiguous().view(self.batch_size, -1, )
        hidden_states, lens = rnn.pad_packed_sequence(hidden_states, batch_first=True)
        # hidden_states = hidden_states.transpose(0, 1)
        hidden_states_1 = hidden_states[unsort_idx]
        #hidden_states_1 = self.hidden_state_dropout_2_unlabeled(hidden_states_1)

        #########################################3
        predicate_embeds = hidden_states_1[np.arange(0, hidden_states_1.size()[0]), Predicate_idx_batch]
        Head_hidden = self.head_dropout_unlabeled(F.relu(self.hidLayerFOH_SRL(predicate_embeds)))
        Dependent_hidden = self.dep_dropout_unlabeled(F.relu(self.hidLayerFOM_SRL(hidden_states_1)))
        bias_one = torch.ones((self.batch_size, len(sentence[0]), 1)).to(device)
        Dependent_hidden = torch.cat((Dependent_hidden, Variable(bias_one)), 2)
        bias_one = torch.ones((self.batch_size, 1)).to(device)
        Head_hidden = torch.cat((Head_hidden, Variable(bias_one)), 1)
        left_part = torch.mm(Dependent_hidden.view(self.batch_size * len(sentence[0]), -1), self.W_R_SRL)
        left_part = left_part.view(self.batch_size, len(sentence[0]) * self.tagset_size, -1)
        Head_hidden = Head_hidden.view(self.batch_size, -1, 1)
        tag_space = torch.bmm(left_part, Head_hidden).view(self.batch_size, len(sentence[0]), self.tagset_size)
        tag_space = tag_space.view(self.batch_size * len(sentence[0]), -1)

        TagProbs_use = tag_space.view(self.batch_size, len(sentence[0]), -1).detach()
        CVT_SRL_Loss = self.Semi_SRL_Loss(hidden_forward, hidden_backward, TagProbs_use, sentence, lengths, Predicate_idx_batch)

        return CVT_SRL_Loss



    def forward(self, sentence, p_sentence, pos_tags, sent_mask, lengths, target_idx_in, region_marks,
                local_roles_voc, frames, local_roles_mask,
                sent_pred_lemmas_idx, dep_tags, dep_heads, targets, gold_pos_tag, specific_dep_relations,
                Chars=None, Predicate_indicator = None, test=False,
                unlabeled_sentence=None, p_unlabeled_sentence=None, unlabeled_sent_mask=None, unlabeled_lengths=None, cvt_train=False):

        if cvt_train:
            CVT_SRL_Loss = self.CVT_train(unlabeled_sentence, p_unlabeled_sentence, unlabeled_sent_mask, unlabeled_lengths)
            return CVT_SRL_Loss

        """
        SA_learning
        """
        embeds_DEP = self.word_embeddings_DEP(sentence)
        fixed_embeds_DEP = self.word_fixed_embeddings_DEP(p_sentence)
        fixed_embeds_DEP = fixed_embeds_DEP.view(self.batch_size, len(sentence[0]), self.word_emb_dim)
        embeds_forDEP = torch.cat((embeds_DEP, fixed_embeds_DEP), 2)
        embeds_forDEP = self.DEP_input_dropout(embeds_forDEP)

        # first layer
        embeds_sort, lengths_sort, unsort_idx = self.sort_batch(embeds_forDEP, lengths)
        embeds_sort = rnn.pack_padded_sequence(embeds_sort, lengths_sort, batch_first=True)
        hidden_states, self.SA_primary_hidden = self.BiLSTM_SA_primary(embeds_sort, self.SA_primary_hidden)
        hidden_states, lens = rnn.pad_packed_sequence(hidden_states, batch_first=True)
        hidden_states_0 = hidden_states[unsort_idx]

        # second_layer
        embeds_sort, lengths_sort, unsort_idx = self.sort_batch(hidden_states_0, lengths)
        embeds_sort = rnn.pack_padded_sequence(embeds_sort, lengths_sort, batch_first=True)
        # hidden states [time_steps * batch_size * hidden_units]
        hidden_states, self.SA_high_hidden = self.BiLSTM_SA_high(embeds_sort, self.SA_high_hidden)
        # it seems that hidden states is already batch first, we don't need swap the dims
        # hidden_states = hidden_states.permute(1, 2, 0).contiguous().view(self.batch_size, -1, )
        hidden_states, lens = rnn.pad_packed_sequence(hidden_states, batch_first=True)
        # hidden_states = hidden_states.transpose(0, 1)
        hidden_states_1 = hidden_states[unsort_idx]
        hidden_states_1 = self.DEP_hidden_state_dropout_2(hidden_states_1)

        tag_space = self.PI_MLP(hidden_states_1).view(
            len(sentence[0]) * self.batch_size, -1)
        PI_label = np.argmax(tag_space.cpu().data.numpy(), axis=1)
        loss_function = nn.CrossEntropyLoss(ignore_index=-1)
        PI_loss = loss_function(tag_space, Predicate_indicator.view(-1))


        tag_space = self.POS_MLP(hidden_states_1).view(
            len(sentence[0]) * self.batch_size, -1)
        POS_label = np.argmax(tag_space.cpu().data.numpy(), axis=1)
        loss_function = nn.CrossEntropyLoss(ignore_index=0)
        POS_loss = loss_function(tag_space, gold_pos_tag.view(-1))
        ######################################################

        """
        SRL_learning
        """
        #########################################################
        embeds_SRL = self.word_embeddings_SRL(sentence)
        fixed_embeds_SRL = self.word_fixed_embeddings(p_sentence)
        #pos_embeds = self.pos_embeddings(pos_tags)
        #sent_pred_lemmas_embeds = self.p_lemma_embeddings(sent_pred_lemmas_idx)
        region_marks = self.region_embeddings(region_marks).view(self.batch_size, len(sentence[0]), 16)
        embeds_forSRL = torch.cat((embeds_SRL, fixed_embeds_SRL, region_marks), 2)
        embeds_forSRL = self.SRL_input_dropout(embeds_forSRL)

        # first layer
        embeds_sort, lengths_sort, unsort_idx = self.sort_batch(embeds_forSRL, lengths)
        embeds_sort = rnn.pack_padded_sequence(embeds_sort, lengths_sort, batch_first=True)
        hidden_states, self.SRL_primary_hidden = self.BiLSTM_SRL_primary(embeds_sort, self.SRL_primary_hidden )
        hidden_states, lens = rnn.pad_packed_sequence(hidden_states, batch_first=True)
        hidden_states_0 = hidden_states[unsort_idx]

        # second_layer
        embeds_sort, lengths_sort, unsort_idx = self.sort_batch(hidden_states_0, lengths)
        embeds_sort = rnn.pack_padded_sequence(embeds_sort, lengths_sort, batch_first=True)
        # hidden states [time_steps * batch_size * hidden_units]
        hidden_states, self.SRL_high_hidden = self.BiLSTM_SRL_high(embeds_sort, self.SRL_high_hidden)
        # it seems that hidden states is already batch first, we don't need swap the dims
        # hidden_states = hidden_states.permute(1, 2, 0).contiguous().view(self.batch_size, -1, )
        hidden_states, lens = rnn.pad_packed_sequence(hidden_states, batch_first=True)
        # hidden_states = hidden_states.transpose(0, 1)
        hidden_states_1 = hidden_states[unsort_idx]
        #hidden_states_1 = self.hidden_state_dropout_2(hidden_states_1)

        #########################################3
        predicate_embeds = hidden_states_1[np.arange(0, hidden_states_1.size()[0]), target_idx_in]
        Head_hidden = self.head_dropout(F.relu(self.hidLayerFOH_SRL(predicate_embeds)))
        Dependent_hidden = self.dep_dropout(F.relu(self.hidLayerFOM_SRL(hidden_states_1)))
        bias_one = torch.ones((self.batch_size, len(sentence[0]), 1)).to(device)
        Dependent_hidden = torch.cat((Dependent_hidden, Variable(bias_one)), 2)
        bias_one = torch.ones((self.batch_size, 1)).to(device)
        Head_hidden = torch.cat((Head_hidden, Variable(bias_one)), 1)
        left_part = torch.mm(Dependent_hidden.view(self.batch_size * len(sentence[0]), -1), self.W_R_SRL)
        left_part = left_part.view(self.batch_size, len(sentence[0])*self.tagset_size, -1)
        Head_hidden = Head_hidden.view(self.batch_size, -1, 1)
        tag_space = torch.bmm(left_part, Head_hidden).view(self.batch_size, len(sentence[0]), self.tagset_size)
        tag_space = tag_space.view(self.batch_size * len(sentence[0]), -1)
        SRLprobs = F.softmax(tag_space, dim=1)
        loss_function = nn.CrossEntropyLoss(ignore_index=-1)
        SRLloss = loss_function(tag_space, targets.view(-1))
        ##########################################

        Link_right, Link_all, \
        POS_right, POS_all, PI_right, PI_nonull_preidcates, PI_nonull_truth = 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1

        for a, b in zip(PI_label, Predicate_indicator.view(-1).cpu().data.numpy()):
            if b == -1:
                continue
            if a == 1:
                PI_nonull_preidcates += 1
            if b == 1:
                PI_nonull_truth += 1
                if a == b:
                    PI_right += 1

        for a, b in zip(POS_label, gold_pos_tag.view(-1).cpu().data.numpy()):
            if b == 0:
                continue
            POS_all += 1
            if a == b:
                POS_right += 1



        Tag_DEPloss = 0
        Link_DEPloss = 0

        return SRLloss, Link_DEPloss, Tag_DEPloss, POS_loss, PI_loss, SRLprobs, Link_right, Link_all, \
               POS_right, POS_all, PI_right, PI_nonull_preidcates, PI_nonull_truth \

    @staticmethod
    def sort_batch(x, l):
        l = torch.from_numpy(np.asarray(l))
        l_sorted, sidx = l.sort(0, descending=True)
        x_sorted = x[sidx]
        _, unsort_idx = sidx.sort()
        return x_sorted, l_sorted, unsort_idx