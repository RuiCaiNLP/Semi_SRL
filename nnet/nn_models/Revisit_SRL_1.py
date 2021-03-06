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

class BiLSTMTagger(nn.Module):

    #def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
    def __init__(self, hps, *_):
        super(BiLSTMTagger, self).__init__()

        batch_size = hps['batch_size']
        lstm_hidden_dim = hps['sent_hdim']
        sent_embedding_dim_DEP = 2*hps['sent_edim'] + 16
        sent_embedding_dim_SRL = 2 * hps['sent_edim'] + 16
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
        self.word_embeddings_PI = nn.Embedding(vocab_size, hps['sent_edim'])

        self.pos_embeddings = nn.Embedding(self.pos_size, hps['pos_edim'])
        self.pos_embeddings_DEP = nn.Embedding(self.pos_size, hps['pos_edim'])
        self.p_lemma_embeddings = nn.Embedding(self.frameset_size, hps['sent_edim'])
        self.dep_embeddings = nn.Embedding(self.dep_size, self.pos_size)
        self.region_embeddings = nn.Embedding(2, 16)
        #self.lr_dep_embeddings = nn.Embedding(self.lr_dep_size, hps[])

        self.word_fixed_embeddings = nn.Embedding(vocab_size, hps['sent_edim'])
        self.word_fixed_embeddings.weight.data.copy_(torch.from_numpy(hps['word_embeddings']))

        self.word_fixed_embeddings_DEP = nn.Embedding(vocab_size, hps['sent_edim'])
        self.word_fixed_embeddings_DEP.weight.data.copy_(torch.from_numpy(hps['word_embeddings']))

        self.word_fixed_embeddings_PI = nn.Embedding(vocab_size, hps['sent_edim'])
        self.word_fixed_embeddings_PI.weight.data.copy_(torch.from_numpy(hps['word_embeddings']))


        self.role_embeddings = nn.Embedding(self.tagset_size, role_embedding_dim)
        self.frame_embeddings = nn.Embedding(self.frameset_size, frame_embedding_dim)


        self.hidden2tag = nn.Linear(4*lstm_hidden_dim, 2*lstm_hidden_dim)
        self.MLP = nn.Linear(2*lstm_hidden_dim, self.specific_dep_size)
        self.tag2hidden = nn.Linear(self.specific_dep_size, self.pos_size)

        self.Head_Proj = nn.Linear(4 * lstm_hidden_dim, lstm_hidden_dim)
        self.W_R = nn.Parameter(torch.rand(lstm_hidden_dim, self.dep_size * lstm_hidden_dim))
        self.W_share = nn.Parameter(torch.rand(lstm_hidden_dim, self.dep_size * lstm_hidden_dim))
        self.Dep_Proj = nn.Linear(4 * lstm_hidden_dim, lstm_hidden_dim)

        self.MLP_identification = nn.Linear(4*lstm_hidden_dim, 2*lstm_hidden_dim)
        self.Idenficiation = nn.Linear(2*lstm_hidden_dim, 3)




        self.MLP_classifier_1 = nn.Linear(400, 400)
        self.MLP_classifier_0 = nn.Linear(400, self.tagset_size)

        self.elmo_emb_size = 200
        self.elmo_mlp_word = nn.Sequential(nn.Linear(1024, self.elmo_emb_size), nn.ReLU())
        self.elmo_word = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.elmo_gamma_word = nn.Parameter(torch.ones(1))

        self.elmo_mlp = nn.Sequential(nn.Linear(2 * lstm_hidden_dim, self.elmo_emb_size), nn.ReLU())
        self.elmo_w = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.elmo_gamma = nn.Parameter(torch.ones(1))

        self.SRL_input_dropout = nn.Dropout(p=0.3)
        self.DEP_input_dropout = nn.Dropout(p=0.3)
        self.PI_input_dropout = nn.Dropout(p=0.3)
        self.hidden_state_dropout_0 = nn.Dropout(p=0.3)
        self.hidden_state_dropout_1 = nn.Dropout(p=0.3)
        self.hidden_state_dropout_DEP = nn.Dropout(p=0.3)
        self.hidden_state_dropout_SRL = nn.Dropout(p=0.3)
        self.dropout_1_DEP = nn.Dropout(p=0.3)
        self.dropout_2_DEP = nn.Dropout(p=0.3)

        self.dropout_1 = nn.Dropout(p=0.3)
        self.dropout_2 = nn.Dropout(p=0.3)

        self.dropout_1_FF = nn.Dropout(p=0.3)
        self.dropout_2_FF = nn.Dropout(p=0.3)

        self.dropout_1_BB = nn.Dropout(p=0.3)
        self.dropout_2_BB = nn.Dropout(p=0.3)

        self.dropout_1_FB = nn.Dropout(p=0.3)
        self.dropout_2_FB = nn.Dropout(p=0.3)

        self.dropout_1_BF = nn.Dropout(p=0.3)
        self.dropout_2_BF = nn.Dropout(p=0.3)

        self.label_dropout_3 = nn.Dropout(p=0.3)
        self.label_dropout_4 = nn.Dropout(p=0.3)
        self.id_dropout = nn.Dropout(p=0.3)
        self.id_dropout_h = nn.Dropout(p=0.3)
        #self.use_dropout = nn.Dropout(p=0.2)


        # Predicate identification
        self.MLP_identification = nn.Linear(2 * lstm_hidden_dim, lstm_hidden_dim)
        self.Idenficiation = nn.Linear(lstm_hidden_dim, 3)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.num_layers = 1
        self.BiLSTM_0 = nn.LSTM(input_size=sent_embedding_dim_DEP , hidden_size=lstm_hidden_dim, batch_first=True,
                              bidirectional=True, num_layers=self.num_layers)

        init.orthogonal_(self.BiLSTM_0.all_weights[0][0])
        init.orthogonal_(self.BiLSTM_0.all_weights[0][1])
        init.orthogonal_(self.BiLSTM_0.all_weights[1][0])
        init.orthogonal_(self.BiLSTM_0.all_weights[1][1])

        self.num_layers = 1
        self.BiLSTM_1 = nn.LSTM(input_size=sent_embedding_dim_SRL, hidden_size=lstm_hidden_dim, batch_first=True,
                                bidirectional=True, num_layers=self.num_layers)

        init.orthogonal_(self.BiLSTM_1.all_weights[0][0])
        init.orthogonal_(self.BiLSTM_1.all_weights[0][1])
        init.orthogonal_(self.BiLSTM_1.all_weights[1][0])
        init.orthogonal_(self.BiLSTM_1.all_weights[1][1])

        self.num_layers = 2
        self.BiLSTM_SRL = nn.LSTM(input_size=2*lstm_hidden_dim, hidden_size=lstm_hidden_dim, batch_first=True,
                                  bidirectional=True, num_layers=self.num_layers)

        init.orthogonal_(self.BiLSTM_SRL.all_weights[0][0])
        init.orthogonal_(self.BiLSTM_SRL.all_weights[0][1])
        init.orthogonal_(self.BiLSTM_SRL.all_weights[1][0])
        init.orthogonal_(self.BiLSTM_SRL.all_weights[1][1])

        self.num_layers = 1
        self.BiLSTM_1_2nd = nn.LSTM(input_size=sent_embedding_dim_SRL, hidden_size=lstm_hidden_dim, batch_first=True,
                                bidirectional=True, num_layers=self.num_layers)

        init.orthogonal_(self.BiLSTM_1.all_weights[0][0])
        init.orthogonal_(self.BiLSTM_1.all_weights[0][1])
        init.orthogonal_(self.BiLSTM_1.all_weights[1][0])
        init.orthogonal_(self.BiLSTM_1.all_weights[1][1])

        self.num_layers = 2
        self.BiLSTM_SRL_2nd = nn.LSTM(input_size=2 * lstm_hidden_dim, hidden_size=lstm_hidden_dim, batch_first=True,
                                  bidirectional=True, num_layers=self.num_layers)

        init.orthogonal_(self.BiLSTM_SRL.all_weights[0][0])
        init.orthogonal_(self.BiLSTM_SRL.all_weights[0][1])
        init.orthogonal_(self.BiLSTM_SRL.all_weights[1][0])
        init.orthogonal_(self.BiLSTM_SRL.all_weights[1][1])

        self.num_layers = 1
        self.BiLSTM_DEP = nn.LSTM(input_size=2 * lstm_hidden_dim, hidden_size=lstm_hidden_dim, batch_first=True,
                                  bidirectional=True, num_layers=self.num_layers)

        init.orthogonal_(self.BiLSTM_DEP.all_weights[0][0])
        init.orthogonal_(self.BiLSTM_DEP.all_weights[0][1])
        init.orthogonal_(self.BiLSTM_DEP.all_weights[1][0])
        init.orthogonal_(self.BiLSTM_DEP.all_weights[1][1])


        self.num_layers = 2
        self.BiLSTM_PI = nn.LSTM(input_size=sent_embedding_dim_SRL - 16, hidden_size=lstm_hidden_dim, batch_first=True,
                                    bidirectional=True, num_layers=self.num_layers)

        init.orthogonal_(self.BiLSTM_PI.all_weights[0][0])
        init.orthogonal_(self.BiLSTM_PI.all_weights[0][1])
        init.orthogonal_(self.BiLSTM_PI.all_weights[1][0])
        init.orthogonal_(self.BiLSTM_PI.all_weights[1][1])

        self.Non_Predicate_Proj_DEP = nn.Linear(2 * lstm_hidden_dim, lstm_hidden_dim)
        self.Predicate_Proj_DEP = nn.Linear(2 * lstm_hidden_dim, lstm_hidden_dim)
        self.W_R_DEP = nn.Parameter(torch.rand(lstm_hidden_dim + 1, self.dep_size * lstm_hidden_dim))

        self.Non_Predicate_Proj = nn.Linear(2 * lstm_hidden_dim, lstm_hidden_dim)
        self.Predicate_Proj = nn.Linear(2 * lstm_hidden_dim, lstm_hidden_dim)
        self.W_R = nn.Parameter(torch.rand(lstm_hidden_dim + 1, self.tagset_size * (lstm_hidden_dim+1)))

        self.Non_Predicate_Proj_2nd = nn.Linear(2 * lstm_hidden_dim, lstm_hidden_dim)
        self.Predicate_Proj_2nd = nn.Linear(2 * lstm_hidden_dim, lstm_hidden_dim)
        self.W_R_2nd = nn.Parameter(torch.rand(lstm_hidden_dim + 1, self.tagset_size * (lstm_hidden_dim + 1)))


        # Dependency extractor: auxiliary FF
        self.Non_Predicate_Proj_FF = nn.Linear(lstm_hidden_dim, lstm_hidden_dim)
        self.Predicate_Proj_FF = nn.Linear(lstm_hidden_dim, lstm_hidden_dim)
        self.W_R_FF = nn.Parameter(torch.rand(lstm_hidden_dim + 1, self.tagset_size * (lstm_hidden_dim+1)))

        # Dependency extractor: auxiliary BB
        self.Non_Predicate_Proj_BB = nn.Linear(lstm_hidden_dim, lstm_hidden_dim)
        self.Predicate_Proj_BB = nn.Linear(lstm_hidden_dim, lstm_hidden_dim)
        self.W_R_BB = nn.Parameter(torch.rand(lstm_hidden_dim + 1, self.tagset_size * (lstm_hidden_dim+1)))

        # Dependency extractor: auxiliary BF
        self.Non_Predicate_Proj_BF = nn.Linear(lstm_hidden_dim, lstm_hidden_dim)
        self.Predicate_Proj_BF = nn.Linear(lstm_hidden_dim, lstm_hidden_dim)
        self.W_R_BF = nn.Parameter(torch.rand(lstm_hidden_dim + 1, self.tagset_size * (lstm_hidden_dim+1)))

        # Dependency extractor: auxiliary FB
        self.Non_Predicate_Proj_FB = nn.Linear(lstm_hidden_dim, lstm_hidden_dim)
        self.Predicate_Proj_FB = nn.Linear(lstm_hidden_dim, lstm_hidden_dim)
        self.W_R_FB = nn.Parameter(torch.rand(lstm_hidden_dim + 1, self.tagset_size * (lstm_hidden_dim+1)))

        # Init hidden state
        self.hidden = self.init_hidden_spe()
        self.hidden_2 = self.init_hidden_spe()
        self.hidden_3 = self.init_hidden_spe()
        self.hidden_DEP_base = self.init_hidden_spe()
        self.hidden_DEP = self.init_hidden_spe()
        self.hidden_SRL_base = self.init_hidden_spe()
        self.hidden_SRL = self.init_hidden_SRL()
        self.hidden_PI = self.init_hidden_share()

    def init_hidden_SRL(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        #return (Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
        #        Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))
        return (torch.zeros(2 * 2, self.batch_size, self.hidden_dim, requires_grad=False).to(device),
                torch.zeros(2 * 2, self.batch_size, self.hidden_dim, requires_grad=False).to(device))

    def init_hidden_share(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        #return (Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
        #        Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))
        return (torch.zeros(2 * 2, self.batch_size, self.hidden_dim, requires_grad=False).to(device),
                torch.zeros(2 * 2, self.batch_size, self.hidden_dim, requires_grad=False).to(device))

    def init_hidden_spe(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        #return (Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
        #        Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))
        return (torch.zeros(1 * 2, self.batch_size, self.hidden_dim, requires_grad=False).to(device),
                torch.zeros(1 * 2, self.batch_size, self.hidden_dim, requires_grad=False).to(device))



    def Predicate_Id(self, sentence, p_sentence, lengths):
        # contruct input for shared BiLSTM Encoder
        embeds_PI = self.word_embeddings_DEP(sentence)
        embeds_PI = embeds_PI.view(self.batch_size, len(sentence[0]), self.word_emb_dim)
        # sharing pretrained word_embeds
        fixed_embeds_PI = self.word_fixed_embeddings_PI(p_sentence)
        fixed_embeds_PI = fixed_embeds_PI.view(self.batch_size, len(sentence[0]), self.word_emb_dim)

        embeds_forPI = torch.cat((embeds_PI, fixed_embeds_PI), 2)
        embeds_forPI = self.PI_input_dropout(embeds_forPI)

        embeds_sort, lengths_sort, unsort_idx = self.sort_batch(embeds_forPI, lengths)
        embeds_sort = rnn.pack_padded_sequence(embeds_sort, lengths_sort.cpu().numpy(), batch_first=True)
        # hidden states [time_steps * batch_size * hidden_units]
        self.hidden_PI = self.init_hidden_share()
        hidden_states, self.hidden_PI = self.BiLSTM_PI(embeds_sort, self.hidden_PI)
        # it seems that hidden states is already batch first, we don't need swap the dims
        # hidden_states = hidden_states.permute(1, 2, 0).contiguous().view(self.batch_size, -1, )
        hidden_states, lens = rnn.pad_packed_sequence(hidden_states, batch_first=True)
        # hidden_states = hidden_states.transpose(0, 1)
        hidden_states = hidden_states[unsort_idx]
        hidden_states = self.id_dropout(hidden_states)

        Predicate_identification = self.Idenficiation(
            self.id_dropout_h(F.relu(self.MLP_identification(hidden_states))))
        Predicate_identification_space = Predicate_identification.view(
            len(sentence[0]) * self.batch_size, -1)

        return Predicate_identification_space

    def forward(self, sentence, p_sentence,  pos_tags, lengths, target_idx_in, region_marks,
                local_roles_voc, frames, local_roles_mask,
                sent_pred_lemmas_idx,  dep_tags,  dep_heads, targets, P_identification, all_l_ids,
                Predicate_link, Predicate_Labels_nd, Predicate_Labels, Chars_in,
                unlabeled_sentence=None, p_unlabeled_sentence=None, unlabeled_lengths=None,test=False, cvt_train=False):


        if cvt_train:
            CVT_SRL_Loss = self.CVT_train(unlabeled_sentence, p_unlabeled_sentence, unlabeled_lengths)
            return CVT_SRL_Loss
        """
        perform predicate Identificaiton first
        """
        Predicate_Identification_Space = self.Predicate_Id(sentence, p_sentence, lengths)

        # +++++++++++++++++++++++
        wrong_l_nums = 0.0
        all_l_nums = 0.0

        right_noNull_predict = 0.0
        noNull_predict = 0.0
        noNUll_truth = 0.0

        PI_labels = np.argmax(Predicate_Identification_Space.cpu().data.numpy(), axis=1)
        for predict_l, gold_l in zip(PI_labels, P_identification.cpu().view(-1).data.numpy()):
            if predict_l > 1 and gold_l != 0:
                noNull_predict += 1
            if gold_l != 0:
                all_l_nums += 1
                if gold_l != 1:
                    noNUll_truth += 1
                    if gold_l == predict_l:
                        right_noNull_predict += 1
            if predict_l != gold_l and gold_l != 0:
                wrong_l_nums += 1

        """
        construct DEP_input
        """

        fixed_embeds = self.word_fixed_embeddings(p_sentence)
        fixed_embeds = fixed_embeds.view(self.batch_size, len(sentence[0]), self.word_emb_dim)
        embeds_SRL = self.word_embeddings_SRL(sentence)
        embeds_SRL = embeds_SRL.view(self.batch_size, len(sentence[0]), self.word_emb_dim)
        region_marks = self.region_embeddings(region_marks).view(self.batch_size, len(sentence[0]), 16)

        SRL_hidden_states = torch.cat((embeds_SRL, fixed_embeds,
                                       region_marks), 2)
        SRL_hidden_states = self.SRL_input_dropout(SRL_hidden_states)

        # SRL layer
        embeds_sort, lengths_sort, unsort_idx = self.sort_batch(SRL_hidden_states, lengths)
        embeds_sort = rnn.pack_padded_sequence(embeds_sort, lengths_sort, batch_first=True)
        # hidden states [time_steps * batch_size * hidden_units]
        hidden_states, self.hidden_SRL_base = self.BiLSTM_1(embeds_sort, self.hidden_SRL_base)
        # it seems that hidden states is already batch first, we don't need swap the dims
        # hidden_states = hidden_states.permute(1, 2, 0).contiguous().view(self.batch_size, -1, )
        hidden_states, lens = rnn.pad_packed_sequence(hidden_states, batch_first=True)
        # hidden_states = hidden_states.transpose(0, 1)
        hidden_states_0 = hidden_states[unsort_idx]

        embeds_sort, lengths_sort, unsort_idx = self.sort_batch(hidden_states_0, lengths)
        embeds_sort = rnn.pack_padded_sequence(embeds_sort, lengths_sort.cpu().numpy(), batch_first=True)
        # hidden states [time_steps * batch_size * hidden_units]
        hidden_states, self.hidden_SRL = self.BiLSTM_SRL(embeds_sort, self.hidden_SRL)
        # it seems that hidden states is already batch first, we don't need swap the dims
        # hidden_states = hidden_states.permute(1, 2, 0).contiguous().view(self.batch_size, -1, )
        hidden_states, lens = rnn.pad_packed_sequence(hidden_states, batch_first=True)
        # hidden_states = hidden_states.transpose(0, 1)
        hidden_states = hidden_states[unsort_idx]
        hidden_states = self.hidden_state_dropout_SRL(hidden_states)

        # B * H
        hidden_states_3 = hidden_states
        hidden_states_word = self.dropout_1(F.relu(self.Non_Predicate_Proj(hidden_states_3)))
        predicate_embeds = hidden_states_3[np.arange(0, hidden_states_3.size()[0]), target_idx_in]
        hidden_states_predicate = self.dropout_2(F.relu(self.Predicate_Proj(predicate_embeds)))

        bias_one = torch.ones((self.batch_size, len(sentence[0]), 1)).to(device)
        hidden_states_word = torch.cat((hidden_states_word, Variable(bias_one)), 2)

        bias_one = torch.ones((self.batch_size, 1)).to(device)
        hidden_states_predicate = torch.cat((hidden_states_predicate, Variable(bias_one)), 1)

        left_part = torch.mm(hidden_states_word.view(self.batch_size * len(sentence[0]), -1), self.W_R)
        left_part = left_part.view(self.batch_size, len(sentence[0]) * self.tagset_size, -1)
        hidden_states_predicate = hidden_states_predicate.view(self.batch_size, -1, 1)
        tag_space = torch.bmm(left_part, hidden_states_predicate).view(
            len(sentence[0]) * self.batch_size, -1)

        # 2nd  SRL layer

        embeds_sort, lengths_sort, unsort_idx = self.sort_batch(SRL_hidden_states, lengths)
        embeds_sort = rnn.pack_padded_sequence(embeds_sort, lengths_sort, batch_first=True)
        # hidden states [time_steps * batch_size * hidden_units]
        hidden_states, self.hidden_SRL_base = self.BiLSTM_1_2nd(embeds_sort, self.hidden_SRL_base)
        # it seems that hidden states is already batch first, we don't need swap the dims
        # hidden_states = hidden_states.permute(1, 2, 0).contiguous().view(self.batch_size, -1, )
        hidden_states, lens = rnn.pad_packed_sequence(hidden_states, batch_first=True)
        # hidden_states = hidden_states.transpose(0, 1)
        hidden_states_0 = hidden_states[unsort_idx]

        embeds_sort, lengths_sort, unsort_idx = self.sort_batch(hidden_states_0, lengths)
        embeds_sort = rnn.pack_padded_sequence(embeds_sort, lengths_sort.cpu().numpy(), batch_first=True)
        # hidden states [time_steps * batch_size * hidden_units]
        hidden_states, self.hidden_SRL = self.BiLSTM_SRL_2nd(embeds_sort, self.hidden_SRL)
        # it seems that hidden states is already batch first, we don't need swap the dims
        # hidden_states = hidden_states.permute(1, 2, 0).contiguous().view(self.batch_size, -1, )
        hidden_states, lens = rnn.pad_packed_sequence(hidden_states, batch_first=True)
        # hidden_states = hidden_states.transpose(0, 1)
        hidden_states = hidden_states[unsort_idx]
        hidden_states = self.hidden_state_dropout_SRL(hidden_states)

        # B * H
        hidden_states_3 = hidden_states
        hidden_states_word = self.dropout_1(F.relu(self.Non_Predicate_Proj_2nd(hidden_states_3)))
        predicate_embeds = hidden_states_3[np.arange(0, hidden_states_3.size()[0]), target_idx_in]
        hidden_states_predicate = self.dropout_2(F.relu(self.Predicate_Proj_2nd(predicate_embeds)))

        bias_one = torch.ones((self.batch_size, len(sentence[0]), 1)).to(device)
        hidden_states_word = torch.cat((hidden_states_word, Variable(bias_one)), 2)

        bias_one = torch.ones((self.batch_size, 1)).to(device)
        hidden_states_predicate = torch.cat((hidden_states_predicate, Variable(bias_one)), 1)

        left_part = torch.mm(hidden_states_word.view(self.batch_size * len(sentence[0]), -1), self.W_R_2nd)
        left_part = left_part.view(self.batch_size, len(sentence[0]) * self.tagset_size, -1)
        hidden_states_predicate = hidden_states_predicate.view(self.batch_size, -1, 1)
        tag_space_2nd = torch.bmm(left_part, hidden_states_predicate).view(
            len(sentence[0]) * self.batch_size, -1)
        SRLprobs = F.softmax(tag_space_2nd, dim=1)



        loss_function = nn.CrossEntropyLoss(ignore_index=0)

        SRLloss = loss_function(tag_space, targets.view(-1))
        SRLloss_2nd = loss_function(tag_space_2nd, targets.view(-1))

        IDloss = loss_function(Predicate_Identification_Space, P_identification.view(-1))

        return SRLloss, SRLloss_2nd, IDloss, SRLprobs, wrong_l_nums, all_l_nums, wrong_l_nums, all_l_nums,  \
               right_noNull_predict, noNull_predict, noNUll_truth,\
               right_noNull_predict, noNull_predict, noNUll_truth

    @staticmethod
    def sort_batch(x, l):
        l = torch.from_numpy(np.asarray(l))
        l_sorted, sidx = l.sort(0, descending=True)
        x_sorted = x[sidx]
        _, unsort_idx = sidx.sort()
        return x_sorted, l_sorted, unsort_idx