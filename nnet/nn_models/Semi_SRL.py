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
        sent_embedding_dim_DEP = 2*hps['sent_edim']
        sent_embedding_dim_SRL = 2 * hps['sent_edim']  + 16
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

        self.SRL_input_dropout = nn.Dropout(p=0.3)
        self.DEP_input_dropout = nn.Dropout(p=0.3)
        self.hidden_state_dropout = nn.Dropout(p=0.3)
        self.label_dropout_1 = nn.Dropout(p=0.3)
        self.label_dropout_2 = nn.Dropout(p=0.3)
        self.label_dropout_3 = nn.Dropout(p=0.3)
        self.label_dropout_4 = nn.Dropout(p=0.3)
        self.id_dropout = nn.Dropout(p=0.3)
        #self.use_dropout = nn.Dropout(p=0.2)



        # The BiLSTM encoder
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        self.num_layers = 1
        self.word_embeddings_DEP = nn.Embedding(vocab_size, hps['sent_edim'])

        self.word_fixed_embeddings_DEP = nn.Embedding(vocab_size, hps['sent_edim'])
        self.word_fixed_embeddings_DEP.weight.data.copy_(torch.from_numpy(hps['word_embeddings']))

        self.BiLSTM_0 = nn.LSTM(input_size=sent_embedding_dim_DEP , hidden_size=lstm_hidden_dim, batch_first=True,
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



        # SRL: primary prediciton
        self.num_layers = 3
        self.word_embeddings_SRL = nn.Embedding(vocab_size, hps['sent_edim'])

        self.word_fixed_embeddings_SRL = nn.Embedding(vocab_size, hps['sent_edim'])
        self.word_fixed_embeddings_SRL.weight.data.copy_(torch.from_numpy(hps['word_embeddings']))

        self.dep_embeddings = nn.Embedding(self.dep_size, self.pos_size)
        self.region_embeddings = nn.Embedding(2, 16)
        self.elmo_emb_size = 200
        self.BiLSTM_SRL = nn.LSTM(input_size= sent_embedding_dim_SRL+ self.elmo_emb_size * 1 + 1 * self.pos_size, hidden_size=lstm_hidden_dim, batch_first=True,
                                    bidirectional=True, num_layers=self.num_layers)

        init.orthogonal_(self.BiLSTM_SRL.all_weights[0][0])
        init.orthogonal_(self.BiLSTM_SRL.all_weights[0][1])
        init.orthogonal_(self.BiLSTM_SRL.all_weights[1][0])
        init.orthogonal_(self.BiLSTM_SRL.all_weights[1][1])

        self.elmo_mlp = nn.Sequential(nn.Linear(2 * lstm_hidden_dim, self.elmo_emb_size), nn.ReLU())
        self.elmo_w = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.elmo_gamma = nn.Parameter(torch.ones(1))

        self.Head_Proj = nn.Linear(4 * lstm_hidden_dim, lstm_hidden_dim)
        self.W_R = nn.Parameter(torch.rand(lstm_hidden_dim, self.tagset_size * lstm_hidden_dim))
        self.W_share = nn.Parameter(torch.rand(lstm_hidden_dim, self.tagset_size * lstm_hidden_dim))
        self.Dep_Proj = nn.Linear(4 * lstm_hidden_dim, lstm_hidden_dim)

        self.Non_Predicate_Proj = nn.Linear(2 * lstm_hidden_dim, lstm_hidden_dim)
        self.Predicate_Proj = nn.Linear(2 * lstm_hidden_dim, lstm_hidden_dim)

           ## SRL: auxiliary prediction: fwd-fwd

        self.BiLSTM_SRL_fwd_fwd = nn.LSTM(input_size= 2*hps['sent_edim'] + 16 + lstm_hidden_dim,
                                  hidden_size=lstm_hidden_dim, batch_first=True,
                                  bidirectional=True, num_layers=self.num_layers)

        self.Head_Proj_FF = nn.Linear(4 * lstm_hidden_dim, lstm_hidden_dim)
        self.W_R_FF = nn.Parameter(torch.rand(lstm_hidden_dim, self.tagset_size * lstm_hidden_dim))
        self.W_share_FF = nn.Parameter(torch.rand(lstm_hidden_dim, self.tagset_size * lstm_hidden_dim))
        self.Dep_Proj_FF = nn.Linear(4 * lstm_hidden_dim, lstm_hidden_dim)

        self.Non_Predicate_Proj_FF = nn.Linear(2 * lstm_hidden_dim, lstm_hidden_dim)
        self.Predicate_Proj_FF = nn.Linear(2 * lstm_hidden_dim, lstm_hidden_dim)

            ## SRL: auxiliary prediction: fwd-fwd

        self.BiLSTM_SRL_bwd_bwd = nn.LSTM(input_size=2 * hps['sent_edim'] + 16 + lstm_hidden_dim,
                                          hidden_size=lstm_hidden_dim, batch_first=True,
                                          bidirectional=True, num_layers=self.num_layers)

        self.Head_Proj_BB = nn.Linear(4 * lstm_hidden_dim, lstm_hidden_dim)
        self.W_R_BB = nn.Parameter(torch.rand(lstm_hidden_dim, self.tagset_size * lstm_hidden_dim))
        self.W_share_BB = nn.Parameter(torch.rand(lstm_hidden_dim, self.tagset_size * lstm_hidden_dim))
        self.Dep_Proj_BB = nn.Linear(4 * lstm_hidden_dim, lstm_hidden_dim)

        self.Non_Predicate_Proj_BB = nn.Linear(2 * lstm_hidden_dim, lstm_hidden_dim)
        self.Predicate_Proj_BB = nn.Linear(2 * lstm_hidden_dim, lstm_hidden_dim)

        # Dependency extractor: primary preidition
        self.hidden2tag_1 = nn.Linear(4 * lstm_hidden_dim, 2 * lstm_hidden_dim)
        self.hidden2tag_2 = nn.Linear(4 * lstm_hidden_dim, 2 * lstm_hidden_dim)
        self.MLP_1 = nn.Linear(4 * lstm_hidden_dim, 2 * lstm_hidden_dim)
        self.MLP_2 = nn.Linear(2 * lstm_hidden_dim, self.specific_dep_size)
        self.tag2hidden = nn.Linear(self.specific_dep_size, self.pos_size)


        # Predicate identification
        self.MLP_identification = nn.Linear(4 * lstm_hidden_dim, 2 * lstm_hidden_dim)
        self.Idenficiation = nn.Linear(2 * lstm_hidden_dim, 3)




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
        #return (Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
        #        Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))
        return (torch.zeros(3 * 2, self.batch_size, self.hidden_dim, requires_grad=False).to(device),
                torch.zeros(3 * 2, self.batch_size, self.hidden_dim, requires_grad=False).to(device))

    def init_hidden_spe(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        #return (Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
        #        Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))
        return (torch.zeros(1 * 2, self.batch_size, self.hidden_dim, requires_grad=False).to(device),
                torch.zeros(1 * 2, self.batch_size, self.hidden_dim, requires_grad=False).to(device))



    def shared_BilSTMEncoder_foward(self, sentence, p_sentence, lengths):
        # contruct input for shared BiLSTM Encoder
        embeds_DEP = self.word_embeddings_DEP(sentence)
        embeds_DEP = embeds_DEP.view(self.batch_size, len(sentence[0]), self.word_emb_dim)
        # sharing pretrained word_embeds
        fixed_embeds_DEP = self.word_fixed_embeddings_DEP(p_sentence)
        fixed_embeds_DEP = fixed_embeds_DEP.view(self.batch_size, len(sentence[0]), self.word_emb_dim)

        embeds_forDEP = torch.cat((embeds_DEP, fixed_embeds_DEP), 2)
        embeds_forDEP = self.DEP_input_dropout(embeds_forDEP)

        # first layer
        embeds_sort, lengths_sort, unsort_idx = self.sort_batch(embeds_forDEP, lengths)
        embeds_sort = rnn.pack_padded_sequence(embeds_sort, lengths_sort, batch_first=True)
        # hidden states [time_steps * batch_size * hidden_units]
        self.hidden = self.init_hidden_spe()
        hidden_states, self.hidden = self.BiLSTM_0(embeds_sort, self.hidden)
        # it seems that hidden states is already batch first, we don't need swap the dims
        # hidden_states = hidden_states.permute(1, 2, 0).contiguous().view(self.batch_size, -1, )
        hidden_states, lens = rnn.pad_packed_sequence(hidden_states, batch_first=True)
        # hidden_states = hidden_states.transpose(0, 1)
        hidden_states_0 = hidden_states[unsort_idx]

        # second_layer
        embeds_sort, lengths_sort, unsort_idx = self.sort_batch(hidden_states_0, lengths)
        embeds_sort = rnn.pack_padded_sequence(embeds_sort, lengths_sort, batch_first=True)
        # hidden states [time_steps * batch_size * hidden_units]
        hidden_states, self.hidden_2 = self.BiLSTM_1(embeds_sort, self.hidden_2)
        # it seems that hidden states is already batch first, we don't need swap the dims
        # hidden_states = hidden_states.permute(1, 2, 0).contiguous().view(self.batch_size, -1, )
        hidden_states, lens = rnn.pad_packed_sequence(hidden_states, batch_first=True)
        # hidden_states = hidden_states.transpose(0, 1)
        hidden_states_1 = hidden_states[unsort_idx]

        return hidden_states_0, hidden_states_1

    def find_predicate_embeds(self, hidden_states, target_idx_in):
        Label_composer = hidden_states
        predicate_embeds = Label_composer[np.arange(0, Label_composer.size()[0]), target_idx_in]
        # T * B * H
        added_embeds = torch.zeros(Label_composer.size()[1], Label_composer.size()[0],
                                   Label_composer.size()[2]).to(device)
        concat_embeds = (added_embeds + predicate_embeds).transpose(0, 1)
        return concat_embeds

    def forward(self, sentence, p_sentence,  pos_tags, lengths, target_idx_in, region_marks,
                local_roles_voc, frames, local_roles_mask,
                sent_pred_lemmas_idx,  dep_tags,  dep_heads, targets, P_identification, all_l_ids,
                Predicate_link, Predicate_Labels_nd, Predicate_Labels,
                unlabeled_sentence=None, p_unlabeled_sentence=None, unlabeled_lengths=None, test=False):


        hidden_states_0, hidden_states_1 = self.shared_BilSTMEncoder_foward(sentence, p_sentence, lengths)

        # predicate identification
        Hidden_states_forID = torch.cat((hidden_states_0, hidden_states_1), 2)
        Predicate_identification = self.Idenficiation(self.label_dropout_1(F.relu(self.MLP_identification(Hidden_states_forID))))
        Predicate_identification_space = Predicate_identification.view(
            len(sentence[0]) * self.batch_size, -1)


        # dependency extractor
        concat_embeds_0 = self.find_predicate_embeds(hidden_states_0, target_idx_in)
        concat_embeds_1 = self.find_predicate_embeds(hidden_states_1, target_idx_in)

        Word_hidden = F.relu(self.hidden2tag_1(torch.cat((hidden_states_0, hidden_states_1), 2)))
        Predicate_hidden = F.relu(self.hidden2tag_2(torch.cat((concat_embeds_0, concat_embeds_1), 2)))
        FFF = torch.cat((Word_hidden, Predicate_hidden), 2)
        dep_tag_space = self.MLP_2(self.label_dropout_2(F.relu(self.MLP_1(FFF)))).view(
            len(sentence[0]) * self.batch_size, -1)
        TagProbs_use = F.softmax(dep_tag_space, dim=1).view(self.batch_size, len(sentence[0]), -1)


        # SRL module
        # construct SRL input
        TagProbs_noGrad = TagProbs_use.detach()
        h1 = F.tanh(self.tag2hidden(TagProbs_noGrad))

        h_layer_0 = hidden_states_0
        h_layer_1 = hidden_states_1

        w = F.softmax(self.elmo_w, dim=0)
        SRL_composer = self.elmo_gamma * (w[0] * h_layer_0 + w[1] * h_layer_1)
        SRL_composer = self.elmo_mlp(SRL_composer)


        fixed_embeds = self.word_fixed_embeddings_SRL(p_sentence)
        fixed_embeds = fixed_embeds.view(self.batch_size, len(sentence[0]), self.word_emb_dim)
        #sent_pred_lemmas_embeds = self.p_lemma_embeddings(sent_pred_lemmas_idx)
        embeds_SRL = self.word_embeddings_SRL(sentence)
        embeds_SRL = embeds_SRL.view(self.batch_size, len(sentence[0]), self.word_emb_dim)
        #pos_embeds = self.pos_embeddings(pos_tags)
        region_marks = self.region_embeddings(region_marks).view(self.batch_size, len(sentence[0]), 16)
        SRL_hidden_states = torch.cat((embeds_SRL,  fixed_embeds, region_marks,
                                       h1, SRL_composer), 2)
        SRL_hidden_states = self.SRL_input_dropout(SRL_hidden_states)


        # SRL layer
        embeds_sort, lengths_sort, unsort_idx = self.sort_batch(SRL_hidden_states, lengths)
        embeds_sort = rnn.pack_padded_sequence(embeds_sort, lengths_sort.cpu().numpy(), batch_first=True)
        # hidden states [time_steps * batch_size * hidden_units]
        hidden_states, self.hidden_4 = self.BiLSTM_SRL(embeds_sort, self.hidden_4)
        # it seems that hidden states is already batch first, we don't need swap the dims
        # hidden_states = hidden_states.permute(1, 2, 0).contiguous().view(self.batch_size, -1, )
        hidden_states, lens = rnn.pad_packed_sequence(hidden_states, batch_first=True)
        # hidden_states = hidden_states.transpose(0, 1)
        hidden_states = hidden_states[unsort_idx]
        hidden_states = self.hidden_state_dropout(hidden_states)

        # B * H
        hidden_states_3 = hidden_states
        hidden_states_word = F.relu(self.Non_Predicate_Proj(hidden_states_3))
        predicate_embeds = self.find_predicate_embeds(hidden_states_3, target_idx_in)
        hidden_states_predicate = F.relu(self.Predicate_Proj(predicate_embeds))

        left_part = torch.mm(hidden_states_word.view(self.batch_size * len(sentence[0]), -1), self.W_R + self.W_share)
        left_part = left_part.view(self.batch_size * len(sentence[0]), self.tagset_size, -1)
        hidden_states_predicate = hidden_states_predicate.view(self.batch_size * len(sentence[0]), -1, 1)
        tag_space = torch.bmm(left_part, hidden_states_predicate).view(
            len(sentence[0]) * self.batch_size, -1)

        SRLprobs = F.softmax(tag_space, dim=1)

        # +++++++++++++++++++++++
        wrong_l_nums = 0.0
        all_l_nums = 0.0

        right_noNull_predict = 0.0
        noNull_predict = 0.0
        noNUll_truth = 0.0
        dep_labels = np.argmax(dep_tag_space.cpu().data.numpy(), axis=1)
        for predict_l, gold_l in zip(dep_labels, Predicate_Labels.cpu().view(-1).data.numpy()):
            if predict_l > 1 and gold_l>0:
                noNull_predict += 1
            if gold_l != 0:
                all_l_nums += 1
                if gold_l != 1:
                    noNUll_truth += 1
                    if gold_l == predict_l:
                        right_noNull_predict += 1
            if predict_l != gold_l and gold_l != 0:
                wrong_l_nums += 1

        # +++++++++++++++++++++++
        wrong_l_nums_spe = 0.0
        all_l_nums_spe = 0.0

        right_noNull_predict_spe = 0.0
        noNull_predict_spe = 0.0
        noNUll_truth_spe = 0.0

        dep_labels = np.argmax(Predicate_identification_space.cpu().data.numpy(), axis=1)
        for predict_l, gold_l in zip(dep_labels,P_identification.cpu().view(-1).data.numpy()):
            if predict_l > 1 and gold_l!=0:
                noNull_predict_spe += 1
            if gold_l != 0:
                all_l_nums_spe += 1
                if gold_l != 1:
                    noNUll_truth_spe += 1
                    if gold_l == predict_l:
                        right_noNull_predict_spe += 1
            if predict_l != gold_l and gold_l != 0:
                wrong_l_nums_spe += 1


        loss_function = nn.CrossEntropyLoss(ignore_index=0)

        SRLloss = loss_function(tag_space, targets.view(-1))
        DEPloss = loss_function(dep_tag_space, Predicate_Labels.view(-1))
        IDloss = loss_function(Predicate_identification_space, P_identification.view(-1))

        loss = SRLloss + 0.5 *DEPloss + 0.5*IDloss

        if test:
            return SRLloss, DEPloss, IDloss, loss, SRLprobs, wrong_l_nums, all_l_nums, wrong_l_nums, all_l_nums, \
                   right_noNull_predict, noNull_predict, noNUll_truth, \
                   right_noNull_predict_spe, noNull_predict_spe, noNUll_truth_spe


        ## start unlabeled training:

        hidden_states_0, hidden_states_1 = self.shared_BilSTMEncoder_foward(unlabeled_sentence, p_unlabeled_sentence, unlabeled_lengths)

        ## perform primary predicate identification

        Hidden_states_forID = torch.cat((hidden_states_0, hidden_states_1), 2)
        Predicate_identification = self.Idenficiation(
            self.label_dropout_1(F.relu(self.MLP_identification(Hidden_states_forID))))
        Predicate_identification_space = Predicate_identification.view(
            len(unlabeled_sentence[0]) * self.batch_size, -1)
        Predicate_probs = Predicate_identification_space[:, 2].view(self.batch_size, len(unlabeled_sentence[0]))
        Predicate_idx_batch = np.argmax(Predicate_probs.cpu().data.numpy(), axis=1)

        # primary dependency extractor
        concat_embeds_0 = self.find_predicate_embeds(hidden_states_0, target_idx_in)
        concat_embeds_1 = self.find_predicate_embeds(hidden_states_1, target_idx_in)

        Word_hidden = F.relu(self.hidden2tag_1(torch.cat((hidden_states_0, hidden_states_1), 2)))
        Predicate_hidden = F.relu(self.hidden2tag_2(torch.cat((concat_embeds_0, concat_embeds_1), 2)))
        FFF = torch.cat((Word_hidden, Predicate_hidden), 2)
        dep_tag_space = self.MLP_2(self.label_dropout_2(F.relu(self.MLP_1(FFF)))).view(
            len(unlabeled_sentence[0]) * self.batch_size, -1)
        TagProbs_use = F.softmax(dep_tag_space, dim=1).view(self.batch_size, len(unlabeled_sentence[0]), -1)

        unlabeled_region_mark = np.zeros(unlabeled_sentence.size(), dtype='int64')
        for i in range(len(unlabeled_region_mark)):
            unlabeled_region_mark[i][Predicate_idx_batch[i]] = 1

        unlabeled_region_mark_in = torch.from_numpy(unlabeled_region_mark).to(device)
        unlabeled_region_mark_embeds = self.region_embeddings(unlabeled_region_mark_in)

        hidden_forward , hidden_backward= hidden_states_0.split(self.hidden_dim, 2)


        ## perform primary SRL

        TagProbs_noGrad = TagProbs_use.detach()
        h1 = F.tanh(self.tag2hidden(TagProbs_noGrad))

        h_layer_0 = hidden_states_0
        h_layer_1 = hidden_states_1

        w = F.softmax(self.elmo_w, dim=0)
        SRL_composer = self.elmo_gamma * (w[0] * h_layer_0 + w[1] * h_layer_1)
        SRL_composer = self.elmo_mlp(SRL_composer)

        fixed_embeds = self.word_fixed_embeddings_SRL(p_unlabeled_sentence)
        fixed_embeds = fixed_embeds.view(self.batch_size, len(unlabeled_sentence[0]), self.word_emb_dim)
        # sent_pred_lemmas_embeds = self.p_lemma_embeddings(sent_pred_lemmas_idx)
        embeds_SRL = self.word_embeddings_SRL(unlabeled_sentence)
        embeds_SRL = embeds_SRL.view(self.batch_size, len(unlabeled_sentence[0]), self.word_emb_dim)
        # pos_embeds = self.pos_embeddings(pos_tags)
        SRL_hidden_states = torch.cat((embeds_SRL, fixed_embeds, unlabeled_region_mark_embeds,
                                       h1, SRL_composer), 2)
        SRL_hidden_states = self.SRL_input_dropout(SRL_hidden_states)

        # SRL layer
        embeds_sort, lengths_sort, unsort_idx = self.sort_batch(SRL_hidden_states, unlabeled_lengths)
        embeds_sort = rnn.pack_padded_sequence(embeds_sort, lengths_sort.cpu().numpy(), batch_first=True)
        # hidden states [time_steps * batch_size * hidden_units]
        self.hidden_4 = self.init_hidden_share()
        hidden_states, self.hidden_4 = self.BiLSTM_SRL(embeds_sort, self.hidden_4)
        # it seems that hidden states is already batch first, we don't need swap the dims
        # hidden_states = hidden_states.permute(1, 2, 0).contiguous().view(self.batch_size, -1, )
        hidden_states, lens = rnn.pad_packed_sequence(hidden_states, batch_first=True)
        # hidden_states = hidden_states.transpose(0, 1)
        hidden_states = hidden_states[unsort_idx]
        hidden_states = self.hidden_state_dropout(hidden_states)

        # B * H
        hidden_states_3 = hidden_states
        hidden_states_word = F.relu(self.Non_Predicate_Proj(hidden_states_3))
        predicate_embeds = self.find_predicate_embeds(hidden_states_3, Predicate_idx_batch)
        hidden_states_predicate = F.relu(self.Predicate_Proj(predicate_embeds))

        left_part = torch.mm(hidden_states_word.view(self.batch_size * len(unlabeled_sentence[0]), -1), self.W_R + self.W_share)
        left_part = left_part.view(self.batch_size * len(unlabeled_sentence[0]), self.tagset_size, -1)
        hidden_states_predicate = hidden_states_predicate.view(self.batch_size * len(unlabeled_sentence[0]), -1, 1)
        tag_space = torch.bmm(left_part, hidden_states_predicate).view(
            len(unlabeled_sentence[0]) * self.batch_size, -1)

        ## obtain the teacher probs
        SRLprobs_teacher = F.softmax(tag_space, dim=1)#.detach()

        ## perform FF SRL

        fixed_embeds = self.word_fixed_embeddings_SRL(p_unlabeled_sentence)
        fixed_embeds = fixed_embeds.view(self.batch_size, len(unlabeled_sentence[0]), self.word_emb_dim)
        # sent_pred_lemmas_embeds = self.p_lemma_embeddings(sent_pred_lemmas_idx)
        embeds_SRL = self.word_embeddings_SRL(unlabeled_sentence)
        embeds_SRL = embeds_SRL.view(self.batch_size, len(unlabeled_sentence[0]), self.word_emb_dim)
        # pos_embeds = self.pos_embeddings(pos_tags)
        SRL_hidden_states = torch.cat((embeds_SRL, fixed_embeds, unlabeled_region_mark_embeds, hidden_forward), 2)
        SRL_hidden_states = self.SRL_input_dropout(SRL_hidden_states)

        # SRL layer
        embeds_sort, lengths_sort, unsort_idx = self.sort_batch(SRL_hidden_states, unlabeled_lengths)
        embeds_sort = rnn.pack_padded_sequence(embeds_sort, lengths_sort.cpu().numpy(), batch_first=True)
        self.hidden_4 = self.init_hidden_share()
        hidden_states, self.hidden_4 = self.BiLSTM_SRL_fwd_fwd(embeds_sort, self.hidden_4)
        hidden_states, lens = rnn.pad_packed_sequence(hidden_states, batch_first=True)
        # hidden_states = hidden_states.transpose(0, 1)
        hidden_states = hidden_states[unsort_idx]
        hidden_states = self.hidden_state_dropout(hidden_states)

        # B * H
        hidden_states_3 = hidden_states
        hidden_states_word = F.relu(self.Non_Predicate_Proj_FF(hidden_states_3))
        predicate_embeds = self.find_predicate_embeds(hidden_states_3, Predicate_idx_batch)
        hidden_states_predicate = F.relu(self.Predicate_Proj_FF(predicate_embeds))

        left_part = torch.mm(hidden_states_word.view(self.batch_size * len(unlabeled_sentence[0]), -1), self.W_R_FF + self.W_share_FF)
        left_part = left_part.view(self.batch_size * len(unlabeled_sentence[0]), self.tagset_size, -1)
        hidden_states_predicate = hidden_states_predicate.view(self.batch_size * len(unlabeled_sentence[0]), -1, 1)
        tag_space = torch.bmm(left_part, hidden_states_predicate).view(
            len(unlabeled_sentence[0]) * self.batch_size, -1)

        ## obtain the teacher probs
        SRLprobs_student_FF = F.softmax(tag_space, dim=1)

        unlabeled_loss_function = nn.KLDivLoss()
        SRL_FF_loss = unlabeled_loss_function(SRLprobs_student_FF, SRLprobs_teacher)
        log(SRL_FF_loss)


        return SRLloss, DEPloss, IDloss, loss, SRLprobs, wrong_l_nums, all_l_nums, wrong_l_nums, all_l_nums,  \
               right_noNull_predict, noNull_predict, noNUll_truth,\
               right_noNull_predict_spe, noNull_predict_spe, noNUll_truth_spe

    @staticmethod
    def sort_batch(x, l):
        l = torch.from_numpy(np.asarray(l))
        l_sorted, sidx = l.sort(0, descending=True)
        x_sorted = x[sidx]
        _, unsort_idx = sidx.sort()
        return x_sorted, l_sorted, unsort_idx