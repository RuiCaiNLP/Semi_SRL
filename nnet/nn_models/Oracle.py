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
_BIG_NUMBER = 10. ** 6.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BiLSTMTagger(nn.Module):

    #def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
    def __init__(self, hps, *_):
        super(BiLSTMTagger, self).__init__()

        batch_size = hps['batch_size']
        lstm_hidden_dim = hps['sent_hdim']
        sent_embedding_dim_SRL = 3 * hps['sent_edim'] + 1 * hps['pos_edim'] + 1
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

        self.pos_embeddings = nn.Embedding(self.pos_size, hps['pos_edim'])
        self.p_lemma_embeddings = nn.Embedding(self.frameset_size, hps['sent_edim'])
        #self.lr_dep_embeddings = nn.Embedding(self.lr_dep_size, hps[])

        self.word_fixed_embeddings = nn.Embedding(vocab_size, hps['sent_edim'])
        self.word_fixed_embeddings.weight.data.copy_(torch.from_numpy(hps['word_embeddings']))


        self.role_embeddings = nn.Embedding(self.tagset_size, role_embedding_dim)
        self.frame_embeddings = nn.Embedding(self.frameset_size, frame_embedding_dim)

        self.DEP_Label_embeddings = nn.Embedding(self.dep_size, hps['pos_edim'])
        self.DEP_Link_embeddings = nn.Embedding(4, hps['pos_edim'])



        self.SRL_input_dropout = nn.Dropout(p=0.3)
        self.hidden_state_dropout = nn.Dropout(p=0.3)


        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.

        self.num_layers = 4
        self.BiLSTM_SRL = nn.LSTM(sent_embedding_dim_SRL, hidden_size=lstm_hidden_dim, batch_first=True,
                                    bidirectional=True, num_layers=self.num_layers)

        init.orthogonal_(self.BiLSTM_SRL.all_weights[0][0])
        init.orthogonal_(self.BiLSTM_SRL.all_weights[0][1])
        init.orthogonal_(self.BiLSTM_SRL.all_weights[1][0])
        init.orthogonal_(self.BiLSTM_SRL.all_weights[1][1])


        # non-linear map to role embedding
        self.role_map = nn.Linear(in_features=role_embedding_dim * 2, out_features=self.hidden_dim * 4)

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
        return (torch.zeros(4 * 2, self.batch_size, self.hidden_dim, requires_grad=False).to(device),
                torch.zeros(4 * 2, self.batch_size, self.hidden_dim, requires_grad=False).to(device))

    def init_hidden_spe(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        #return (Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
        #        Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))
        return (torch.zeros(1 * 2, self.batch_size, self.hidden_dim, requires_grad=False).to(device),
                torch.zeros(1 * 2, self.batch_size, self.hidden_dim, requires_grad=False).to(device))


    def forward(self, sentence, p_sentence,  pos_tags, lengths, target_idx_in, region_marks,
                local_roles_voc, frames, local_roles_mask,
                sent_pred_lemmas_idx,  dep_tags,  dep_heads, targets, specific_dep_tags, specific_dep_relations, test=False):



        #contruct input for DEP
        pos_embeds = self.pos_embeddings(pos_tags)
        region_marks = region_marks.view(self.batch_size, len(sentence[0]), 1)
        fixed_embeds = self.word_fixed_embeddings(p_sentence)
        fixed_embeds = fixed_embeds.view(self.batch_size, len(sentence[0]), self.word_emb_dim)



        h_label_embeddings = self.DEP_Label_embeddings(dep_tags)
        h_link_embeddings = self.DEP_Link_embeddings(specific_dep_relations)

        sent_pred_lemmas_embeds = self.p_lemma_embeddings(sent_pred_lemmas_idx)
        embeds_SRL = self.word_embeddings_SRL(sentence)
        embeds_SRL = embeds_SRL.view(self.batch_size, len(sentence[0]), self.word_emb_dim)
        SRL_hidden_states = torch.cat((embeds_SRL, fixed_embeds, sent_pred_lemmas_embeds, pos_embeds, region_marks), 2)
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
        predicate_embeds = hidden_states_3[np.arange(0, hidden_states_3.size()[0]), target_idx_in]
        # T * B * H
        added_embeds = torch.zeros(hidden_states_3.size()[1], hidden_states_3.size()[0], hidden_states_3.size()[2]).to(device)
        predicate_embeds = added_embeds + predicate_embeds
        # B * T * H
        predicate_embeds = predicate_embeds.transpose(0, 1)
        hidden_states = torch.cat((hidden_states_3, predicate_embeds), 2)
        # print(hidden_states)
        # non-linear map and rectify the roles' embeddings
        # roles = Variable(torch.from_numpy(np.arange(0, self.tagset_size)))

        # B * roles
        # log(local_roles_voc)
        # log(frames)

        # B * roles * h
        role_embeds = self.role_embeddings(local_roles_voc)
        frame_embeds = self.frame_embeddings(frames)

        role_embeds = torch.cat((role_embeds, frame_embeds), 2)
        mapped_roles = F.relu(self.role_map(role_embeds))
        mapped_roles = torch.transpose(mapped_roles, 1, 2)

        # b, times, roles
        tag_space = torch.matmul(hidden_states, mapped_roles)
        #tag_space = hidden_states.mm(mapped_roles)



        # b, roles
        #sub = torch.div(torch.add(local_roles_mask, -1.0), _BIG_NUMBER)
        sub = torch.add(local_roles_mask, -1.0) * _BIG_NUMBER
        sub = torch.FloatTensor(sub.cpu().numpy()).to(device)
        # b, roles, times
        tag_space = torch.transpose(tag_space, 0, 1)
        tag_space += sub
        # b, T, roles
        tag_space = torch.transpose(tag_space, 0, 1)
        tag_space = tag_space.view(len(sentence[0])*self.batch_size, -1)

        SRLprobs = F.softmax(tag_space, dim=1)



        #loss_function = nn.NLLLoss(ignore_index=0)
        targets = targets.view(-1)
        #tag_scores = F.log_softmax(tag_space)
        #loss = loss_function(tag_scores, targets)
        loss_function = nn.CrossEntropyLoss(ignore_index=0)

        SRLloss = loss_function(tag_space, targets)


        #weight = float(SRLloss.cpu().data.numpy())
        #if weight > 0.1:
        #    weight = 0.1
        #p = nr.rand()
        #if p<0.2:
        #    loss = SRLloss + DEPloss + SPEDEPloss
        #else:
        #    loss = SRLloss
        loss = SRLloss
        return SRLloss, SRLloss, SRLloss, loss, SRLprobs, 1, 1, 1, 1,  \
               1, 1, 1,\
               1, 1, 1

    @staticmethod
    def sort_batch(x, l):
        l = torch.from_numpy(np.asarray(l))
        l_sorted, sidx = l.sort(0, descending=True)
        x_sorted = x[sidx]
        _, unsort_idx = sidx.sort()
        return x_sorted, l_sorted, unsort_idx


