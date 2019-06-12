from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model.charrep import CharRep
from model.subwordrep import SubwordRep


class WordLSTM(nn.Module):
    
    def __init__(
        self: nn.Module, 
        config_dic: dict, 
        word_vocab_dim: int, 
        char_vocab_dim: int, 
        sw_vocab_dim_list: List[int], 
        pretrain_word_embedding: np.ndarray, 
        dropout_rate: float
    ):
        super().__init__()
        self.gpu = config_dic.get("gpu")
        self.use_sw = False
        self.char_rep = CharRep(config_dic, char_vocab_dim, dropout_rate)
        self.lstm_input_dim = config_dic.get("word_emb_dim")
        self.lstm_input_dim += config_dic.get("char_hidden_dim")
        self.subword_rep_list = nn.ModuleList()
        for sw_vocab_dim in sw_vocab_dim_list:
            print(sw_vocab_dim)
            self.use_sw = True
            self.subword_rep_list.append(SubwordRep(config_dic, sw_vocab_dim, dropout_rate))
            self.lstm_input_dim += config_dic.get("sw_hidden_dim")
        self.embedding_dim = config_dic.get("word_emb_dim")
        self.dropout = nn.Dropout(dropout_rate)
        self.word_hidden_dim = config_dic.get("word_hidden_dim")
        self.word_embedding = nn.Embedding(word_vocab_dim, self.embedding_dim)
        if pretrain_word_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(pretrain_word_embedding))
        else:
            self.word_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(word_vocab_dim, self.embedding_dim)))
        self.lstm = nn.LSTM(self.lstm_input_dim, self.word_hidden_dim // 2, batch_first=True, bidirectional=True)

        if self.gpu:
            self.dropout.cuda()
            self.word_embedding.cuda()
            self.lstm.cuda()

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def forward(self, word_features, char_features, sw_features_list):
        word_ids = word_features.get("word_ids")
        batch_size, sent_len = word_ids.shape
        word_lstm_input = []
        
        char_rep = self.char_rep.get_last_hiddens(char_features)
        
        
        lengths = word_ids.eq(0).lt(1).sum(dim=1)
        word_sorted_lengths, word_perm_ix = lengths.sort(descending=True)
        word_embs =  self.word_embedding(word_ids)

        word_lstm_input.append(char_rep[word_perm_ix])
        word_lstm_input.append(word_embs[word_perm_ix])

        for i, sw_features in enumerate(sw_features_list):
            sw_rep, sw_sorted_lengths = self.subword_rep_list[i].get_masked_hidden(sw_features)
            assert all(sw_sorted_lengths == word_sorted_lengths),  \
                "Not Equal sw_sorted_lenghts, word_sorted_lengths"
            word_lstm_input.append(sw_rep)
        
        word_rep = torch.cat(word_lstm_input, 2)  # sw_repはsort済み
        
        ### Word LSTM ###
        word_rep = self.dropout(word_rep)
        packed_words = pack_padded_sequence(word_rep, word_sorted_lengths, batch_first=True)
        lstm_out, hidden = self.lstm(packed_words)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = self.dropout(lstm_out)

        _, recover_parm_ix = word_perm_ix.sort()
        return lstm_out[recover_parm_ix]
