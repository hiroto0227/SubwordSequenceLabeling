
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model.charrep import CharRep
from model.subwordrep import SubwordRep


class WordLSTM(nn.Module):
    
    def __init__(self, config_dic, word_vocab_dim, char_vocab_dim, sw_vocab_dim, pretrain_word_embedding: np.ndarray):
        super().__init__()
        self.gpu = config_dic.get("gpu")
        self.char_rep = CharRep(config_dic, char_vocab_dim)
        self.subword_rep = SubwordRep(config_dic, sw_vocab_dim)
        self.lstm_input_dim = config_dic.get("word_emb_dim", 50)
        self.lstm_input_dim += config_dic.get("char_hidden_dim", 50)
        self.lstm_input_dim += config_dic.get("sw_hidden_dim", 100)
        self.embedding_dim = config_dic.get("sord_emb_dim", 50)
        self.drop = nn.Dropout(config_dic.get("dropout", 0.5))
        self.word_hidden_dim = config_dic.get("word_hidden_dim", 200)
        self.word_embedding = nn.Embedding(word_vocab_dim, self.embedding_dim)
        if config_dic.get(pretrain_word_embedding) is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(config_dic.get(pretrain_word_embedding)))
        else:
            self.word_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(word_vocab_dim, self.embedding_dim)))
        self.lstm = nn.LSTM(self.lstm_input_dim, self.word_hidden_dim // 2, batch_first=True, bidirectional=True)

        if self.gpu:
            self.drop.cuda()
            self.word_embedding.cuda()
            self.lstm.cuda()

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb


    def forward(self, word_features, char_features, sw_features):
        word_ids = word_features.get("word_ids")
        batch_size, sent_len = word_ids.shape
        
        char_rep = self.char_rep.get_last_hiddens(char_features)
        sw_rep, sw_sorted_lengths = self.subword_rep.get_masked_hidden(sw_features)

        lengths = word_ids.eq(0).lt(1).sum(dim=1)
        word_sorted_lengths, word_perm_ix = lengths.sort(descending=True)
        assert all(sw_sorted_lengths == word_sorted_lengths),  \
            "Not Equal sw_sorted_lenghts, word_sorted_lengths"

        word_embs =  self.word_embedding(word_ids)
        word_rep = torch.cat([word_embs[word_perm_ix], char_rep[word_perm_ix], sw_rep], 2)  # sw_repはsort済み

        ### Word LSTM ###
        packed_words = pack_padded_sequence(word_rep, word_sorted_lengths, True)
        hidden = None
        lstm_out, hidden = self.lstm(packed_words, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)  # (seq_len, seq_len, hidden_size)
        lstm_out = self.drop(lstm_out.transpose(1, 0))  # (batch_size, seq_len, hidden_size)

        _, recover_parm_ix = word_perm_ix.sort()
        return lstm_out[recover_parm_ix]