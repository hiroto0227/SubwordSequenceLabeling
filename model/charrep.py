
import numpy as np
import torch
import torch.nn as nn


class CharRep(nn.Module):
    
    def __init__(self, config_dic, char_vocab_dim):
        super().__init__()
        self.gpu = config_dic.get("gpu")
        self.char_emb_dim = config_dic.get("char_emb_dim", 30)
        self.char_hidden_dim = config_dic.get("char_hidden_dim", 100)# bidirectional
        self.dropout = nn.Dropout(config_dic.get("dropout", 0.5))
        self.char_embeddings = nn.Embedding(char_vocab_dim, self.char_emb_dim)
        self.char_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(char_vocab_dim, self.char_emb_dim)))
        self.char_lstm = nn.LSTM(self.char_emb_dim, self.char_hidden_dim // 2, batch_first=True, bidirectional=True)
        if self.gpu:
            self.dropout.cuda()
            self.char_embeddings.cuda()
            self.char_lstm.cuda()

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def get_last_hiddens(self, char_features):
        """
            input:
                input: Variable(batch_size, word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        char_ids = char_features.get("char_ids")
        batch_size, seq_length, char_seq_length = char_ids.shape

        char_ids = char_ids.view(-1, char_seq_length)
        lengths = char_ids.eq(0).lt(1).sum(dim=1)

        char_embeds = self.dropout(self.char_embeddings(char_ids))  # B x L x L_c x Emb
        char_hidden = None
        char_rnn_out, char_hidden = self.char_lstm(char_embeds, char_hidden)

        out = torch.zeros((batch_size * seq_length, self.char_hidden_dim), dtype=torch.float)
        for i in range(batch_size * seq_length):
            out[i, :self.char_hidden_dim // 2] = char_rnn_out[i, lengths[i] - 1, :self.char_hidden_dim // 2]  # get last value on forward
            out[i, self.char_hidden_dim // 2:] = char_rnn_out[i, 0, self.char_hidden_dim // 2:]  # get first value on backward
        
        if self.gpu:
            return out.view(batch_size, seq_length, -1).cuda()
        else:
            return out.view(batch_size, seq_length, -1)