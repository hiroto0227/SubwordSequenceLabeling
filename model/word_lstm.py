
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

from model.charrep import CharRep
from model.subwordrep import SubwordRep


class WordLSTM(nn.Module):
    def __init__(self, dataset):
        super(WordLSTM, self).__init__()
        self.gpu = dataset.HP_gpu
        self.batch_size = dataset.HP_batch_size
        self.sw_num = dataset.sw_num
        self.input_size = dataset.word_emb_dim

        # char
        self.char_rep = CharRep(dataset.char_alphabet_size,
                                dataset.pretrain_char_embedding,
                                dataset.char_emb_dim,
                                dataset.HP_char_hidden_dim,
                                dataset.HP_dropout,
                                dataset.HP_gpu)
        self.input_size += dataset.HP_char_hidden_dim

        # subword
        self.subword_reps = [SubwordRep(
            alphabet_size=dataset.sw_alphabet_size_list[i],
            embedding_dim=dataset.sw_emb_dim,
            hidden_dim=dataset.HP_sw_hidden_dim,
            dropout=dataset.HP_dropout,
            gpu=dataset.HP_gpu) for i in range(self.sw_num)]
        self.input_size += dataset.HP_sw_hidden_dim * self.sw_num

        # word
        self.embedding_dim = dataset.word_emb_dim
        self.drop = nn.Dropout(dataset.HP_dropout)
        self.word_embedding = nn.Embedding(dataset.word_alphabet.size(), self.embedding_dim)
        if dataset.pretrain_word_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(dataset.pretrain_word_embedding))
        else:
            self.word_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(dataset.word_alphabet.size(), self.embedding_dim)))
        self.droplstm = nn.Dropout(dataset.HP_dropout)
        lstm_hidden = dataset.HP_hidden_dim // 2
        self.lstm = nn.LSTM(self.input_size, lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)

        if self.gpu:
            self.drop.cuda()
            self.word_embedding.cuda()
            self.droplstm.cuda()
            self.lstm.cuda()

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb


    def forward(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover , sw_inputs, sw_seqs_lengths, sw_seqs_recover, sw_fmasks, sw_bmasks):
        batch_size = word_inputs.size(0)
        sent_len = word_inputs.size(1)
        word_embs =  self.word_embedding(word_inputs)
        word_list = [word_embs]

        ### Char ###
        char_rep = self.char_rep.get_last_hiddens(char_inputs, char_seq_lengths.cpu().numpy())
        char_rep = char_rep[char_seq_recover]
        char_rep = char_rep.view(batch_size, sent_len, -1)
        word_list.append(char_rep)
        #word_embs = torch.cat([word_embs, char_rep], 2)

        ### SubWord ###
        for idx in range(self.sw_num):
            sw_rep = self.subword_reps[idx].get_masked_hidden(sw_inputs[idx], sw_seqs_lengths[idx].cpu().numpy(), sw_fmasks[idx], sw_bmasks[idx], int(word_embs.shape[1]))
            sw_rep = sw_rep[sw_seqs_recover[idx]]
            word_list.append(sw_rep)
    
        word_embs = torch.cat(word_list, 2)
        word_represent = self.drop(word_embs)

        ### Word LSTM ###
        packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
        hidden = None
        lstm_out, hidden = self.lstm(packed_words, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)  # (seq_len, seq_len, hidden_size)
        lstm_out = self.droplstm(lstm_out.transpose(1, 0))  # (batch_size, seq_len, hidden_size)
        return lstm_out