
import torch.nn as nn
import torch.nn.functional as F

from model.word_lstm import WordLSTM


class LanguageModel(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.word_lstm = WordLSTM(dataset)
        self.f_linear = nn.Linear(dataset.hidden_dim // 2, dataset.word_alphabet.size())
        self.b_linear = nn.Linear(dataset.hidden_dim // 2, dataset.word_alphabet.size())

        if dataset.gpu:
            self.f_linear.cuda()
            self.b_linear.cuda()
  
    def forward(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, sw_inputs, sw_seqs_lengths, sw_seqs_recovers, sw_fmasks, sw_bmasks):
        output = self.word_lstm(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, sw_inputs, sw_seqs_lengths, sw_seqs_recovers, sw_fmasks, sw_bmasks)
        return output

    def calc_next_word_id_prob(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, sw_inputs, sw_seqs_lengths, sw_seqs_recovers, sw_fmasks, sw_bmasks):
        output = self.word_lstm(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, sw_inputs, sw_seqs_lengths, sw_seqs_recovers, sw_fmasks, sw_bmasks)
        f_out, b_out = output.split(output.shape[2] // 2, dim=2)
        f_out = self.f_linear(f_out)
        b_out = self.b_linear(b_out) 
        return f_out, b_out