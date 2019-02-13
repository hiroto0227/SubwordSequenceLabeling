
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model.word_lstm import WordLSTM
from model.crf import CRF


class SeqLabel(nn.Module):
    def __init__(self, dataset):
        super(SeqLabel, self).__init__()
        self.gpu = dataset.HP_gpu
        self.average_batch = dataset.average_batch_loss

        self.word_lstm = WordLSTM(dataset)
        label_size = dataset.label_alphabet_size
        dataset.label_alphabet_size += 2 # for START and END tag
        self.hidden2tag = nn.Linear(dataset.HP_hidden_dim, dataset.label_alphabet_size)

        if self.gpu:
            self.crf = CRF(label_size, self.gpu)


    def neg_log_likelihood_loss(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, sw_inputs, sw_seqs_lengths, sw_seqs_recovers, sw_fmasks, sw_bmasks, batch_label, mask):
        outs = self.word_lstm(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, sw_inputs, sw_seqs_lengths, sw_seqs_recovers, sw_fmasks, sw_bmasks)
        batch_size = word_inputs.size(0)
        total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
        _, tag_seq = self.crf._viterbi_decode(outs, mask)
        if self.average_batch:
            total_loss = total_loss / batch_size
        return total_loss, tag_seq

    def forward(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, sw_inputs, sw_seqs_lengths, sw_seqs_recovers, sw_fmasks, sw_bmasks, mask):
        outputs = self.word_lstm(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, sw_inputs, sw_seqs_lengths, sw_seqs_recovers, sw_fmasks, sw_bmasks)
        _, tag_seq = self.crf._viterbi_decode(outputs, mask)
        return tag_seq
