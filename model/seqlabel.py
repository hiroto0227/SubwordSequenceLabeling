
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model.word_lstm import WordLSTM
from model.crf import CRF


class SeqLabel(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.gpu = dataset.gpu
        self.average_batch = dataset.average_batch_loss

        self.word_lstm = WordLSTM(dataset)
        label_size = len(dataset.label_alphabet.instances)
        self.hidden2tag = nn.Linear(dataset.hidden_dim, label_size + 2)
        self.crf = CRF(label_size, self.gpu)
        
        if self.gpu:
            self.hidden2tag.cuda()


    def neg_log_likelihood_loss(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, sw_inputs, sw_seqs_lengths, sw_seqs_recovers, sw_fmasks, sw_bmasks, batch_label, mask):
        batch_size = word_inputs.size(0)
        outs = self.word_lstm(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, sw_inputs, sw_seqs_lengths, sw_seqs_recovers, sw_fmasks, sw_bmasks)
        outs = self.hidden2tag(outs)
        print(outs)
        total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
        _, tag_seq = self.crf._viterbi_decode(outs, mask)
        if self.average_batch:
            total_loss = total_loss / batch_size
        return total_loss, tag_seq

    def forward(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, sw_inputs, sw_seqs_lengths, sw_seqs_recovers, sw_fmasks, sw_bmasks, mask):
        outputs = self.word_lstm(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, sw_inputs, sw_seqs_lengths, sw_seqs_recovers, sw_fmasks, sw_bmasks)
        outputs = self.hidden2tag(outputs)
        _, tag_seq = self.crf._viterbi_decode(outputs, mask)
        return tag_seq

    def load_state_dict(self, state_dict):
        sd = {k.replace("word_lstm.", ""): v for k,v in state_dict.items()}
        try:
            self.word_lstm.load_state_dict(sd)
        except RuntimeError as e:
            print(e)
