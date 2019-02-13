
import torch.nn as nn

from model.word_lstm import WordLSTM


class LanguageModel(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.word_lstm = WordLSTM(dataset)
        self.hidden2emb = nn.Linear(dataset.HP_hidden_dim, dataset.word_emb_dim)

        if dataset.HP_gpu:
            self.hidden2emb = self.hidden2tag.cuda()
  
    def forward(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, sw_inputs, sw_seqs_lengths, sw_seqs_recovers, sw_fmasks, sw_bmasks):
        output = self.word_lstm(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, sw_inputs, sw_seqs_lengths, sw_seqs_recovers, sw_fmasks, sw_bmasks)
        #self.hidden2emb(output)
        return output
