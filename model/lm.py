from typing import List

import numpy as np
import torch.nn as nn

from model.word_lstm import WordLSTM

class LanguageModel(nn.Module):
    def __init__(
        self, 
        config_dic: dict, 
        word_vocab_dim: int, 
        char_vocab_dim: int, 
        sw_vocab_dim_list: List[int], 
        pretrain_word_embedding: np.ndarray
    ):
        super().__init__()
        self.word_lstm = WordLSTM(
            config_dic,
            word_vocab_dim, 
            char_vocab_dim, 
            sw_vocab_dim_list, 
            pretrain_word_embedding, 
            config_dic.get("use_modality_attention"), 
            config_dic.get("lm_dropout"))
        self.f_linear = nn.Linear(config_dic.get("word_hidden_dim") // 2, word_vocab_dim)
        self.b_linear = nn.Linear(config_dic.get("word_hidden_dim") // 2, word_vocab_dim)
        self.softmax = nn.LogSoftmax(dim=1)

        if config_dic.get("gpu"):
            self.f_linear.cuda()
            self.b_linear.cuda()
            self.softmax.cuda()

    def calc_next_word_id_prob(self, word_features, char_features, sw_features_list):
        self.zero_grad()
        output = self.word_lstm(word_features, char_features, sw_features_list)
        f_out, b_out = output.split(output.shape[2] // 2, dim=2)
        f_out = self.f_linear(f_out)
        b_out = self.b_linear(b_out) 
        return self.softmax(f_out), self.softmax(b_out)
