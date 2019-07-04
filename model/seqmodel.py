from typing import List

import numpy as np
import torch
import torch.nn as nn

from model.word_lstm import WordLSTM
from model.crf import CRF

class SeqModel(nn.Module):
    def __init__(
        self, 
        config_dic: dict, 
        word_vocab_dim: int, 
        char_vocab_dim: int, 
        sw_vocab_dim_list: List[int], 
        label_vocab_dim: int, 
        pretrain_word_embedding: np.ndarray,
    ):
        super().__init__()
        self.gpu = config_dic.get("gpu")
        self.label_vocab_dim = label_vocab_dim

        self.word_lstm = WordLSTM(
            config_dic, 
            word_vocab_dim,
            char_vocab_dim,
            sw_vocab_dim_list,
            pretrain_word_embedding,
            config_dic.get("use_modality_attention"),
            config_dic.get("ner_dropout"))
        self.hidden2tag = nn.Linear(config_dic.get("word_hidden_dim"), self.label_vocab_dim + 2)  # for START and END tag
        self.crf = CRF(self.label_vocab_dim, self.gpu)
        
        if self.gpu:
            self.word_lstm.cuda()
            self.hidden2tag.cuda()

    def neg_log_likelihood_loss(self, word_features, char_features, sw_features_list, label_features, size_average=True):
        self.zero_grad()
        mask = word_features.get("masks")
        lstm_out = self.word_lstm(word_features, char_features, sw_features_list)
        out = self.hidden2tag(lstm_out)
        total_loss = self.crf.neg_log_likelihood_loss(out, mask, label_features.get("label_ids"))  # outとmaskの並びはあっているっぽい。
        _, tag_seq  = self.crf._viterbi_decode(out, mask)
        # if size_average:
        #     total_loss = total_loss / (torch.sum(mask) / out.shape[0])
        return total_loss, tag_seq

    def forward(self, word_features, char_features, sw_features_list):
        self.zero_grad()
        mask = word_features.get("masks")
        lstm_out = self.word_lstm(word_features, char_features, sw_features_list)
        out = self.hidden2tag(lstm_out)
        _, tag_seq = self.crf._viterbi_decode(out, mask)
        return tag_seq

    def load_expanded_state_dict(self, lm_state_dict):
        """lmとnerの時ではvocab_sizeが異なるのでrandom valueでexpandしてload
        """
        expanded_state_dict = self.state_dict()
        for lm_key, lm_value in lm_state_dict.items():
            if lm_key in expanded_state_dict.keys():
                if expanded_state_dict.get(lm_key).shape == lm_value.shape:
                    expanded_state_dict[lm_key] = lm_value
                else:
                    expanded_state_dict[lm_key] = expand_weight(lm_value, expanded_state_dict.get(lm_key).shape, self.gpu)
        self.load_state_dict(expanded_state_dict)

        
def random_embedding(vocab_size, embedding_dim, gpu: bool):
    pretrain_emb = np.empty([vocab_size, embedding_dim])
    scale = np.sqrt(3.0 / embedding_dim)
    for index in range(vocab_size):
        pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
    if gpu:
        return torch.from_numpy(pretrain_emb).float().cuda()
    else:
        return torch.from_numpy(pretrain_emb).float()
    

def expand_weight(value, expanded_shape, gpu: bool):
    lm_vocab_size = value.shape[0]
    ner_vocab_size = expanded_shape[0]
    pad_random_embedding = random_embedding(ner_vocab_size - lm_vocab_size, value.shape[1], gpu)
    return torch.cat([value, pad_random_embedding], dim=0)