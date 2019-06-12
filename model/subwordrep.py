import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


class SubwordRep(nn.Module):
    def __init__(self, config_dic, sw_vocab_dim, dropout_rate: float):
        super().__init__()
        self.gpu = config_dic.get("gpu", False)
        self.sw_emb_dim = config_dic.get("sw_emb_dim")
        self.sw_hidden_dim = config_dic.get("sw_hidden_dim")
        self.dropout = nn.Dropout(dropout_rate)
        self.embeddings = nn.Embedding(sw_vocab_dim, self.sw_emb_dim)
        self.embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(sw_vocab_dim, self.sw_emb_dim)))
        self.lstm = nn.LSTM(self.sw_emb_dim, self.sw_hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        
        if self.gpu:
            self.dropout.cuda()
            self.embeddings.cuda()
            self.lstm.cuda()

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def get_masked_hidden(self, sw_features) -> (torch.FloatTensor, torch.LongTensor):
        """sw_featuresを受け取りLSTMにより計算し、mask処理を施し、単語レベルの長さにsortされたtensorとその長さを返す。
        """
        sw_ids = sw_features.get("sw_ids")
        batch_size, sw_seq_length = sw_ids.shape

        # calc forward lstm
        lengths = sw_ids.eq(0).lt(1).sum(dim=1)
        sorted_lengths, perm_ix = lengths.sort(descending=True)
        embeds = self.dropout(self.embeddings(sw_ids[perm_ix]))
        pack_input = pack_padded_sequence(embeds, sorted_lengths, batch_first=True)
        hidden = None
        lstm_out, lstm_hidden = self.lstm(pack_input, hidden)
        lstm_out, sorted_lengths = pad_packed_sequence(lstm_out, batch_first=True)

        # mask process
        fmasks = sw_features.get("fmasks")[perm_ix] # (B, L_sw)
        bmasks = sw_features.get("bmasks")[perm_ix]
        out_seq_lengths = fmasks.sum(dim=1) # sw_lengths順でのperm後のfmaskした後の単語レベルの長さ(bmaskの場合でも同じ)
        fmasks = fmasks.unsqueeze(2).expand(batch_size, sw_seq_length, self.sw_hidden_dim // 2).byte()  # fmaskをsw_hidden_dimに引き延ばす。
        bmasks = bmasks.unsqueeze(2).expand(batch_size, sw_seq_length, self.sw_hidden_dim // 2).byte()  # bmaskをsw_hidden_dimに引き延ばす。

        f_lstm_out = lstm_out[:, :, :self.sw_hidden_dim // 2] # (B, L_sw, Ec // 2)
        b_lstm_out = lstm_out[:, :, self.sw_hidden_dim // 2:]

        f_out, b_out = [], []
        for i in range(batch_size):
            f_out.append(f_lstm_out[i].masked_select(fmasks[i]).view(-1, self.sw_hidden_dim // 2))
            b_out.append(b_lstm_out[i].masked_select(bmasks[i]).view(-1, self.sw_hidden_dim // 2))
        f_padded_out = pad_sequence(f_out, batch_first=True)
        b_padded_out = pad_sequence(b_out, batch_first=True)

        # マスクした後のword_lengthでの並び順を直す。(sw lengthとword lengthの順番が一致しないことがある。)
        out_sorted_lengths, out_perm_ix = out_seq_lengths.sort(descending=True)  # forward, backward 共通
        return torch.cat([f_padded_out[out_perm_ix], b_padded_out[out_perm_ix]], dim=2), out_sorted_lengths
