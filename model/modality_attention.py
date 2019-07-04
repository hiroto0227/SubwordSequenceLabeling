
import torch
import torch.nn as nn


class ModalityAttention(nn.Module):
    """https://www.aclweb.org/anthology/N18-1078 
    Section3.3: ModalityAttention参照
    """
    def __init__(self, modality_num: int, embedding_size: int, dropout: float):
        super().__init__()
        self.modality_dim = embedding_size * modality_num
        self.modality_num = modality_num
        self.embedding_size = embedding_size
        self.drop = nn.Dropout(dropout)
        self.w_m = nn.Linear(self.modality_dim, self.modality_dim, bias=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        _atts = self.sigmoid(self.w_m(self.drop(x)))  #(B, L, Mdim)
        _sum_exp_att = sum([torch.exp(_att) for _att in _atts.split(self.embedding_size, dim=2)])
        _alphas = torch.cat([torch.exp(_att) / _sum_exp_att for _att in _atts.split(self.embedding_size, dim=2)], dim=2)
        _x_ts = []
        for _x, _alpha in zip(x.split(self.embedding_size, dim=2), _alphas.split(self.embedding_size, dim=2)):
            _x_ts.append(_alpha * _x)
        context_vector = sum(_x_ts)
        
        assert context_vector.shape == (x.shape[0], x.shape[1], x.shape[2] / self.modality_num), \
            f"The result shape is Not Valid. {context_vector.shape}"
        
        return context_vector