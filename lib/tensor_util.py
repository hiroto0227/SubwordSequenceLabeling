
import torch
import numpy as np


def get_embedding_tensor_from_id_tensor(id_tensor, word2vec: np.ndarray):
    embed = torch.zeros(id_tensor.shape[0], id_tensor.shape[1], word2vec.shape[1])
    for batch_ix, tensor in enumerate(id_tensor):
        for j, word_id in enumerate(tensor):
            embed[batch_ix, j] = torch.from_numpy(word2vec[word_id])
    return embed
