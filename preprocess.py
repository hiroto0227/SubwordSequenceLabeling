from itertools import chain
from typing import List

from chem_sentencepiece.chem_sentencepiece import ChemSentencePiece
from gensim.corpora import Dictionary
import torch
from torch import LongTensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from label import UNKNOWN, PADDING


def get_word_features(word_documents: List[List[str]], word_dic: Dictionary, gpu: bool):
    # idnize
    word_ids = [[word_dic.token2id.get(word, word_dic.token2id.get(UNKNOWN)) for word in document] for document in word_documents]
    # padding
    padded_tensor = pad_sequence([LongTensor(x) for x in word_ids], batch_first=True)
    masks = padded_tensor.eq(0).le(0) # not equall
    if gpu:
        return {"word_ids": padded_tensor.cuda(),
                "masks": masks.cuda()}
    else:
        return {"word_ids": padded_tensor, 
                "masks": masks}


def get_char_features(word_documents: List[List[str]], char_dic: Dictionary, gpu: bool):
    # idnize and padding
    max_seq_length: int = max([len(document) for document in word_documents])
    max_char_length: int = max([max([len(word) for word in document]) for document in word_documents])
    pad_id = char_dic.token2id.get(PADDING, char_dic.token2id.get(UNKNOWN))
    padded_char_ids = torch.zeros((len(word_documents), max_seq_length, max_char_length), dtype=torch.long).fill_(pad_id)
    for batch_ix, word_document in enumerate(word_documents):
        for seq_ix, word in enumerate(word_document):
            for char_ix, char in enumerate(word):
                padded_char_ids[batch_ix, seq_ix, char_ix] = char_dic.token2id.get(char, char_dic.token2id.get(UNKNOWN))
    if gpu:
        return {"char_ids": padded_char_ids.cuda()}
    else:
        return {"char_ids": padded_char_ids}


def get_sw_features(word_documents: List[List[str]], sw_dic: Dictionary, sp: ChemSentencePiece, gpu: bool):
    sw_documents, sw_ids = [], []
    for word_document in word_documents:
        sws = [sp.tokenize(word) for word in word_document]
        sw_documents.append(list(chain.from_iterable(sws)))
        sw_ids.append([sw_dic.token2id.get(sw, sw_dic.token2id.get(UNKNOWN)) for sw in chain.from_iterable(sws)])
    fmasks = get_fmasks(sw_documents)
    padded_fmask_tensor = pad_sequence([LongTensor(x) for x in fmasks], batch_first=True)
    bmasks = get_bmasks(sw_documents)
    padded_bmask_tensor = pad_sequence([LongTensor(x) for x in bmasks], batch_first=True)
    # padding
    padded_tensor = pad_sequence([LongTensor(x) for x in sw_ids], batch_first=True)
    if gpu:
        return {"sw_ids": padded_tensor.cuda(),
                "fmasks": padded_fmask_tensor.cuda(),
                "bmasks": padded_bmask_tensor.cuda()}
    else:
        return {"sw_ids": padded_tensor,
                "fmasks": padded_fmask_tensor,
                "bmasks": padded_bmask_tensor}


def get_label_features(label_documents: List[List[str]], label_dic: Dictionary, gpu: bool):
    label_ids = [[label_dic.token2id.get(l, label_dic.token2id.get(PADDING)) for l in label_document]
                  for label_document in label_documents]
    padded_tensor = pad_sequence([torch.LongTensor(x) for x in label_ids], batch_first=True)
    if gpu:
        return {"label_ids": padded_tensor.cuda()}
    else:
        return {"label_ids": padded_tensor}


def get_fmasks(sw_documents):
    f_masks = []
    for sw_document in sw_documents:
        f_mask = [0 for i in range(len(sw_document))]
        for i, sw in enumerate(sw_document):
            if sw.startswith("▁"):
                f_mask[i] = 1
        f_masks.append(f_mask)
    return f_masks


def get_bmasks(sw_documents):
    b_masks = []
    for sw_document in sw_documents:
        is_pre_first = True
        b_mask = [0 for i in range(len(sw_document))]
        for i, sw in enumerate(sw_document[::-1]):
            if is_pre_first:
                b_mask[i] = 1
            is_pre_first = sw.startswith("▁")
        b_masks.append(b_mask[::-1])
    return b_masks
