import sys

import numpy as np
import torch
from tqdm import tqdm

from lib.label import PADDING
from lib.charset import CHARSET
from lib.function import normalize_word


def batchify_with_label(input_batch_list, gpu, if_train=True, label=True):
    """
        input: list of words, chars and labels, various length. [[words, chars, labels],[words, chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
            sws_list: subword ids for one senteces, various length. (batch_size, subword_num, sw_length)
            sw_fmasks_list: 1 if sw at word end else 0. (batch_size, subword_num, sw_length)
            sw_bmasks_list: 1 if sw at word start else 0. (batch_size, subword_num, sw_length)
            labels: label ids for one sentence. (batch_size, sent_len)
        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            sw_seqs_tensor: (sw_num, batch_size, max_sw_seq_len) List[Variable]
            sw_seqs_lengths: (sw_num, batch_size, max_sw_seq_len, 1) List[Tensor]
            sw_seqs_recover: (sw_num, batch_size, max_sw_seq_len, 1) List[Tensor] recover sw sequence order
            sw_fmasks_list: (sw_num, batch_size, max_sw_seq_len)
            sw_bmasks_list: (sw_num, batch_size, max_sw_seq_len)
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
    """
    ### initialize
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    chars = [sent[1] for sent in input_batch_list]
    sw_num = len(input_batch_list[0][2])
    sws_list = [sent[2] for sent in input_batch_list]
    sw_fmasks_list = [sent[3] for sent in input_batch_list]
    sw_bmasks_list = [sent[4] for sent in input_batch_list]
    if label:
        labels = [sent[5] for sent in input_batch_list]
    else:
        labels = words
    
    ### words and labels
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max().item()  #(batch_size, seq_length, 1)
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad =  if_train).long()  #(batch_size, max_seq_len)
    label_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad =  if_train).long()  #(batch_size, max_seq_len)
    mask = torch.zeros((batch_size, max_seq_len), requires_grad =  if_train).byte()  #(batch_size, max_seq_len)
    word_pad_lengths_for_sw = []
    for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
        seqlen = seqlen.item()
        word_pad_lengths_for_sw.append(max_seq_len - seqlen)
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1]*seqlen)
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)  # word_seq_lengthが大きい順に並び替え。
    word_seq_tensor = word_seq_tensor[word_perm_idx]

    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    ### deal with char
    pad_chars = [chars[idx] + [[0]] * (max_seq_len-len(chars[idx])) for idx in range(len(chars))]  # (batch_size, max_seq_len)
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]  # (batch_size, max_seq_len, 1)
    max_word_len = max(map(max, length_list))
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad =  if_train).long()
    char_seq_lengths = torch.LongTensor(length_list) #(batch_size, max_seq_len, 1)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len,-1)  #(batch_size*max_seq_len, -1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len,)  #(batch_size*max_seq_len, 1)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)

    ### deal with subword
    sw_seqs_tensor, sw_seqs_lengths, sw_seqs_recover, sw_seqs_fmask, sw_seqs_bmask = [], [], [], [], []
    sws_list_swapped = [[] for _ in range(sw_num)]
    sw_fmasks_list_swapped = [[] for _ in range(sw_num)]
    sw_bmasks_list_swapped = [[] for _ in range(sw_num)]
    # transpose(1, 0)
    for sw_ix in range(sw_num):   
        sws_list_swapped[sw_ix] = [sws_list[batch_ix][sw_ix] for batch_ix in range(batch_size)]  # (sw_num, batch_size, variable)
        sw_fmasks_list_swapped[sw_ix] = [sw_fmasks_list[batch_ix][sw_ix] for batch_ix in range(batch_size)]  # (sw_num, batch_size, variable)
        sw_bmasks_list_swapped[sw_ix] = [sw_bmasks_list[batch_ix][sw_ix] for batch_ix in range(batch_size)]  # (sw_num, batch_size, variable)
    for sw_ix in range(sw_num):
        sws = sws_list_swapped[sw_ix]
        # word の paddingも考慮に入れる。
        sw_seq_lengths = torch.LongTensor(np.array(list(map(len, sws)) + np.array(word_pad_lengths_for_sw))) #(batch_size, 1)
        max_sw_seq_len = sw_seq_lengths.max()
        sw_seq_tensor = torch.zeros((batch_size, max_sw_seq_len), requires_grad =  if_train).long()
        sw_seq_fmask = torch.zeros((batch_size, max_sw_seq_len)).byte()
        sw_seq_bmask = torch.zeros((batch_size, max_sw_seq_len)).byte()
        for idx, (sw, fmask, bmask) in enumerate(zip(sws, sw_fmasks_list_swapped[sw_ix], sw_bmasks_list_swapped[sw_ix])):
            seqlen = len(sw)
            sw_seq_tensor[idx, :seqlen] = torch.LongTensor(sw) 
            sw_seq_fmask[idx, :seqlen] = torch.ByteTensor(fmask)
            sw_seq_fmask[idx, seqlen:seqlen+word_pad_lengths_for_sw[idx]] = 1  # swのwordに対してのpadding用
            sw_seq_bmask[idx, :seqlen] = torch.ByteTensor(bmask)
            sw_seq_bmask[idx, seqlen:seqlen+word_pad_lengths_for_sw[idx]] = 1  # swのwordに対してのpadding用
        sw_seq_lengths, sw_perm_idx = sw_seq_lengths.sort(0, descending=True)
        sw_seq_tensor = sw_seq_tensor[sw_perm_idx]
        sw_seq_fmask = sw_seq_fmask[sw_perm_idx]
        sw_seq_bmask = sw_seq_bmask[sw_perm_idx]
        _, sw_seq_recover = sw_perm_idx.sort(0, descending=False)
        
        sw_seqs_tensor.append(sw_seq_tensor)
        sw_seqs_lengths.append(sw_seq_lengths)
        sw_seqs_recover.append(sw_seq_recover)
        sw_seqs_fmask.append(sw_seq_fmask)
        sw_seqs_bmask.append(sw_seq_bmask)

    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        sw_seqs_tensor = [sw_seqs_tensor[idx].cuda() for idx in range(sw_num)]
        sw_seqs_lengths = [sw_seqs_lengths[idx].cuda() for idx in range(sw_num)]
        sw_seqs_recover = [sw_seqs_recover[idx].cuda() for idx in range(sw_num)]
        sw_seqs_fmask = [sw_seqs_fmask[idx].cuda() for idx in range(sw_num)]
        sw_seqs_bmask = [sw_seqs_bmask[idx].cuda() for idx in range(sw_num)]
        mask = mask.cuda()

    return (word_seq_tensor, 
            word_seq_lengths, 
            word_seq_recover, 
            char_seq_tensor, 
            char_seq_lengths, 
            char_seq_recover, 
            sw_seqs_tensor, 
            sw_seqs_lengths, 
            sw_seqs_recover, 
            sw_seqs_fmask, 
            sw_seqs_bmask, 
            label_seq_tensor, 
            mask)
