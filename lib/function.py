import sys
import re

import numpy as np
import torch


cammel_re = re.compile("^[A-Z][a-z0-9]+$")

def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word

def cammel_normalize_word(word):
    if cammel_re.match(word):
        return word.lower()
    return word

def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    alphabet_size = word_alphabet.size()
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim], dtype=np.float)
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_alphabet.iteritems():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index, :] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index, :] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrain_emb[np.isnan(pretrain_emb)] = np.random.uniform(-scale, scale, [1])
    pretrained_size = len(embedd_dict)
    
    print("""Embedding:     
        pretrain word:{}, 
        prefect match:{}, 
        case_match:{}, 
        oov:{}, 
        oov%%:{}
        """.format(pretrained_size, perfect_match, case_match, not_match, ((not_match+0.)/alphabet_size)))

    return pretrain_emb, embedd_dim

def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square

def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            #else:
            #     assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            try:
                embedd[:] = tokens[1:]
                if sys.version_info[0] < 3:
                    first_col = tokens[0].decode('utf-8')
                else:
                    first_col = tokens[0]
                embedd_dict[first_col] = embedd
            except ValueError:
                print("Value Error: {}".format(line))
    return embedd_dict, embedd_dim


def tokenize(text):
    """textをtoken単位に分割したリストを返す。"""
    tokens = re.split("( | |\xa0|\t|\n|\d+|…|\'|\"|·|~|↔|•|\!|@|#|\$|%|\^|&|\*|-|=|_|\+|ˉ|\(|\)|\[|\]|\{|\}|;|‘|:|“|,|\.|\/|<|>|×|>|<|≤|≥|↑|↓|→|¬|®|•|′|°|~|≈|\?|Δ|÷|≠|‘|’|“|”|§|£|€|™|⋅|-|\u2000|⁺|\u2009)", text)
    tokens = [token.replace(' ',  '') for token in tokens]
    return list(filter(None, tokens))


if __name__ == '__main__':
    a = np.arange(9.0)
    print(a)
    print(norm2one(a))
