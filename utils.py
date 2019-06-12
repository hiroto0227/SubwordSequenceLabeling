import sys
import re

import numpy as np
import torch
from gensim.corpora import Dictionary


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


def normalize_number_word(word: str):
    return re.sub("\d", "0", word)


def build_pretrain_embeddings(word2vec, word_dic: Dictionary, emb_dim: int) -> np.ndarray:
    pretrain_embeddings: np.ndarray = np.empty([len(word_dic.token2id), emb_dim])
    perfect_match, case_match, not_match = 0, 0, 0
    scale = np.sqrt(3.0 / emb_dim)
    for i, (word, index) in enumerate(word_dic.token2id.items()):
        if word in word2vec.keys():
            #pretrain_embeddings[index] = norm2one(np.array(word2vec[word], dtype=np.float32))
            pretrain_embeddings[index] = np.array(word2vec[word], dtype=np.float32)
            perfect_match += 1
        elif word.lower() in word2vec.keys():
            #pretrain_embeddings[index] = norm2one(np.array(word2vec[word.lower()], dtype=np.float32))
            pretrain_embeddings[index] = np.array(word2vec[word.lower()], dtype=np.float32)
            case_match += 1
        else:
            pretrain_embeddings[index] = np.random.uniform(-scale, scale, (emb_dim,))
            not_match += 1
    print(f"PerfectMatch: {perfect_match}, CaseMatch: {case_match}, NotMatch: {not_match}.")
    return pretrain_embeddings


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square


def load_pretrain_embeddings(embedding_path: str, emb_dim: int):
    word2vec = {}
    with open(embedding_path, 'r') as file:
        for i, line in enumerate(file):
            splited_line = line.split()
            word2vec["".join(splited_line[:len(splited_line) - emb_dim])] = np.array(splited_line[len(splited_line) - emb_dim:], dtype=np.float32)
    return word2vec


SPECIAL_CHARS = "|".join([chr(i) for i in range(127, 161)])
SPECIAL_BLANKS = "|".join([chr(i) for i in range(8191, 8207)])


def tokenize(text):
    """textをtoken単位に分割したリストを返す。"""
    text = re.sub(f"({SPECIAL_CHARS}|{SPECIAL_BLANKS})", " ", text)
    tokens = re.split("( | |\t|\n|\d+\.\d+|…|\'|\"|·|~|↔|•|\!|@|#|\$|%|\^|&|\*|-|=|_|\+|ˉ|\(|\)|\[|\]|\{|\}|;|‘|:|“|,|\/|<|>|×|>|<|≤|≥|↑|↓|→|¬|®|•|′|°|~|≈|\?|Δ|÷|≠|‘|’|“|”|§|£|€|™|⋅|-|⁺)", text)
    tokens = [token.replace(' ',  '') for token in tokens]
    return list(filter(None, tokens))


if __name__ == '__main__':
    a = np.arange(9.0)
    print(a)
    print(norm2one(a))
