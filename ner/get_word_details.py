import argparse
from datetime import datetime
import gc
from itertools import chain
import os
import pprint
import random
import re
import sys
import time
from typing import List
from collections import Counter

from gensim.corpora import Dictionary
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append("./")
from chem_sentencepiece.chem_sentencepiece import ChemSentencePiece
from config import config_dics
from label import UNKNOWN, PADDING, NUM, START, END
from utils import load_pretrain_embeddings, build_pretrain_embeddings, normalize_number_word
from model.seqmodel import SeqModel
from ner.evaluate import evaluate
from ner.train import load_seq_data, predict_check
from preprocess import get_char_features, get_sw_features, get_word_features, get_label_features


seed_num = 42
random.seed(seed_num)
np.random.seed(seed_num)
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)
torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--out", default="", type=str)
    parser.add_argument("--model", default="", type=str)
    args = parser.parse_args()

    print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
    config_dic = config_dics[args.config]
    if not args.model:
        args.model = os.path.join(config_dic.get("ner_model_dir"), f"{args.config}.model")

    # load sentence piece
    if config_dic.get("sp_path"):
        sp = ChemSentencePiece()
        sp.load(config_dic.get("sp_path"))
    else:
        sp = None

    # load train data
    print("=========== Load train data ===========")
    train_word_documents, train_label_documents = load_seq_data(os.path.join(config_dic.get("ner_input_dir"), "train.bioes"), config_dic.get("number_normalize"))
    valid_word_documents, valid_label_documents = load_seq_data(os.path.join(config_dic.get("ner_input_dir"), "valid.bioes"), config_dic.get("number_normalize"))
    test_word_documents, test_label_documents = load_seq_data(os.path.join(config_dic.get("ner_input_dir"), "test.bioes"), config_dic.get("number_normalize"))

    train_char_documents = [[[char for char in word] for word in document] for document in train_word_documents] # Document数 x 文字数
    valid_char_documents = [[[char for char in word] for word in document] for document in valid_word_documents]
    test_char_documents = [[[char for char in word] for word in document] for document in test_word_documents]

    if config_dic.get("sp_path"):
        train_sw_documents = get_sw_documents(train_word_documents, sp)
        valid_sw_documents = get_sw_documents(valid_word_documents, sp)
        test_sw_documents = get_sw_documents(test_word_documents, sp)

    # load vocabulary
    print("=========== Build vocabulary ===========")
    if os.path.exists(os.path.join(config_dic.get("vocab_dir"), f"{args.config}.word.dic")):
        word_dic = Dictionary.load(os.path.join(config_dic.get("vocab_dir"), f"{args.config}.word.dic"))
        char_dic = Dictionary.load(os.path.join(config_dic.get("vocab_dir"), f"{args.config}.char.dic"))
        if sp:
            sw_dic = Dictionary.load(os.path.join(config_dic.get("vocab_dir"), f"{args.config}.sw.dic"))
        else:
            sw_dic = None
    else:
        special_token_dict = {PADDING: 0, UNKNOWN: 1, START: 2, END: 3}
        word_dic = Dictionary()
        word_dic.token2id = special_token_dict
        char_dic = Dictionary()
        char_dic.token2id = special_token_dict
        if sp:
            sw_dic = Dictionary()
            sw_dic.token2id = special_token_dict
        else:
            sw_dic = None
    label_dic = Dictionary(train_label_documents)
    label_dic.patch_with_special_tokens({PADDING: 0})
    label_dic.id2token = {_id: label for label, _id in label_dic.token2id.items()}

    # add vocabulary
    word_dic.add_documents(train_word_documents)
    char_dic.add_documents(list(chain.from_iterable(train_char_documents)))
    if sp:
        sw_dic.add_documents(train_sw_documents)

    # load GloVe
    if config_dic.get("glove_path"):
        print("============== Load Pretrain Word Embeddings ================")
        word2vec = load_pretrain_embeddings(config_dic.get("glove_path"), emb_dim=config_dic.get("word_emb_dim"))
        pretrain_embeddings = build_pretrain_embeddings(word2vec, word_dic, emb_dim=config_dic.get("word_emb_dim"))
    else:
        pretrain_embeddings = None

    if config_dic.get("sp_path"):
        seq_model = SeqModel(config_dic, len(word_dic.token2id), len(char_dic.token2id), len(sw_dic.token2id), len(label_dic.token2id), pretrain_embeddings)
    else:
        seq_model = SeqModel(config_dic, len(word_dic.token2id), len(char_dic.token2id), None, len(label_dic.token2id), pretrain_embeddings)
    
    # modelをload
    print(f"Load model {args.model} !")
    seq_model.load_state_dict(torch.load(args.model))

    word_dic.id2token = {v: k for k, v in word_dic.token2id.items()}
    char_dic.id2token = {v: k for k, v in char_dic.token2id.items()}
    if config_dic.get("sp_path"):
        sw_dic.id2token = {v: k for k, v in sw_dic.token2id.items()}


    ################ valid predict check #####################
    print("============== Predict Check==========")
    true_seqs, pred_seqs, word_seqs, char_seqs = [], [], [], []
    right_token, total_token = 0, 0
    batch_size = config_dic.get("ner_batch_size")
    batch_steps = len(valid_word_documents) // batch_size + 1
    random_ids = list(range(len(valid_word_documents)))
    seq_model.eval()
    for batch_i in tqdm(range(batch_steps)):
        batch_ids = random_ids[batch_i * batch_size: (batch_i + 1) * batch_size]
        batch_word_documents = [valid_word_documents[i] for i in batch_ids]
        batch_label_documents = [valid_label_documents[i] for i in batch_ids]

        valid_word_features = get_word_features(batch_word_documents, word_dic, config_dic.get("gpu"))
        valid_char_features = get_char_features(batch_word_documents, char_dic, config_dic.get("gpu"))
        if config_dic.get("sp_path"):
            valid_sw_features = get_sw_features(batch_word_documents, sw_dic, sp, config_dic.get("gpu"))
        else:
            valid_sw_features = None
        valid_label_features = get_label_features(batch_label_documents, label_dic, config_dic.get("gpu"))
        valid_tag_seq = seq_model.forward(valid_word_features, valid_char_features, valid_sw_features)
        rt, tt = predict_check(valid_tag_seq, valid_label_features.get("label_ids"), valid_word_features.get("masks"))
        right_token += rt
        total_token += tt
        ################ evaluate by precision, recall and fscore ###################
        masks = valid_word_features.get("masks")
        word_seqs.extend([word_dic.id2token.get(int(word_id)) for word_id in valid_word_features.get("word_ids").masked_select(masks)])

    #     char_ids = valid_char_features.get("char_ids")
    #     char_masks = masks.unsqueeze(-1).expand(char_ids.shape)
    #     chars = []
    #     for i, char_id in enumerate(char_ids.masked_select(char_masks)):
    #         if i != 0 and i % char_ids.shape[-1] == 0:
    #             char_seqs.append(chars)
    #             chars = []
    #         if not char_id == char_dic.token2id[PADDING]:
    #             chars.append(char_dic.id2token.get(int(char_id)))
    #     char_seqs.append(chars)

        true_seqs.extend([label_dic.id2token[int(label_id)] for label_id in valid_label_features.get("label_ids").masked_select(masks)])
        pred_seqs.extend([label_dic.id2token[int(label_id)] for label_id in valid_tag_seq.masked_select(masks)])

    precision, recall, fscore = evaluate(true_seqs, pred_seqs)
    print(f"Right: {right_token}, Total: {total_token}, Accuracy: {right_token / total_token:.4f}")
    print(f"Precision: {precision}, Recall: {recall}, Fscore: {fscore}")


    true_entities = []
    true_entity = ""
    for i, (word, label) in enumerate(zip(word_seqs, true_seqs)):
        if label == "O":
            pass
        elif label == "B-CHEM" and not true_entity:
            true_entity += word
        elif label == "I-CHEM" and true_entity:
            true_entity += (" " + word)
        elif label == "E-CHEM" and true_entity:
            true_entity += (" " + word)
            true_entities.append(true_entity)
            true_entity = ""
        elif label == "S-CHEM":
            true_entities.append(word)
            true_entity = ""
        else:
            # ありえないやつも考えなくてはいけない。
            print("Warning!! The combination of the labels is not compatible.")
            true_entity = ""
        pre_label = label
    true_entity_counter = Counter(true_entities)

    pred_entities = []
    pred_entity = ""
    for i, (word, label) in enumerate(zip(word_seqs, pred_seqs)):
        if label == "O":
            pass
        elif label == "B-CHEM" and not pred_entity:
            pred_entity += word
        elif label == "I-CHEM" and pred_entity:
            pred_entity += (" " + word)
        elif label == "E-CHEM" and pred_entity:
            pred_entity += (" " + word)
            pred_entities.append(pred_entity)
            pred_entity = ""
        elif label == "S-CHEM":
            pred_entities.append(word)
            pred_entity = ""
        else:
            # ありえないやつも考えなくてはいけない。
            print("Warning!! The combination of the labels is not compatible.")
            pred_entity = ""
        pre_label = label
    pred_entity_counter = Counter(pred_entities)

    results = {"entity": [], "true_count": [], "pred_count": [], "tp": [], "fp": [], "fn": []}
    for entity in set(true_entities) | set(pred_entities):
        pred_count = pred_entity_counter.get(entity, 0)
        true_count = true_entity_counter.get(entity, 0)
        results["entity"].append(entity)
        results["true_count"].append(true_count)
        results["pred_count"].append(pred_count)
        results["tp"].append(min(true_count, pred_count))
        results["fp"].append(max(true_count - pred_count, 0))
        results["fn"].append(max(pred_count - true_count, 0))

df = pd.DataFrame(results).sort_values("true_count", ascending=False)
df.to_csv(f"{args.out + args.config}.csv")
print(f"Save! {args.out + args.config}")