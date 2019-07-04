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
from ner.train import load_seq_data, predict_check, get_sw_documents
from preprocess import get_char_features, get_sw_features, get_word_features, get_label_features


seed_num = 42
random.seed(seed_num)
np.random.seed(seed_num)
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)
torch.backends.cudnn.deterministic = True


def get_entity_from_labels(word_seqs: List[str], label_seqs: List[str]) -> List[str]:
    """BIOES法でエンコードされた単語とラベルから、抽出したEntityを獲得。
    """
    entities = []
    entity = ""
    for i, (word, label) in enumerate(zip(word_seqs, label_seqs)):
        if label == "O":
            pass
        elif label == "B-CHEM" and not entity:
            entity += word
        elif label == "I-CHEM" and entity:
            entity += (" " + word)
        elif label == "E-CHEM" and entity:
            entity += (" " + word)
            entities.append(entity)
            entity = ""
        elif label == "S-CHEM":
            entities.append(word)
            entity = ""
        else:
            # ありえないやつも考えなくてはいけない。
            print("Warning!! The combination of the labels is not compatible.")
            entity = ""
        pre_label = label
    return entities


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
    
    if not args.out:
        args.out = args.config

    # load sentence piece
    sps: dict = {}
    for sp_key, sp_path in config_dic["sp_path"].items():
        sps[sp_key] = ChemSentencePiece.load(sp_path)

    # load train data
    print("=========== Load train data ===========")
    train_word_documents, train_label_documents = load_seq_data(os.path.join(config_dic.get("ner_input_dir"), "train.bioes"), config_dic.get("number_normalize"))
    valid_word_documents, valid_label_documents = load_seq_data(os.path.join(config_dic.get("ner_input_dir"), "valid.bioes"), config_dic.get("number_normalize"))
    test_word_documents, test_label_documents = load_seq_data(os.path.join(config_dic.get("ner_input_dir"), "test.bioes"), config_dic.get("number_normalize"))

    train_char_documents = [[[char for char in word] for word in document] for document in train_word_documents] # Document数 x 文字数
    valid_char_documents = [[[char for char in word] for word in document] for document in valid_word_documents]
    test_char_documents = [[[char for char in word] for word in document] for document in test_word_documents]

    train_sw_documents_dicts = {}
    valid_sw_documents_dicts = {}
    test_sw_documents_dicts = {}
    for sp_key, sp in sps.items():
        train_sw_documents_dicts[sp_key] = get_sw_documents(train_word_documents, sp)
        valid_sw_documents_dicts[sp_key] = get_sw_documents(valid_word_documents, sp)
        test_sw_documents_dicts[sp_key] = get_sw_documents(test_word_documents, sp)

    # load vocabulary
    print("=========== Build vocabulary ===========")
    if os.path.exists(os.path.join(config_dic.get("vocab_dir"), f"{args.config}.word.dic")):
        word_dic = Dictionary.load(os.path.join(config_dic.get("vocab_dir"), f"{args.config}.word.dic"))
        char_dic = Dictionary.load(os.path.join(config_dic.get("vocab_dir"), f"{args.config}.char.dic"))
        sw_dicts = {}
        for sp_key, sp in sps.items():
            sw_dicts[sp_key] = Dictionary.load(os.path.join(config_dic.get("vocab_dir"), f"{args.config}.{sp_key}.dic"))
    else:
        special_token_dict = {PADDING: 0, UNKNOWN: 1, START: 2, END: 3}
        word_dic = Dictionary()
        word_dic.token2id = special_token_dict
        char_dic = Dictionary()
        char_dic.token2id = special_token_dict
        sw_dicts = {}
        for sp_key, sp in sps.items():
            _dic = Dictionary()
            _dic.token2id = special_token_dict
            sw_dicts[sp_key] = _dic
    label_dic = Dictionary(train_label_documents)
    label_dic.patch_with_special_tokens({PADDING: 0})
    label_dic.id2token = {_id: label for label, _id in label_dic.token2id.items()}

    # add vocabulary
    word_dic.add_documents(train_word_documents)
    char_dic.add_documents(list(chain.from_iterable(train_char_documents)))
    for sp_key, train_sw_documents in train_sw_documents_dicts.items():
        sw_dicts[sp_key].add_documents(train_sw_documents)

    # load GloVe
    if config_dic.get("glove_path"):
        print("============== Load Pretrain Word Embeddings ================")
        word2vec = load_pretrain_embeddings(config_dic.get("glove_path"), emb_dim=config_dic.get("word_emb_dim"))
        pretrain_embeddings = build_pretrain_embeddings(word2vec, word_dic, emb_dim=config_dic.get("word_emb_dim"))
    else:
        pretrain_embeddings = None

    # initialize Model
    seq_model = SeqModel(
        config_dic, 
        len(word_dic.token2id), 
        len(char_dic.token2id), 
        [len(sw_dic.token2id) for sw_dic in sw_dicts.values()], 
        len(label_dic.token2id), 
        pretrain_embeddings
    )

    # load Model
    print(f"Load model {args.model} !")
    seq_model.load_state_dict(torch.load(args.model))

    word_dic.id2token = {v: k for k, v in word_dic.token2id.items()}
    char_dic.id2token = {v: k for k, v in char_dic.token2id.items()}
    for sp_key, sp in sps.items():
        sw_dicts[sp_key].id2token = {v: k for k, v in sw_dicts[sp_key].token2id.items()}

    ################ test predict check #####################
    print("============== Predict Check==========")
    true_seqs, pred_seqs, word_seqs, char_seqs = [], [], [], []
    right_token, total_token = 0, 0
    batch_size = config_dic.get("ner_batch_size")
    batch_steps = len(test_word_documents) // batch_size + 1
    random_ids = list(range(len(test_word_documents)))
    seq_model.eval()
    for batch_i in tqdm(range(batch_steps)):
        batch_ids = random_ids[batch_i * batch_size: (batch_i + 1) * batch_size]
        batch_word_documents = [test_word_documents[i] for i in batch_ids]
        batch_label_documents = [test_label_documents[i] for i in batch_ids]

        test_word_features = get_word_features(batch_word_documents, word_dic, config_dic.get("gpu"))
        test_char_features = get_char_features(batch_word_documents, char_dic, config_dic.get("gpu"))
        
        test_sw_features_list = []
        for sp_key, sp in sps.items():
            test_sw_features_list.append(get_sw_features(batch_word_documents, sw_dicts[sp_key], sp, config_dic.get("gpu")))
        
        test_label_features = get_label_features(batch_label_documents, label_dic, config_dic.get("gpu"))
        test_tag_seq = seq_model.forward(test_word_features, test_char_features, test_sw_features_list)
        rt, tt = predict_check(test_tag_seq, test_label_features.get("label_ids"), test_word_features.get("masks"))
        right_token += rt
        total_token += tt
        ################ evaluate by precision, recall and fscore ###################
        masks = test_word_features.get("masks")
        word_seqs.extend([word for word_document in batch_word_documents for word in word_document])

        true_seqs.extend([label_dic.id2token[int(label_id)] for label_id in test_label_features.get("label_ids").masked_select(masks)])
        pred_seqs.extend([label_dic.id2token[int(label_id)] for label_id in test_tag_seq.masked_select(masks)])

    precision, recall, fscore = evaluate(true_seqs, pred_seqs)
    print(f"Right: {right_token}, Total: {total_token}, Accuracy: {right_token / total_token:.4f}")
    print(f"Precision: {precision}, Recall: {recall}, Fscore: {fscore}")


    true_entities = get_entity_from_labels(word_seqs, true_seqs)
    true_entity_counter = Counter(true_entities)

    pred_entities = get_entity_from_labels(word_seqs, pred_seqs)
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
    # 未知語フラグを作成。
    all_word_set = word_dic.token2id.keys()
    df["Unknown"] = df.entity.map(lambda x: any([w not in all_word_set for w in x.split(" ")]))
    # Subword分割結果を作成。
    for sp_id, sp in sps.items():
        df[sp_id] = df.entity.map(lambda x: [" ".join(sp.tokenize(word)) for word in x.split(" ")])
    
    df.to_csv(f"{args.out}.csv")
    print(f"Save! {args.out}.csv")