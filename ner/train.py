import argparse
import gc
from itertools import chain
import os
import random
import re
import sys
import time
from typing import List

from gensim.corpora import Dictionary
import numpy as np
import torch
from tqdm import tqdm

sys.path.append("./")
from chem_sentencepiece.chem_sentencepiece import ChemSentencePiece
from config import config_dic
from label import UNKNOWN, PADDING, NUM
from model.seqmodel import SeqModel
from evaluate import evaluate
from preprocess import get_char_features, get_sw_features, get_word_features, get_label_features

seed_num = 0
random.seed(0)

def load_seq_data(file_path: str, number_normalize=True) -> (List[List[str]], List[List[str]]):
    """下記のフォーマットのデータをWordとLabelのdocumentsの型にして返す。
    word label
    word label
    word label
    """
    with open(file_path, "rt", encoding="utf-8") as f:
        lines: List[str] = f.read().split("\n")
    word_documents, label_documents = [], []
    word_document, label_document = [], []
    num_re = re.compile("\d+\.*\d*")
    for line in tqdm(lines):
        if len(line) > 0:
            word, label = line.split(" ")
            if number_normalize and num_re.match(word):
                word = NUM
            word_document.append(word)
            label_document.append(label)
        elif len(word_document):
            word_documents.append(word_document)
            label_documents.append(label_document)
            word_document, label_document = [], []
    return word_documents, label_documents


def get_sw_documents(word_documents: List[List[str]]) -> List[List[str]]:
    sw_documents = []
    for word_document in word_documents:
        sws = [sp.tokenize(word) for word in word_document]
        sw_documents.append(list(chain.from_iterable(sws)))
    return sw_documents


def predict_check(pred_variable, gold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    # print("right: %s, total: %s"%(right_token, total_token))
    return right_token, total_token


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    # load sentence piece
    sp = ChemSentencePiece()
    sp.load(config_dic.get("sp_path"))

    # load train data
    print("=========== Load train data ===========")
    train_word_documents, train_label_documents = load_seq_data(os.path.join(config_dic.get("ner_input_dir"), "train.bioes"), config_dic.get("number_normalize"))
    valid_word_documents, valid_label_documents = load_seq_data(os.path.join(config_dic.get("ner_input_dir"), "valid.bioes"), config_dic.get("number_normalize"))
    test_word_documents, test_label_documents = load_seq_data(os.path.join(config_dic.get("ner_input_dir"), "test.bioes"), config_dic.get("number_normalize"))
    
    train_char_documents = [[[char for char in word] for word in document] for document in train_word_documents] # Document数 x 文字数
    valid_char_documents = [[[char for char in word] for word in document] for document in valid_word_documents]
    test_char_documents = [[[char for char in word] for word in document] for document in test_word_documents]

    train_sw_documents = get_sw_documents(train_word_documents)
    valid_sw_documents = get_sw_documents(valid_word_documents)
    test_sw_documents = get_sw_documents(test_word_documents)

    # load vocabulary
    print("=========== Build vocabulary ===========")
    if config_dic.get("vocab_dir"):
        word_dic = Dictionary.load(os.path.join(config_dic.get("vocab_dir"), f"{config_dic.get('train_name')}.word.dic"))
        char_dic = Dictionary.load(os.path.join(config_dic.get("vocab_dir"), f"{config_dic.get('train_name')}.char.dic"))
        sw_dic = Dictionary.load(os.path.join(config_dic.get("vocab_dir"), f"{config_dic.get('train_name')}.sw.dic"))
    else:
        special_token_dict = {PADDING: 0, UNKNOWN: 1}
        word_dic = Dictionary()
        word_dic.patch_with_special_tokens(special_token_dict)
        char_dic = Dictionary()
        char_dic.patch_with_special_tokens(special_token_dict)
        sw_dic = Dictionary()
        sw_dic.patch_with_special_tokens(special_token_dict)
    label_dic = Dictionary(train_label_documents)
    label_dic.patch_with_special_tokens({PADDING: 0})
    label_dic.id2token = {_id: label for label, _id in label_dic.token2id.items()}

    # add vocabulary
    word_dic.add_documents(train_word_documents)
    char_dic.add_documents(list(chain.from_iterable(train_char_documents)))
    sw_dic.add_documents(train_sw_documents)
    
    ################### Train Initialize ########################
    if config_dic.get("gpu") and torch.cuda.is_available():
        print("============== Use GPU ================")
        config_dic["gpu"] = True
    else:
        config_dic["gpu"] = False
    seq_model = SeqModel(config_dic, len(word_dic.token2id), len(char_dic.token2id), len(sw_dic.token2id), len(label_dic.token2id), None)
    # load pretrained language model
    if config_dic.get("lm_model_dir"):
        lm_state_dict = torch.load(os.path.join(config_dic.get("lm_model_dir"), f"{config_dic.get('train_name')}.model"))
        seq_model.load_expanded_state_dict(lm_state_dict)
    optimizer = torch.optim.SGD(seq_model.parameters(), lr=config_dic.get("ner_lr"), weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    print(seq_model)

    ## start training
    epoch = config_dic.get("ner_epoch", 1)
    for epoch_i in range(epoch):
        print("Epoch: %s/%s" %(epoch_i, epoch))
        lr_scheduler.step()
        print(f"Learning Rate: {lr_scheduler.get_lr()}")
        # shuffle
        random_ids = list(range(len(train_word_documents)))
        random.shuffle(random_ids)
        seq_model.train()
    
        #####################  Batch Initialize ############################
        batch_ave_loss = 0
        batch_size = config_dic.get("ner_batch_size")
        batch_steps = len(train_word_documents) // batch_size
        for batch_i in tqdm(range(batch_steps)):
            start_time = time.time()
            optimizer.step()
            seq_model.zero_grad()

            batch_ids = random_ids[batch_i * batch_size: (batch_i + 1) * batch_size - 1]
            batch_word_documents = [train_word_documents[i] for i in batch_ids]
            batch_label_documents = [train_label_documents[i] for i in batch_ids]
            word_features = get_word_features(batch_word_documents, word_dic, config_dic.get("gpu"))
            char_features = get_char_features(batch_word_documents, char_dic, config_dic.get("gpu"))
            sw_features = get_sw_features(batch_word_documents, sw_dic, sp, config_dic.get("gpu"))
            label_features = get_label_features(batch_label_documents, label_dic, config_dic.get("gpu"))
            # print(f"word_shape: {word_features.get('word_ids').shape}")
            loss, train_tag_seq = seq_model.neg_log_likelihood_loss(word_features, char_features, sw_features, label_features)
            batch_ave_loss += loss.data
            loss.backward()

            if batch_i % 10 == 0:
                if batch_ave_loss > 1e8 or str(loss) == "nan":
                    print("Error: Loss Explosion (>1e8)! EXIT...")
                    exit(1)
                sys.stdout.flush()
                right_token, total_token = predict_check(train_tag_seq, label_features.get("label_ids"), word_features.get("masks"))
                print(f"""Batch: {batch_i}; Time(sec/batch): {time.time() - start_time:.4f}; Loss: {batch_ave_loss:.4f} Right: {right_token}, Total: {total_token}, Accuracy: {right_token / total_token:.4f}""") 
                batch_ave_loss = 0

        ################ valid predict check #####################
        print("============== Predict Check==========")
        true_seqs, pred_seqs = [], []
        right_token, total_token = 0, 0
        batch_steps = len(valid_word_documents) // batch_size
        random_ids = list(range(len(train_word_documents)))
        for batch_i in tqdm(range(batch_steps)):
            batch_ids = random_ids[batch_i * batch_size: (batch_i + 1) * batch_size - 1]
            batch_word_documents = [train_word_documents[i] for i in batch_ids]
            batch_label_documents = [train_label_documents[i] for i in batch_ids]

            valid_word_features = get_word_features(batch_word_documents, word_dic, config_dic.get("gpu"))
            valid_char_features = get_char_features(batch_word_documents, char_dic, config_dic.get("gpu"))
            valid_sw_features = get_sw_features(batch_word_documents, sw_dic, sp, config_dic.get("gpu"))
            valid_label_features = get_label_features(batch_label_documents, label_dic, config_dic.get("gpu"))
            valid_tag_seq = seq_model.forward(valid_word_features, valid_char_features, valid_sw_features)
            rt, tt = predict_check(valid_tag_seq, valid_label_features.get("label_ids"), valid_word_features.get("masks"))
            right_token += rt
            total_token += tt
            ################ evaluate by precision, recall and fscore ###################
            masks = valid_word_features.get("masks")
            true_seqs.extend([label_dic.id2token[int(label_id)] for label_id in valid_label_features.get("label_ids").masked_select(masks)])
            pred_seqs.extend([label_dic.id2token[int(label_id)] for label_id in valid_tag_seq.masked_select(masks)])
        precision, recall, fscore = evaluate(true_seqs, pred_seqs)
        print(f"Right: {right_token}, Total: {total_token}, Accuracy: {right_token / total_token:.4f}")
        print(f"Precision: {precision}, Recall: {recall}, Fscore: {fscore}")

        gc.collect()
        torch.save(seq_model.state_dict(), os.path.join(config_dic.get("ner_model_dir"), f"{config_dic.get('train_name')}.{epoch_i}.model"))
    
    torch.save(seq_model.state_dict(), os.path.join(config_dic.get("ner_model_dir"), f"{config_dic.get('train_name')}.model"))
