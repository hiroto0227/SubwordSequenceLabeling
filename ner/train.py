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

from gensim.corpora import Dictionary
import numpy as np
import torch
from tqdm import tqdm

sys.path.append("./")
from chem_sentencepiece.chem_sentencepiece import ChemSentencePiece
from chem_sentencepiece.char_tokenizer import CharTokenizer
from config import config_dics
from label import UNKNOWN, PADDING, NUM, START, END
from utils import load_pretrain_embeddings, build_pretrain_embeddings, normalize_number_word
from model.seqmodel import SeqModel
from ner.evaluate import evaluate
from preprocess import get_char_features, get_sw_features, get_word_features, get_label_features
from trainer.opt import NoamOpt

seed_num = 42
random.seed(seed_num)
np.random.seed(seed_num)
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)
torch.backends.cudnn.deterministic = True

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
    for line in tqdm(lines):
        if len(line) > 0:
            word, label = line.split(" ")
            if number_normalize:
                word = normalize_number_word(word)
            word_document.append(word)
            label_document.append(label)
        elif len(word_document):
            word_documents.append(word_document)
            label_documents.append(label_document)
            word_document, label_document = [], []
    return word_documents, label_documents


def get_sw_documents(word_documents: List[List[str]], sp: ChemSentencePiece) -> List[List[str]]:
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
    return right_token, total_token


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
    config_dic = config_dics[args.config]
    pprint.pprint(config_dic)

    # load sentence piece
    sps: dict = {"Char": CharTokenizer()}
    for sp_key, sp_path in config_dic["sp_path"].items():
        sps[sp_key] = ChemSentencePiece.load(sp_path)
    
    # load train data
    print("=========== Load train data ===========")
    train_word_documents, train_label_documents = load_seq_data(os.path.join(config_dic.get("ner_input_dir"), "train.bioes"), config_dic.get("number_normalize"))
    valid_word_documents, valid_label_documents = load_seq_data(os.path.join(config_dic.get("ner_input_dir"), "valid.bioes"), config_dic.get("number_normalize"))
    test_word_documents, test_label_documents = load_seq_data(os.path.join(config_dic.get("ner_input_dir"), "test.bioes"), config_dic.get("number_normalize"))

    # train_char_documents = [[[char for char in word] for word in document] for document in train_word_documents] # Document数 x 文字数
    # valid_char_documents = [[[char for char in word] for word in document] for document in valid_word_documents]
    # test_char_documents = [[[char for char in word] for word in document] for document in test_word_documents]

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
        #char_dic = Dictionary.load(os.path.join(config_dic.get("vocab_dir"), f"{args.config}.char.dic"))
        sw_dicts = {}
        for sp_key, sp in sps.items():
            sw_dicts[sp_key] = Dictionary.load(os.path.join(config_dic.get("vocab_dir"), f"{args.config}.{sp_key}.dic"))
    else:
        special_token_dict = {PADDING: 0, UNKNOWN: 1, START: 2, END: 3}
        word_dic = Dictionary()
        word_dic.token2id = special_token_dict
        #char_dic = Dictionary()
        #char_dic.token2id = special_token_dict
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
    #char_dic.add_documents(list(chain.from_iterable(train_char_documents)))
    for sp_key, train_sw_documents in train_sw_documents_dicts.items():
        sw_dicts[sp_key].add_documents(train_sw_documents)

    # load GloVe
    if config_dic.get("glove_path"):
        print("========= Load Pretrain Word Embeddings ==========")
        word2vec = load_pretrain_embeddings(config_dic.get("glove_path"), emb_dim=config_dic.get("word_emb_dim"))
        pretrain_embeddings = build_pretrain_embeddings(word2vec, word_dic, emb_dim=config_dic.get("word_emb_dim"))
    else:
        pretrain_embeddings = None
    
    ################### Train Initialize ########################
    if config_dic.get("gpu") and torch.cuda.is_available():
        print("============== Use GPU ================")
        config_dic["gpu"] = True
    else:
        config_dic["gpu"] = False
  

    seq_model = SeqModel(
        config_dic, 
        len(word_dic.token2id), 
        None, 
        [len(sw_dic.token2id) for sw_dic in sw_dicts.values()], 
        len(label_dic.token2id), 
        pretrain_embeddings
    )

    # load pretrained language model
    if config_dic.get("lm_model_dir"):
       lm_state_dict = torch.load(os.path.join(config_dic.get("lm_model_dir"), f"{args.config}.model"))
       seq_model.load_expanded_state_dict(lm_state_dict)

    optimizer = torch.optim.SGD(seq_model.parameters(), lr=config_dic.get("ner_lr"), weight_decay=config_dic.get("weight_decay"))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.99)
    # def get_std_opt(model):
    #     return NoamOpt(50, 25, 100000,
    #             torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    # lr_scheduler = get_std_opt(seq_model)
    
    print(seq_model)
    print(optimizer)

    ## start training
    epoch = config_dic.get("ner_epoch")
    for epoch_i in range(epoch):
        print("Epoch: %s/%s" %(epoch_i, epoch))
        lr_scheduler.step()
        #print(f"Learning Rate: {lr_scheduler.rate()}")
        print(f"Learning Rate: {lr_scheduler.get_lr()}")

        random_ids = list(range(len(train_word_documents)))
        random.shuffle(random_ids)
        
        #####################  Batch Initialize ############################
        total_loss, batch_ave_loss, right_token, total_token = 0, 0, 0, 0
        batch_size = config_dic.get("ner_batch_size")
        batch_steps = len(train_word_documents) // batch_size + 1
        seq_model.train()
        seq_model.zero_grad()
        optimizer.zero_grad()

        for batch_i in tqdm(range(batch_steps)):
            #lr_scheduler.step()
            start_time = time.time()
            batch_ids = random_ids[batch_i * batch_size: (batch_i + 1) * batch_size]
            batch_word_documents = [train_word_documents[i] for i in batch_ids]
            batch_label_documents = [train_label_documents[i] for i in batch_ids]
            word_features = get_word_features(batch_word_documents, word_dic, config_dic.get("gpu"))
            #char_features = get_char_features(batch_word_documents, char_dic, config_dic.get("gpu"))
            sw_features_list = []
            for sp_key, sp in sps.items():
                sw_features_list.append(get_sw_features(batch_word_documents, sw_dicts[sp_key], sp, config_dic.get("gpu")))
            label_features = get_label_features(batch_label_documents, label_dic, config_dic.get("gpu"))
            loss, train_tag_seq = seq_model.neg_log_likelihood_loss(word_features, None, sw_features_list, label_features)
            batch_ave_loss += loss.data
            total_loss += loss.data
            loss.backward()

            optimizer.step()
            seq_model.zero_grad()

            rt, tt = predict_check(train_tag_seq, label_features.get("label_ids"), word_features.get("masks"))
            right_token += rt
            total_token += tt
            if batch_i != 0 and batch_i % 50 == 0:
                if batch_ave_loss > 1e8 or str(loss) == "nan":
                    print("Error: Loss Explosion (>1e8)! EXIT...")
                    exit(1)
                sys.stdout.flush()
                print(f"""Batch: {batch_i}; Time(sec/batch): {time.time() - start_time:.4f}; Loss: {batch_ave_loss:.4f} Right: {right_token}, Total: {total_token}, Accuracy: {right_token / total_token:.4f}""") 
                batch_ave_loss = 0
        print(f"Total Loss: {total_loss}")

        ################ valid predict check #####################
        print("============== Valid Evaluate ===========")
        true_seqs, pred_seqs = [], []
        right_token, total_token = 0, 0
        batch_steps = len(valid_word_documents) // batch_size + 1
        random_ids = list(range(len(valid_word_documents)))
        seq_model.eval()
        for batch_i in tqdm(range(batch_steps)):
            batch_ids = random_ids[batch_i * batch_size: (batch_i + 1) * batch_size]
            batch_word_documents = [valid_word_documents[i] for i in batch_ids]
            batch_label_documents = [valid_label_documents[i] for i in batch_ids]

            valid_word_features = get_word_features(batch_word_documents, word_dic, config_dic.get("gpu"))
            #valid_char_features = get_char_features(batch_word_documents, char_dic, config_dic.get("gpu"))
            valid_sw_features_list = []
            for sp_key, sp in sps.items():
                valid_sw_features_list.append(get_sw_features(batch_word_documents, sw_dicts[sp_key], sp, config_dic.get("gpu")))
            valid_label_features = get_label_features(batch_label_documents, label_dic, config_dic.get("gpu"))
            valid_tag_seq = seq_model.forward(valid_word_features, None, valid_sw_features_list)
            masks = valid_word_features.get("masks")
            rt, tt = predict_check(valid_tag_seq, valid_label_features.get("label_ids"), masks)
            right_token += rt
            total_token += tt
            ################ evaluate by precision, recall and fscore ###################
            true_seqs.extend([label_dic.id2token.get(int(label_id), label_dic.token2id["O"]) for label_id in valid_label_features.get("label_ids").masked_select(masks)])
            pred_seqs.extend([label_dic.id2token.get(int(label_id), label_dic.token2id["O"]) for label_id in valid_tag_seq.masked_select(masks)])
        precision, recall, fscore = evaluate(true_seqs, pred_seqs)
        print(f"Right: {right_token}, Total: {total_token}, Accuracy: {right_token / total_token:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, Fscore: {fscore:.4f}")

        ################ test predict check #####################
        print("============== Test Evaluate ===========")
        true_seqs, pred_seqs = [], []
        right_token, total_token = 0, 0
        seq_model.eval()
        batch_steps = len(test_word_documents) // batch_size + 1
        random_ids = list(range(len(test_word_documents)))
        for batch_i in tqdm(range(batch_steps)):
            batch_ids = random_ids[batch_i * batch_size: (batch_i + 1) * batch_size]
            batch_word_documents = [test_word_documents[i] for i in batch_ids]
            batch_label_documents = [test_label_documents[i] for i in batch_ids]

            test_word_features = get_word_features(batch_word_documents, word_dic, config_dic.get("gpu"))
            #test_char_features = get_char_features(batch_word_documents, char_dic, config_dic.get("gpu"))
            test_sw_features_list = []
            for sp_key, sp in sps.items():
                test_sw_features_list.append(get_sw_features(batch_word_documents, sw_dicts[sp_key], sp, config_dic.get("gpu")))
            test_label_features = get_label_features(batch_label_documents, label_dic, config_dic.get("gpu"))
            test_tag_seq = seq_model.forward(test_word_features, None, test_sw_features_list)
            rt, tt = predict_check(test_tag_seq, test_label_features.get("label_ids"), test_word_features.get("masks"))
            right_token += rt
            total_token += tt
            ################ evaluate by precision, recall and fscore ###################
            masks = test_word_features.get("masks")
            true_seqs.extend([label_dic.id2token.get(int(label_id), label_dic.token2id["O"]) for label_id in test_label_features.get("label_ids").masked_select(masks)])
            pred_seqs.extend([label_dic.id2token.get(int(label_id), label_dic.token2id["O"]) for label_id in test_tag_seq.masked_select(masks)])
        precision, recall, fscore = evaluate(true_seqs, pred_seqs)
        print(f"Right: {right_token}, Total: {total_token}, Accuracy: {right_token / total_token:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, Fscore: {fscore:.4f}")

        gc.collect()
        torch.save(seq_model.state_dict(), os.path.join(config_dic.get("ner_model_dir"), f"{args.config}.{epoch_i}.model"))
    
    torch.save(seq_model.state_dict(), os.path.join(config_dic.get("ner_model_dir"), f"{args.config}.model"))
