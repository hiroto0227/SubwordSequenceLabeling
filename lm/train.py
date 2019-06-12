import argparse
import gc
from itertools import chain
import os
import pprint
import random
import re
import sys
import time
from typing import List

import cloudpickle
from gensim.corpora import Dictionary
import numpy as np
import torch
from tqdm import tqdm

sys.path.append("./")
from chem_sentencepiece.chem_sentencepiece import ChemSentencePiece
from config import config_dics
from label import UNKNOWN, PADDING, NUM, START, END
from utils import tokenize, normalize_number_word, load_pretrain_embeddings,build_pretrain_embeddings
from model.lm import LanguageModel
from preprocess import get_char_features, get_sw_features, get_word_features

seed_num = 42
random.seed(seed_num)
np.random.seed(seed_num)
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)
torch.backends.cudnn.deterministic = True


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1 + decay_rate * epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def load_lm_documents(lm_input_path: str, number_normalize: bool) -> List[List[str]]:
    with open(lm_input_path, "rt", encoding="utf-8") as f:
        docs = f.read().splitlines()

    word_documents = []
    for doc in tqdm(docs):
        for line in re.split("\. +", doc):
            if line:
                word_document = []
                if line.endswith("."):
                    line = line + " ."
                else:
                    line = line[:-1] + " ."
                for word in tokenize(line):
                    if number_normalize:
                        word = normalize_number_word(word)
                    word_document.append(word)
                if len(word_document) < 1000:  # MAX_SENT_LENGTH
                    word_documents.append(word_document)
    return word_documents

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    config_dic = config_dics[args.config]
    pprint.pprint(config_dic)

    # load sentence piece
    if config_dic.get("sp_path"):
        sp = ChemSentencePiece()
        sp.load(config_dic.get("sp_path"))
    else:
        sp = None

    # load train data
    print("=========== Load train data ===========")
    _, lm_corpus_name = os.path.split(config_dic.get("lm_input_path"))
    if os.path.exists(os.path.join(config_dic.get("cache_dir"), lm_corpus_name + ".word_documents")):
        print(f"Use Cache data. {os.path.join(config_dic.get('cache_dir'), lm_corpus_name + '.word_documents')}")
        with open(os.path.join(config_dic.get("cache_dir"), lm_corpus_name + ".word_documents"), "rb") as f:
            word_documents = cloudpickle.loads(f.read())
    else:
        word_documents = load_lm_documents(config_dic.get("lm_input_path"), config_dic.get("number_normalize"))
    if os.path.exists(config_dic.get("cache_dir")) and not os.path.exists(os.path.join(config_dic.get("cache_dir"), lm_corpus_name + ".word_documents")):
        print(f"Write Cache data. {os.path.join(config_dic.get('cache_dir'), lm_corpus_name + '.word_documents')}")
        with open(os.path.join(config_dic.get("cache_dir"), lm_corpus_name + ".word_documents"), "wb") as f:
            f.write(cloudpickle.dumps(word_documents))

    if sp:
        print("=========== Split Subword ==============")
        if os.path.exists(os.path.join(config_dic.get('cache_dir'), lm_corpus_name + '.sw_documents')):
            print(f"Use Cache data. {os.path.join(config_dic.get('cache_dir'), lm_corpus_name + '.sw_documents')}")
            with open(os.path.join(config_dic.get("cache_dir"), lm_corpus_name + ".sw_documents"), "rb") as f:
                sw_documents = cloudpickle.loads(f.read())
        else:
            sw_documents = []
            for word_document in tqdm(word_documents):
                sws = [sp.tokenize(word) for word in word_document]
                sw_documents.append(list(chain.from_iterable(sws)))
        if os.path.exists(config_dic.get("cache_dir")) and not os.path.exists(os.path.join(config_dic.get("cache_dir"), lm_corpus_name + ".sw_documents")):
            print(f"Write Cache data. {os.path.join(config_dic.get('cache_dir'), lm_corpus_name + '.sw_documents')}")
            with open(os.path.join(config_dic.get("cache_dir"), lm_corpus_name + ".sw_documents"), "wb") as f:
                f.write(cloudpickle.dumps(sw_documents))
    
    print("=========== Build vocabulary ===========")
    special_token_dict = {PADDING: 0, UNKNOWN: 1, START: 2, END: 3}
    word_dic = Dictionary(word_documents)
    word_dic.filter_extremes(no_below=5, no_above=1.0, keep_n=None)
    word_dic.patch_with_special_tokens(special_token_dict)
    if sp:
        sw_dic = Dictionary(sw_documents)
        sw_dic.filter_extremes(no_below=5, no_above=1.0, keep_n=None)
        sw_dic.patch_with_special_tokens(special_token_dict)
    else:
        sw_dic = None

    char_dic = Dictionary([[char for word in word_document for char in word] for word_document in word_documents])
    char_dic.filter_extremes(no_below=5, no_above=1.0, keep_n=None)
    char_dic.patch_with_special_tokens(special_token_dict)

    word_dic.save(os.path.join(config_dic.get("vocab_dir"), f"{args.config}.word.dic"))
    char_dic.save(os.path.join(config_dic.get("vocab_dir"), f"{args.config}.char.dic"))
    if sw_dic:
        sw_dic.save(os.path.join(config_dic.get("vocab_dir"), f"{args.config}.sw.dic"))

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

    if sw_dic:
        model = LanguageModel(config_dic, len(word_dic.token2id), len(char_dic.token2id), len(sw_dic.token2id), pretrain_embeddings)
    else:
        model = LanguageModel(config_dic, len(word_dic.token2id), len(char_dic.token2id), None, pretrain_embeddings)
    optimizer = torch.optim.SGD(model.parameters(), lr=config_dic.get("lm_lr"), weight_decay=config_dic.get("weight_decay"))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    criterion = torch.nn.NLLLoss(size_average=True, ignore_index=word_dic.token2id.get(PADDING))

    print(model)
    print(lr_scheduler.state_dict())

    save_model_path = os.path.join(config_dic.get("lm_model_dir"), f"{args.config}.model")
    print(f"Save Language Model! {save_model_path}")
    torch.save(model.state_dict(), save_model_path)

    ## start training
    epoch = config_dic.get("lm_epoch")
    for epoch_i in range(epoch):
        print("Epoch: %s/%s" %(epoch_i, epoch))
        print(f"Learning Rate: {lr_scheduler.get_lr()}")
        random.shuffle(word_documents)
        model.train()
    
        #####################  Batch Initialize ############################
        total_loss = 0
        batch_size = config_dic.get("lm_batch_size")
        batch_steps = len(word_documents) // batch_size + 1
        for batch_i in tqdm(range(batch_steps)):
            start_time = time.time()
            model.zero_grad()
            batch_word_documents = word_documents[batch_i * batch_size: (batch_i + 1) * batch_size]
            word_features = get_word_features(batch_word_documents, word_dic, config_dic.get("gpu"))
            char_features = get_char_features(batch_word_documents, char_dic, config_dic.get("gpu"))
            if sp:
                sw_features = get_sw_features(batch_word_documents, sw_dic, sp, config_dic.get("gpu"))
            else:
                sw_features = None
            #print(f"word_shape: {word_features.get('word_ids').shape}")
            forward_out, backward_out = model.calc_next_word_id_prob(word_features, char_features, sw_features)
            #### loss_function ###
            batch_word = word_features.get("word_ids")
            if config_dic.get("gpu"):
                f_next_word_id = torch.cat([batch_word[:, 1:], torch.cuda.LongTensor(batch_word.shape[0], 1).fill_(word_dic.token2id.get(PADDING))], dim=1)
                b_next_word_id = torch.cat([torch.cuda.LongTensor(batch_word.shape[0], 1).fill_(word_dic.token2id.get(START)), batch_word[:, :-1]], dim=1)
            else:
                f_next_word_id = torch.cat([batch_word[:, 1:], torch.LongTensor(batch_word.shape[0], 1).fill_(word_dic.token2id.get(PADDING))], dim=1)
                b_next_word_id = torch.cat([torch.LongTensor(batch_word.shape[0], 1).fill_(word_dic.token2id.get(START)), batch_word[:, :-1]], dim=1)
            
            loss = criterion(forward_out.reshape(-1, forward_out.shape[2]), f_next_word_id.contiguous().reshape(-1))
            loss += criterion(backward_out.reshape(-1, backward_out.shape[2]), b_next_word_id.contiguous().reshape(-1))

            if batch_i % 50 == 0:
                print(f"Batch: {batch_i}; Time(sec/batch): {time.time() - start_time}; Loss: {loss};")
                if loss > 1e8 or str(loss) == "nan":
                    print("Error: Loss Explosion (>1e8)! EXIT...")
                    exit(1)
                sys.stdout.flush()
            
            loss.backward()
            if config_dic.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config_dic.get("grad_clip")) # モデルの重みをclipする。
            optimizer.step()
            model.zero_grad()

            total_loss += loss.data

            # if batch_i == 100000:
            #     print("======break========")
            #     print(total_loss)
            #     break
            gc.collect()
        
        save_model_path = os.path.join(config_dic.get("lm_model_dir"), f"{args.config}.{epoch_i}.model")
        print(f"Save Language Model! {save_model_path}")
        torch.save(model.state_dict(), save_model_path)
    
    save_model_path = os.path.join(config_dic.get("lm_model_dir"), f"{args.config}.model")
    print(f"Save Language Model! {save_model_path}")
    torch.save(model.state_dict(), save_model_path)
