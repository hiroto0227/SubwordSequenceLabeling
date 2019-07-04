import argparse
import gc
from itertools import chain
import os
import random
import re
import sys
import time
from typing import List

import cloudpickle
from gensim.corpora import Dictionary
import torch
from tqdm import tqdm

sys.path.append("./")
from chem_sentencepiece.chem_sentencepiece import ChemSentencePiece
from config import config_dic
from label import UNKNOWN, PADDING, NUM
from utils import tokenize
from model.lm import LanguageModel
from preprocess import get_char_features, get_sw_features, get_word_features

seed_num = 0
random.seed(0)

def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1 + decay_rate * epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    # load sentence piece
    sp = ChemSentencePiece()
    sp.load(config_dic.get("sp_path"))

    # load train data
    print("=========== Load train data ===========")
    if os.path.exists(os.path.join(config_dic.get("cache_dir"), config_dic.get('train_name') + ".word_documents")):
        print(f"Use Cache data. {os.path.join(config_dic.get('cache_dir'), config_dic.get('train_name') + '.word_documents')}")
        with open(os.path.join(config_dic.get("cache_dir"), config_dic.get('train_name') + ".word_documents"), "rb") as f:
            word_documents = cloudpickle.loads(f.read())
    else:
        with open(config_dic.get("lm_input_path"), "rt", encoding="utf-8") as f:
            lines = f.read().split("\n")

        word_documents = []
        num_re = re.compile("\d+\.*\d*")
        for line in tqdm(lines):
            word_document = []
            if line:
                for word in tokenize(line):
                    if config_dic.get("number_normalize") and num_re.match(word):
                        word = NUM
                    word_document.append(word)
                if len(word_document) < 1000:  # MAX_SENT_LENGTH
                    word_documents.append(word_document)
    if os.path.exists(config_dic.get("cache_dir")):
        print(f"Write Cache data. {os.path.join(config_dic.get('cache_dir'), config_dic.get('train_name') + '.word_documents')}")
        with open(os.path.join(config_dic.get("cache_dir"), config_dic.get('train_name') + ".word_documents"), "wb") as f:
            f.write(cloudpickle.dumps(word_documents))

    print("=========== Split Subword ==============")
    if os.path.exists(os.path.join(config_dic.get('cache_dir'), config_dic.get('train_name') + '.sw_documents')):
        print(f"Use Cache data. {os.path.join(config_dic.get('cache_dir'), config_dic.get('train_name') + '.sw_documents')}")
        with open(os.path.join(config_dic.get("cache_dir"), config_dic.get('train_name') + ".sw_documents"), "rb") as f:
            sw_documents = cloudpickle.loads(f.read())
    else:
        sw_documents = []
        for word_document in tqdm(word_documents):
            sws = [sp.tokenize(word) for word in word_document]
            sw_documents.append(list(chain.from_iterable(sws)))
    if os.path.exists(config_dic.get("cache_dir")):
        print(f"Write Cache data. {os.path.join(config_dic.get('cache_dir'), config_dic.get('train_name') + '.sw_documents')}")
        with open(os.path.join(config_dic.get("cache_dir"), config_dic.get('train_name') + ".sw_documents"), "wb") as f:
            f.write(cloudpickle.dumps(sw_documents))
    
    print("=========== Build vocabulary ===========")
    special_token_dict = {PADDING: 0, UNKNOWN: 1}
    word_dic = Dictionary(word_documents)
    word_dic.filter_extremes(no_below=10, no_above=1.0)
    word_dic.patch_with_special_tokens(special_token_dict)
    sw_dic = Dictionary(sw_documents)
    sw_dic.filter_extremes(no_below=5, no_above=1.0)
    sw_dic.patch_with_special_tokens(special_token_dict)
    char_documents = [[[char for char in word] for word in document]
                        for document in word_documents] # Document数 x 文字数
    char_dic = Dictionary(list(chain.from_iterable(char_documents)))
    char_dic.patch_with_special_tokens(special_token_dict)

    word_dic.save(os.path.join(config_dic.get("vocab_dir"), f"{config_dic.get('train_name')}.word.dic"))
    char_dic.save(os.path.join(config_dic.get("vocab_dir"), f"{config_dic.get('train_name')}.char.dic"))
    sw_dic.save(os.path.join(config_dic.get("vocab_dir"), f"{config_dic.get('train_name')}.sw.dic"))


    ################### Train Initialize ########################
    if config_dic.get("gpu") and torch.cuda.is_available():
        print("============== Use GPU ================")
        config_dic["gpu"] = True
    else:
        config_dic["gpu"] = False
    model = LanguageModel(config_dic, len(word_dic.token2id), len(char_dic.token2id), len(sw_dic.token2id), None)
    optimizer = torch.optim.SGD(model.parameters(), lr=config_dic.get("lm_lr"), weight_decay=config_dic.get("weight_decay"))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    criterion = torch.nn.CrossEntropyLoss(size_average=True)
    print(model)
    ## start training
    epoch = config_dic.get("lm_epoch", 1)
    for epoch_i in range(epoch):
        print("Epoch: %s/%s" %(epoch_i, epoch))
        print(f"Learning Rate: {lr_scheduler.get_lr()}")
        random.shuffle(word_documents)
        model.train()
    
        #####################  Batch Initialize ############################
        batch_size = config_dic.get("lm_batch_size")
        batch_steps = len(word_documents) // batch_size
        for batch_i in tqdm(range(batch_steps)):
            start_time = time.time()
            model.zero_grad()
            batch_word_documents = word_documents[batch_i * batch_size: (batch_i + 1) * batch_size - 1]
            word_features = get_word_features(batch_word_documents, word_dic, config_dic.get("gpu"))
            char_features = get_char_features(batch_word_documents, char_dic, config_dic.get("gpu"))
            sw_features = get_sw_features(batch_word_documents, sw_dic, sp, config_dic.get("gpu"))
            #print(f"word_shape: {word_features.get('word_ids').shape}")
            forward_out, backward_out = model.calc_next_word_id_prob(word_features, char_features, sw_features)
            #### loss_function ###
            batch_word = word_features.get("word_ids")
            if config_dic.get("gpu"):
                f_next_word_id = torch.cat([batch_word[:, 1:], torch.cuda.LongTensor(batch_word.shape[0], 1).fill_(0)], dim=1)
                b_next_word_id = torch.cat([torch.cuda.LongTensor(batch_word.shape[0], 1).fill_(0), batch_word[:, :-1]], dim=1)
            else:
                f_next_word_id = torch.cat([batch_word[:, 1:], torch.LongTensor(batch_word.shape[0], 1).fill_(0)], dim=1)
                b_next_word_id = torch.cat([torch.LongTensor(batch_word.shape[0], 1).fill_(0), batch_word[:, :-1]], dim=1)
            loss = criterion(forward_out.reshape(-1, forward_out.shape[2]), f_next_word_id.contiguous().reshape(-1))
            loss += criterion(backward_out.reshape(-1, backward_out.shape[2]), b_next_word_id.contiguous().reshape(-1))

            if batch_i % 10 == 0:
                print(f"Batch: {batch_i}; Time(sec/batch): {time.time() - start_time}; Loss: {loss};")
                if loss > 1e8 or str(loss) == "nan":
                    print("Error: Loss Explosion (>1e8)! EXIT...")
                    exit(1)
                sys.stdout.flush()
            
            loss.backward()
            optimizer.step()

            if batch_i % 10000 == 0:
                print("====== break for short cut ========")
                break
            # torch.save(model.state_dict(), os.path.join(config_dic.get("lm_model_dir"), f"{config_dic.get('train_name', 'test')}.model"))
            # exit(100)
        
        gc.collect()
        torch.save(model.state_dict(), os.path.join(config_dic.get("lm_model_dir"), f"{config_dic.get('train_name', 'test')}.{epoch_i}.model"))
    
    torch.save(model.state_dict(), os.path.join(config_dic.get("lm_model_dir"), f"{config_dic.get('train_name', 'test')}.model"))
