import argparse
import time
import random
import sys
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append("./")
from lib.dataset import Dataset
from lib.batchify import batchify_with_label
from model.lm import LanguageModel


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1 + decay_rate * epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def train(data):
    model = LanguageModel(data)
    if data.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=data.lr, momentum=data.momentum,weight_decay=data.l2)
    criterion = nn.CrossEntropyLoss().cuda()
    print(model)
    ## start training
    for idx in range(data.lm_epoch):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" %(idx, data.lm_epoch))
        if data.optimizer == "SGD":
            optimizer = lr_decay(optimizer, idx, data.lr_decay, data.lr)
        instance_count, loss, all_loss, right_token, whole_token = 0, 0, 0, 0, 0
        random.shuffle(data.lm_Ids)

        ## set model in train model
        model.train()
        model.zero_grad()
        best_loss = 1e+10
        batch_size = data.batch_size
        batch_id = 0
        all_loss = 0
        train_num = len(data.lm_Ids)
        total_batch = train_num//batch_size+1

        print("Train Start!")
        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = data.lm_Ids[start:end]
            if not instance:
                continue
            batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_sws, batch_swlens, batch_swrecovers, batch_swfmask, batch_swbmask, _, mask = batchify_with_label(instance, data.gpu, if_train=True, label=False)
            instance_count += 1

            forward_out, backward_out = model.calc_next_word_id_prob(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_sws, batch_swlens, batch_swrecovers, batch_swfmask, batch_swbmask)
            
            #### loss_function ###
            f_next_word_id = torch.cat([batch_word[:, 1:], torch.cuda.LongTensor(batch_word.shape[0], 1).fill_(0)], dim=1)
            loss = criterion(forward_out.reshape(-1, forward_out.shape[2]), f_next_word_id.contiguous().reshape(-1))

            b_next_word_id = torch.cat([torch.cuda.LongTensor(batch_word.shape[0], 1).fill_(0), batch_word[:, :-1]], dim=1)
            loss += criterion(backward_out.reshape(-1, backward_out.shape[2]), b_next_word_id.contiguous().reshape(-1))

            all_loss += loss.item()
            if end % 1000 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("Instance: %s; Time: %.2fs; loss: %.4f;"%(end, temp_cost, loss))
                if loss > 1e8 or str(loss) == "nan":
                    print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
                    exit(1)
                sys.stdout.flush()
            loss.backward()
            optimizer.step()
            model.zero_grad()
            
        temp_time = time.time()
        temp_cost = temp_time - temp_start

        if all_loss < best_loss:
            model_name = f"{data.pretrain_model_dir}.{data.name}.{idx}.model"
            print("Save current best model in file:", model_name)
            torch.save(model.state_dict(), model_name)
            best_loss = best_loss
        
        gc.collect()
    torch.save(model.state_dict(), f"{data.pretrain_model_dir}.{data.name}.model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("train Language Model for NER")
    parser.add_argument("--config", type=str, help="configuration file")
    parser.add_argument("--cache", type=str, help="if use cachea")
    opt = parser.parse_args()
    dataset = Dataset()
    if not opt.cache:
        dataset.read_config(opt.config)
        dataset.build_alphabet(dataset.lm_dir, mode="lm")
        dataset.word_alphabet.cut_vocab(min_count=dataset.lm_vocab_min_count)
        dataset.fix_alphabet()
        dataset.describe()
        dataset.generate_instance('lm')
        dataset.build_pretrain_emb()
        dset_path = os.path.join(os.path.dirname(dataset.pretrain_model_dir), f"{dataset.name}.dset")
        print(f"dataset save to :{dset_path}")
        dataset.save(dset_path)
    else:
        print(f"load dataset: {opt.cache}")
        dataset.load(opt.cache)
    train(dataset)
