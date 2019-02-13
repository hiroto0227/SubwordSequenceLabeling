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
from lib.tensor_util import get_embedding_tensor_from_id_tensor
from model.lm import LanguageModel


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1 + decay_rate * epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


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


def train(data):
    model = LanguageModel(data)
    if data.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=data.HP_lr, momentum=data.HP_momentum,weight_decay=data.HP_l2)
    best_dev = -10
    criterion = nn.MSELoss(size_average=False)
    ## start training
    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" %(idx, data.HP_iteration))
        if data.optimizer == "SGD":
            optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
        instance_count, loss, all_loss, right_token, whole_token = 0, 0, 0, 0, 0
        random.shuffle(data.train_Ids)

        ## set model in train model
        model.train()
        model.zero_grad()
        best_loss = 1e+10
        batch_size = data.HP_batch_size
        batch_id = 0
        all_loss = 0
        train_num = len(data.lm_Ids)
        total_batch = train_num//batch_size+1

        for batch_id in range(total_batch):
            loss = 0
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = data.lm_Ids[start:end]
            if not instance:
                continue
            batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_sws, batch_swlens, batch_swrecovers, batch_swfmask, batch_swbmask, _, mask = batchify_with_label(instance, data.HP_gpu, if_train=True, label=False)
            instance_count += 1
            output = model(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_sws, batch_swlens, batch_swrecovers, batch_swfmask, batch_swbmask)
            #### loss_function ###
            forward_out, backward_out = output.split(output.shape[2] // 2, dim=2)
            embed_tensor = get_embedding_tensor_from_id_tensor(batch_word, dataset.pretrain_word_embedding)
            loss += criterion(forward_out, torch.cat([embed_tensor[:, 1:], torch.zeros(batch_size, 1, embed_tensor.shape[2])], dim=1))
            loss += criterion(backward_out, torch.cat([torch.zeros(batch_size, 1, embed_tensor.shape[2]), embed_tensor[:, :-1]], dim=1))
            all_loss += loss.item()
            if end % 10 == 0:
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
            model_name = data.model_dir +'.'+ str(idx) + ".model"
            print("Save current best model in file:", model_name)
            torch.save(model.state_dict(), model_name)
            best_loss = best_loss
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("train Language Model for NER")
    parser.add_argument("--config", type=str, help="configuration file")
    opt = parser.parse_args()
    dataset = Dataset()
    dataset.read_config(opt.config)
    dataset.build_alphabet(dataset.lm_dir)
    dataset.fix_alphabet()
    dataset.generate_instance('lm')
    dataset.build_pretrain_emb()
    dataset.save(dataset.model_dir + ".dset")
    train(dataset)
