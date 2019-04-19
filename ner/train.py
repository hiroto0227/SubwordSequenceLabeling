import argparse
import time
import random
import gc
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append("./")
from lib.dataset import Dataset
from lib.batchify import batchify_with_label
from evaluate import evaluate
from model.seqlabel import SeqLabel


seed_num = 0
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


def data_initialization(data):
    data.build_alphabet(data.train_dir)
    data.build_alphabet(data.dev_dir)
    data.build_alphabet(data.test_dir)
    data.fix_alphabet()

def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr/(1+decay_rate*epoch)
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
    save_data_name = data.model_dir +".dset"
    data.save(save_data_name)
    model = SeqLabel(data)
    if data.pretrain_model_dir:
        pretrain_state_dict = torch.load(data.pretrain_model_dir + ".model", map_location=lambda storage, loc: storage)
        model.load_state_dict(pretrain_state_dict)
    if data.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=data.lr, momentum=data.momentum,weight_decay=data.l2)
    print(model)
    best_dev = -10
    ## start training
    for idx in range(data.ner_epoch):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" %(idx,data.ner_epoch))
        if data.optimizer == "SGD":
            optimizer = lr_decay(optimizer, idx, data.lr_decay, data.lr)
        instance_count = 0
        sample_id = 0
        sample_loss = 0
        total_loss = 0
        right_token = 0
        whole_token = 0
        random.shuffle(data.train_Ids)
        ## set model in train model
        model.train()
        model.zero_grad()
        best_epoch = 0
        best_dev_scores = (-1, -1, -1)
        best_test_scores = (-1, -1, -1)
        batch_size = data.batch_size
        batch_id = 0
        train_num = len(data.train_Ids)
        total_batch = train_num//batch_size+1
        for batch_id in range(total_batch):
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size
            if end >train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            if not instance:
                continue
            batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_sws, batch_swlens, batch_swrecovers, batch_swfmask, batch_swbmask, batch_label, mask = batchify_with_label(instance, data.gpu, True)
            instance_count += 1
            loss, tag_seq = model.neg_log_likelihood_loss(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_sws, batch_swlens, batch_swrecovers, batch_swfmask, batch_swbmask, batch_label, mask)
            right, whole = predict_check(tag_seq, batch_label, mask)
            right_token += right
            whole_token += whole
            sample_loss += loss.item()
            total_loss += loss.item()
            if end%1000 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("    Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f"%(end, temp_cost, sample_loss, right_token, whole_token,(right_token+0.)/whole_token))
                if sample_loss > 1e8 or str(sample_loss) == "nan":
                    print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
                    exit(1)
                sys.stdout.flush()
                sample_loss = 0
            loss.backward()
            optimizer.step()
            model.zero_grad()
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print("    Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f"%(end, temp_cost, sample_loss, right_token, whole_token,(right_token+0.)/whole_token))

        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s"%(idx, epoch_cost, train_num/epoch_cost, total_loss))
        print("totalloss:", total_loss)
        if total_loss > 1e12 or str(total_loss) == "nan":
            print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
            exit(1)
        # continue
        speed, acc, p, r, f, _,_ = evaluate(data, model, "dev")
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish
        current_score = acc
        print("Valid: time: %.2fs speed: %.2fst/s; acc: %.4f"%(dev_cost, speed, acc))
        print(f"Valid: precition={p}, recall={r}, fscore={f}")
        if current_score > best_dev:
            print("Exceed previous best acc score:", best_dev)
            model_name = data.model_dir +'.'+ str(idx) + ".model"
            print("Save current best model in file:", model_name)
            torch.save(model.state_dict(), model_name)
            best_epoch = idx
            best_dev_scores = (p, r, f)
            best_dev = current_score

        ## decode test
        speed, acc, p, r, f, _,_ = evaluate(data, model, "test")
        test_finish = time.time()
        test_cost = test_finish - dev_finish
        print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f"%(test_cost, speed, acc))
        print(f"Test: precition={p}, recall={r}, fscore={f}")
        if best_test_scores[2] < f:
            best_test_scores = (p, r, f)
        # if  idx % 10 == 0:
        #     model_name = data.model_dir +'.'+ str(idx) + ".model"
        #     torch.save(model.state_dict(), model_name)
        gc.collect()
    return best_epoch, best_dev_scores, best_test_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Subword Sequence Labeling Training')
    parser.add_argument('--config', help='Configuration File')
    args = parser.parse_args()
    dataset = Dataset()
    dataset.gpu = torch.cuda.is_available()
    dataset.read_config(args.config)
    if dataset.pretrain_model_dir:
        # pretrainを行ったdsetを読み込み、AlphabetをPretrainにする。
        lm_dset = Dataset()
        lm_dset.load(f"{dataset.pretrain_model_dir}.dset")
        dataset.load_lm_dset_alphabets(lm_dset)
        print("===============================")
        dataset.describe()
    else:
        data_initialization(dataset)
    dataset.generate_instance('train')
    dataset.generate_instance('dev')
    dataset.generate_instance('test')
    #dataset.build_pretrain_emb()
    dataset.describe()
    train(dataset)
