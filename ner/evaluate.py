import time
import sys
import argparse

from lib.dataset import batchify_with_label


def evaluate(data, model, name):
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print("Error: wrong evaluate name,", name)
        exit(1)
    pred_results = []
    gold_results = []
    sentence_list = []
    ## set model in eval model
    model.eval()
    batch_size = data.HP_batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//batch_size+1
    print(len(instances))
    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end > train_num:
            end =  train_num
        instance = instances[start:end]
        if not instance:
            continue
        batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_sws, batch_swlens, batch_swrecovers, batch_swfmask, batch_swbmask, batch_label, mask  = batchify_with_label(instance, data.HP_gpu, False)
        tag_seq = model(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_sws, batch_swlens, batch_swrecovers, batch_swfmask, batch_swbmask, mask)
        # print("tag:",tag_seq)
        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)
        for i in range(len(pred_label)):
            _s = []
            for w in batch_word[batch_wordrecover][i]:
                if w:
                    _s.append(data.word_alphabet.get_instance(int(w)))
            sentence_list.append(_s)
        pred_results += pred_label
        gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances)/decode_time
    acc, p, r, f, gold_words, pred_words = get_ner_fmeasure(sentence_list, gold_results, pred_results, data.tagScheme)
    return speed, acc, p, r, f, gold_words, pred_words


def get_words(matrix, word_list, tag="CHEM"):
    words = []
    for row in matrix:
        if "," in row:
            start, end = row.replace(tag, "")[1:-1].split(",")
        else:
            start = row.replace(tag, "")[1: -1]
            end = int(start)
        #print(" ".join(word_list[int(start):int(end) + 1]))
        words.append(" ".join(word_list[int(start):int(end) + 1]))
    return words

## input as sentence level labelst_ner
def get_ner_fmeasure(sentence_lists, golden_lists, predict_lists, label_type="BMES"):
    gold_words = []
    pred_words = []
    sent_num = len(golden_lists)
    right_tag = 0
    all_tag = 0
    for idx in range(0,sent_num):
        word_list = [str(idx) + "_" + str(i) + "_" + word for i, word in enumerate(sentence_lists[idx])]
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]
        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1
        all_tag += len(golden_list)
        if label_type == "BMES":
            gold_matrix = get_ner_BMES(golden_list)
            pred_matrix = get_ner_BMES(predict_list)
        gold_words.extend(get_words(gold_matrix, word_list))
        pred_words.extend(get_words(pred_matrix, word_list))
    right_num, predict_num, golden_num = 0, 0, 0
    for uniq_word in set(gold_words + pred_words):
        gold_flag, pred_flag = False, False
        if uniq_word in pred_words:
            predict_num += 1
            pred_flag = True
        if uniq_word in gold_words:
            golden_num += 1
            gold_flag = True
        if pred_flag and gold_flag:
            right_num += 1
    if predict_num == 0:
        precision = -1
    else:
        precision =  (right_num+0.0)/predict_num
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num+0.0)/golden_num
    if (precision == -1) or (recall == -1) or (precision+recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2*precision*recall/(precision+recall)
    accuracy = (right_tag+0.0)/all_tag
    # print "Accuracy: ", right_tag,"/",all_tag,"=",accuracy
    print("gold_num = ", golden_num, " pred_num = ", predict_num, " right_num = ", right_num)
    return accuracy, precision, recall, f_measure, gold_words, pred_words

def get_ner_BMES(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    end_label = 'E-'
    single_label = 'S-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(begin_label, "", 1) +'[' +str(i)
            index_tag = current_label.replace(begin_label, "", 1)

        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(single_label, "", 1) +'[' +str(i)
            tag_list.append(whole_tag)
            whole_tag = ""
            index_tag = ""
        elif end_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag +',' + str(i))
            whole_tag = ''
            index_tag = ''
        else:
            continue
    if (whole_tag != '')&(index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if  len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix


def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        assert(len(pred)==len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label
