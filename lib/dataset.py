import sys
import re
import pickle as pickle

from tqdm import tqdm

from lib.alphabet import Alphabet
from lib.function import build_pretrain_embedding, normalize_word, cammel_normalize_word, tokenize
from lib.config import config_file_to_dict, str2bool, str2list
from lib.label import PADDING
from chem_sentencepiece.chem_sentencepiece import ChemSentencePiece


class Dataset:
    def __init__(self):
        self.MAX_SENTENCE_LENGTH = 1000
        self.MAX_WORD_LENGTH = -1
        self.number_normalized = True
        self.cammel_normalized = False
        self.norm_word_emb = False
        self.norm_char_emb = False
        self.norm_sw_emb = False
        self.word_alphabet = Alphabet('word')
        self.char_alphabet = Alphabet('character')

        ### SubWord ###
        self.sw_num = 0
        self.sentence_piece_dirs = []
        self.sentence_piece_models = []
        self.sw_alphabet_list = [Alphabet('sw') for i in range(self.sw_num)]

        self.label_alphabet = Alphabet('label', True)
        self.tagScheme = "BMES" ## BMES/BIO

        ### I/O
        self.train_dir = None
        self.dev_dir = None
        self.test_dir = None
        self.raw_dir = None
        self.lm_dir = None

        self.decode_dir = None
        self.dset_dir = None ## data vocabulary related file
        self.model_dir = None ## model save  file
        self.pretrain_model_dir = None
        self.load_model_dir = None ## model load file

        self.word_emb_dir = None
        self.char_emb_dir = None
        self.sw_emb_dirs = []

        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []
        self.raw_texts = []
        self.lm_texts = []

        self.train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []
        self.raw_Ids = []
        self.lm_Ids = []

        self.pretrain_word_embedding = None
        self.pretrain_char_embedding = None
        self.pretrain_sw_embeddings = []

        self.label_size = 0
        self.word_emb_dim = 50
        self.char_emb_dim = 30
        self.sw_emb_dim = 50

        ## Training
        self.average_batch_loss = False
        self.optimizer = "SGD" ## "SGD"/"AdaGrad"/"AdaDelta"/"RMSProp"/"Adam"


    def build_alphabet(self, input_file, mode="ner"):
        print("build alphabet")
        if mode == "ner":
            _iter = iter_seq_data(input_file)
        elif mode == "lm":
            _iter = iter_text_data(input_file)
        for i, (word, label) in enumerate(_iter):
            if word:
                if self.number_normalized:
                    word = normalize_word(word)
                if self.cammel_normalized:
                    word = cammel_normalize_word(word)
                self.word_alphabet.add(word)
                if mode == "ner":
                    self.label_alphabet.add(label)
                for char in word:
                    self.char_alphabet.add(char)
                ####### Sub Word ######
                for sw_id, sp in enumerate(self.sentence_piece_models):
                    for sw in sp.tokenize(word):
                        if self.number_normalized:
                            sw = normalize_word(sw)
                        if self.cammel_normalized:
                            sw = cammel_normalize_word(sw)
                        self.sw_alphabet_list[sw_id].add(sw)

    def describe(self):
        """debug用にデータを表示する。"""
        print("=========== describe Dataset ===========")
        print(f"word_alphabet: {self.word_alphabet.size()}")
        print(f"subword_alphabet: {[a.size() for a in self.sw_alphabet_list]}")
        print(f"char_alphabet: {self.char_alphabet.size()}")
        print(f"label_alphabet: {self.label_alphabet.size()}")


    def fix_alphabet(self):
        self.word_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet.close()
        [self.sw_alphabet_list[i].close() for i in range(len(self.sw_alphabet_list))]


    def build_pretrain_emb(self):
        print("build pretrain embed")
        if self.word_emb_dir:
            print("Load pretrained word embedding, norm: %s, dir: %s"%(self.norm_word_emb, self.word_emb_dir))
            self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(self.word_emb_dir, self.word_alphabet, self.word_emb_dim, self.norm_word_emb)
        if self.char_emb_dir:
            print("Load pretrained char embedding, norm: %s, dir: %s"%(self.norm_char_emb, self.char_emb_dir))
            self.pretrain_char_embedding, self.char_emb_dim = build_pretrain_embedding(self.char_emb_dir, self.char_alphabet, self.char_emb_dim, self.norm_char_emb)
        if self.sw_emb_dirs:
            for i in range(len(self.sw_emb_dirs)):
                print("Load pretrained sw embedding, norm: %s, dir: %s"%(self.norm_sw_emb, self.sw_emb_dirs[i]))
                pretrain_sw_embedding, _ = build_pretrain_embedding(self.sw_emb_dirs[i], self.sw_alphabet_list[i], self.sw_emb_dim, self.norm_sw_emb)
                self.pretrain_sw_embeddings.append(pretrain_sw_embedding)

    def generate_instance(self, name, mode="ner"):
        self.fix_alphabet()
        if mode == "ner":
            if name == "train":
                print("============== read train instance ====================")
                self.train_texts, self.train_Ids = read_instance(self.train_dir, self.word_alphabet, self.char_alphabet, self.sw_alphabet_list, self.label_alphabet, self.number_normalized, self.cammel_normalized, self.MAX_SENTENCE_LENGTH, self.sentence_piece_models)
            elif name == "dev":
                print("============== read dev instance ====================")
                self.dev_texts, self.dev_Ids = read_instance(self.dev_dir, self.word_alphabet, self.char_alphabet, self.sw_alphabet_list, self.label_alphabet, self.number_normalized, self.cammel_normalized, self.MAX_SENTENCE_LENGTH, self.sentence_piece_models)
            elif name == "test":
                print("============== read test instance ====================")
                self.test_texts, self.test_Ids = read_instance(self.test_dir, self.word_alphabet, self.char_alphabet, self.sw_alphabet_list, self.label_alphabet, self.number_normalized, self.cammel_normalized, self.MAX_SENTENCE_LENGTH, self.sentence_piece_models)
            elif name == "raw":
                print("============== read raw instance ====================")
                self.raw_texts, self.raw_Ids = read_instance(self.raw_dir, self.word_alphabet, self.char_alphabet, self.sw_alphabet_list, self.label_alphabet, self.number_normalized, self.cammel_normalized, self.MAX_SENTENCE_LENGTH, self.sentence_piece_models)
            elif name == "lm":
                print("============== read lm instance ====================")
                self.lm_texts, self.lm_Ids = read_instance(self.lm_dir, self.word_alphabet, self.char_alphabet, self.sw_alphabet_list, False, self.number_normalized, self.cammel_normalized, self.MAX_SENTENCE_LENGTH, self.sentence_piece_models, mode="lm")
            else:
                print("Error: you can only generate train/dev/test instance! Illegal input:%s"%(name))
        
    def write_decoded_results(self, predict_results, name):
        fout = open(self.decode_dir,'w')
        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
           content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            print("Error: illegal name during writing predict result, name should be within train/dev/test/raw !")
        assert(sent_num == len(content_list))
        for idx in range(sent_num):
            sent_length = len(predict_results[idx])
            for idy in range(sent_length):
                ## content_list[idx] is a list with [word, char, label]
                #fout.write(content_list[idx][0][idy] + "\t" + predict_results[idx][idy] + '\n')
                fout.write(predict_results[idx][idy] + '\n')
            fout.write('\n')
        fout.close()
        print("Predict %s result has been written into file. %s"%(name, self.decode_dir))

    def load(self,data_file):
        print(f"============ Load Dset: {data_file} ===============")
        f = open(data_file, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)
        self.sentence_piece_models = []
        for sp_path in tmp_dict['sentence_piece_dirs']:
            chem_sp = ChemSentencePiece()
            chem_sp.load(sp_path)
            self.sentence_piece_models.append(chem_sp)

    def save(self,save_file):
        f = open(save_file, 'wb')
        dic =  self.__dict__
        del dic['sentence_piece_models']
        pickle.dump(dic, f, 2)
        f.close()

    def write_nbest_decoded_results(self, predict_results, pred_scores, name):
        ## predict_results : [whole_sent_num, nbest, each_sent_length]
        ## pred_scores: [whole_sent_num, nbest]
        fout = open(self.decode_dir,'w')
        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
           content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            print("Error: illegal name during writing predict result, name should be within train/dev/test/raw !")
        assert(sent_num == len(content_list))
        assert(sent_num == len(pred_scores))
        for idx in range(sent_num):
            sent_length = len(predict_results[idx][0])
            nbest = len(predict_results[idx])
            score_string = "# "
            for idz in range(nbest):
                score_string += format(pred_scores[idx][idz], '.4f')+" "
            fout.write(score_string.strip() + "\n")
            for idy in range(sent_length):
                try:  # Will fail with python3
                    label_string = content_list[idx][0][idy].encode('utf-8') + " "
                except:
                    label_string = content_list[idx][0][idy] + " "
                for idz in range(nbest):
                    label_string += predict_results[idx][idz][idy]+" "
                label_string = label_string.strip() + "\n"
                fout.write(label_string)
            fout.write('\n')
        fout.close()
        print("Predict %s %s-best result has been written into file. %s"%(name,nbest, self.decode_dir))

    def read_config(self,config_file):
        config = config_file_to_dict(config_file)
        self.__dict__.update(config)
        self.sw_alphabet_list = [Alphabet(f'sw{i}') for i in range(self.sw_num)]
        for sp_path in self.sentence_piece_dirs:
            _sp = ChemSentencePiece()
            _sp.load(sp_path)
            self.sentence_piece_models.append(_sp)
        
    def load_lm_dset_alphabets(self, lm_dataset):
        """large corpusで学習した際のアルファベットをしようする。"""
        self.word_alphabet = lm_dataset.word_alphabet
        self.char_alphabet = lm_dataset.char_alphabet
        self.sw_alphabet_list = lm_dataset.sw_alphabet_list
        labels = set([label for _, label in iter_seq_data(self.train_dir)])
        for label in labels:
            if type(label) != bool:
                self.label_alphabet.add(label)
        

def read_instance(
    input_file,
    word_alphabet, 
    char_alphabet, 
    sw_alphabet_list, 
    label_alphabet, 
    number_normalized, 
    cammel_normalized, 
    max_sent_length, 
    sentencepieces, 
    char_padding_size=-1, 
    char_padding_symbol=PADDING,
    mode="ner"
):
    ### Initialization ###
    sw_num = len(sentencepieces)
    instance_texts, instance_Ids = [], []
    words, word_Ids = [], []
    chars, char_Ids = [], []
    sws_list, sw_Ids_list = [[] for _ in range(sw_num)], [[] for _ in range(sw_num)]
    sw_fmasks_list = [[] for _ in range(sw_num)]
    sw_bmasks_list = [[] for _ in range(sw_num)]
    labels, label_Ids = [], []

    if mode == "ner":
        _iter = iter_seq_data(input_file)
    elif mode == "lm":
        _iter = iter_text_data(input_file)
    for i, (word, label) in enumerate(_iter):
        if word:
            if number_normalized:
                word = normalize_word(word)
            if cammel_normalized:
                word = cammel_normalize_word(word)
            words.append(word)
            word_Ids.append(word_alphabet.get_index(word))
            if mode == "ner":
                labels.append(label)
                label_Ids.append(label_alphabet.get_index(label))
            ### get char
            char_list, char_Id = [], []
            for char in word:
                char_list.append(char)
            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
                assert len(char_list) == char_padding_size, "Failed Char Padding."
            for char in char_list:
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)
            ## deal with sw
            for idx, sp in enumerate(sentencepieces):
                sw_list, sw_Id, sw_fmask, sw_bmask = [], [], [], []
                for sw in sp.tokenize(word):
                    if number_normalized:
                        sw = normalize_word(sw)
                    if cammel_normalized:
                        word = cammel_normalize_word(word)
                    sw_list.append(sw)
                    sw_Id.append(sw_alphabet_list[idx].get_index(sw))
                    sw_fmask.append(0)
                    sw_bmask.append(0)
                sw_fmask[-1] = 1
                sw_bmask[0] = 1
                sw_fmasks_list[idx].extend(sw_fmask)
                sw_bmasks_list[idx].extend(sw_bmask)
                sws_list[idx].extend(sw_list)
                sw_Ids_list[idx].extend(sw_Id)
        ### append new Instance
        else:
            if len(word_Ids) <= max_sent_length:
                if mode == "ner":
                    instance_texts.append([words, chars, sws_list, labels])
                    instance_Ids.append([word_Ids, char_Ids, sw_Ids_list, sw_fmasks_list, sw_bmasks_list, label_Ids])
                elif mode == "lm":
                    # 省メモリのため。
                    #instance_texts.append([words, chars, sws_list])
                    instance_Ids.append([word_Ids, char_Ids, sw_Ids_list, sw_fmasks_list, sw_bmasks_list])
            words, word_Ids = [], []
            chars, char_Ids = [], []
            sws_list, sw_Ids_list = [[] for _ in range(sw_num)], [[] for _ in range(sw_num)]
            sw_fmasks_list, sw_bmasks_list = [[] for _ in range(sw_num)], [[] for _ in range(sw_num)]
            labels, label_Ids = [], []
    # 残りのinstanceを加える。
    if words and word_Ids:
        if mode == "ner":
            instance_texts.append([words, chars, sws_list, labels])
            instance_Ids.append([word_Ids, char_Ids, sw_Ids_list, sw_fmasks_list, sw_bmasks_list, label_Ids])
        elif mode == "lm":
            #instance_texts.append([words, chars, sws_list])
            instance_Ids.append([word_Ids, char_Ids, sw_Ids_list, sw_fmasks_list, sw_bmasks_list])
    return instance_texts, instance_Ids


def iter_seq_data(file_name):
    """parse seq data"""
    in_lines = open(file_name,'r').readlines()
    for z, line in tqdm(enumerate(in_lines)):
        if len(line) > 2:
            pairs = line.strip().split()
            yield pairs[0], pairs[1]
        else:
            yield False, False


def iter_text_data(file_name):
    """parse text data"""
    in_lines = open(file_name,'r').readlines()
    for i, line in tqdm(enumerate(in_lines)):
        if len(line) > 2:
            for word in tokenize(line):
                yield word, False
        yield False, False
