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
        self.MAX_SENTENCE_LENGTH = 10000
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
        self.lm_model_dir = None
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
        self.word_alphabet_size = 0
        self.char_alphabet_size = 0
        self.sw_alphabet_size_list = []
        self.label_alphabet_size = 0
        self.word_emb_dim = 50
        self.char_emb_dim = 30
        self.sw_emb_dim = 50

        ## Training
        self.average_batch_loss = False
        self.optimizer = "SGD" ## "SGD"/"AdaGrad"/"AdaDelta"/"RMSProp"/"Adam"

        ### Hyperparameters
        self.HP_cnn_layer = 4
        self.HP_iteration = 100
        self.HP_batch_size = 10
        self.HP_char_hidden_dim = 50
        self.HP_sw_hidden_dim=50
        self.HP_hidden_dim = 200
        self.HP_dropout = 0.5
        self.HP_lstm_layer = 1
        self.HP_bilstm = True

        self.HP_gpu = False
        self.HP_lr = 0.015
        self.HP_lr_decay = 0.05
        self.HP_clip = None
        self.HP_momentum = 0
        self.HP_l2 = 1e-8


    def build_alphabet(self, input_file):
        in_lines = open(input_file,'r').readlines()
        for i, line in enumerate(in_lines):
            # large corpusのとき
            if len(line) > 3:
                for word in tokenize(line):
                    if sys.version_info[0] < 3:
                        word = word.decode('utf-8')
                    if self.number_normalized:
                        word = normalize_word(word)
                    if self.cammel_normalized:
                        word = cammel_normalize_word(word)
                    self.word_alphabet.add(word)
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
            # seq dataのとき
            elif len(line) == 2:
                pairs = line.strip().split()
                word = pairs[0]
                if sys.version_info[0] < 3:
                    word = word.decode('utf-8')
                if self.number_normalized:
                    word = normalize_word(word)
                if self.cammel_normalized:
                    word = cammel_normalize_word(word)
                label = pairs[-1]
                self.label_alphabet.add(label)
                self.word_alphabet.add(word)
                for char in word:
                    self.char_alphabet.add(char)
                ####### Sub Word ######
                #print("=======================================")
                for sw_id, sp in enumerate(self.sentence_piece_models):
                    for sw in sp.tokenize(word):
                        if self.number_normalized:
                            sw = normalize_word(sw)
                        if self.cammel_normalized:
                            sw = cammel_normalize_word(sw)
                        self.sw_alphabet_list[sw_id].add(sw)
        print(self.word_alphabet.instance2index)
        self.word_alphabet_size = self.word_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()
        self.sw_alphabet_size_list = [sw_a.size() for sw_a in self.sw_alphabet_list]

    def fix_alphabet(self):
        self.word_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet.close()
        [self.sw_alphabet_list[i].close() for i in range(len(self.sw_alphabet_list))]


    def build_pretrain_emb(self):
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
                self.train_texts, self.train_Ids = read_instance(self.train_dir, self.word_alphabet, self.char_alphabet, self.sw_alphabet_list, self.label_alphabet, self.number_normalized, self.cammel_normalized, self.MAX_SENTENCE_LENGTH, self.sentence_piece_models)
            elif name == "dev":
                self.dev_texts, self.dev_Ids = read_instance(self.dev_dir, self.word_alphabet, self.char_alphabet, self.sw_alphabet_list, self.label_alphabet, self.number_normalized, self.cammel_normalized, self.MAX_SENTENCE_LENGTH, self.sentence_piece_models)
            elif name == "test":
                self.test_texts, self.test_Ids = read_instance(self.test_dir, self.word_alphabet, self.char_alphabet, self.sw_alphabet_list, self.label_alphabet, self.number_normalized, self.cammel_normalized, self.MAX_SENTENCE_LENGTH, self.sentence_piece_models)
            elif name == "raw":
                self.raw_texts, self.raw_Ids = read_instance(self.raw_dir, self.word_alphabet, self.char_alphabet, self.sw_alphabet_list, self.label_alphabet, self.number_normalized, self.cammel_normalized, self.MAX_SENTENCE_LENGTH, self.sentence_piece_models)
            elif name == "lm":
                self.lm_texts, self.lm_Ids = read_lm_instance(self.lm_dir, self.word_alphabet, self.char_alphabet, self.sw_alphabet_list, self.number_normalized, self.cammel_normalized, self.MAX_SENTENCE_LENGTH, self.sentence_piece_models)
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
        f = open(data_file, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)
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
        ## read data:
        the_item = 'train_dir'
        if the_item in config:
            self.train_dir = config[the_item]
        the_item = 'dev_dir'
        if the_item in config:
            self.dev_dir = config[the_item]
        the_item = 'test_dir'
        if the_item in config:
            self.test_dir = config[the_item]
        the_item = 'raw_dir'
        if the_item in config:
            self.raw_dir = config[the_item]
        the_item = 'lm_dir'
        if the_item in config:
            self.lm_dir = config[the_item]
        the_item = 'decode_dir'
        if the_item in config:
            self.decode_dir = config[the_item]
        the_item = 'dset_dir'
        if the_item in config:
            self.dset_dir = config[the_item]
        the_item = 'model_dir'
        if the_item in config:
            self.model_dir = config[the_item]
        the_item = 'load_model_dir'
        if the_item in config:
            self.load_model_dir = config[the_item]

        the_item = 'word_emb_dir'
        if the_item in config:
            self.word_emb_dir = config[the_item]
        the_item = 'char_emb_dir'
        if the_item in config:
            self.char_emb_dir = config[the_item]

        ### SubWord ###
        the_item = 'sw_num'
        if the_item in config:
            self.sw_num = int(config[the_item])
            self.sw_alphabet_list = [Alphabet('sw') for i in range(self.sw_num)]
        the_item = 'sw_emb_dirs'
        if the_item in config:
            self.sw_emb_dirs = str2list(config[the_item])
        the_item = 'sentence_piece_dirs'
        if the_item in config:
            for sp_path in str2list(config[the_item]):
                chem_spe = ChemSentencePiece()
                self.sentence_piece_dirs.append(sp_path)
                chem_spe.load(sp_path)
                self.sentence_piece_models.append(chem_spe)
        the_item = 'MAX_SENTENCE_LENGTH'
        if the_item in config:
            self.MAX_SENTENCE_LENGTH = int(config[the_item])
        the_item = 'MAX_WORD_LENGTH'
        if the_item in config:
            self.MAX_WORD_LENGTH = int(config[the_item])

        the_item = 'norm_word_emb'
        if the_item in config:
            self.norm_word_emb = str2bool(config[the_item])
        the_item = 'norm_char_emb'
        if the_item in config:
            self.norm_char_emb = str2bool(config[the_item])
        the_item = 'norm_sw_emb'
        if the_item in config:
            self.norm_sw_emb = str2bool(config[the_item])
        the_item = 'number_normalized'
        if the_item in config:
            self.number_normalized = str2bool(config[the_item])
        the_item = 'cammel_normalized'
        if the_item in config:
            self.cammel_normalized = str2bool(config[the_item])
        the_item = 'seg'
        if the_item in config:
            self.seg = str2bool(config[the_item])
        the_item = 'word_emb_dim'
        if the_item in config:
            self.word_emb_dim = int(config[the_item])
        the_item = 'char_emb_dim'
        if the_item in config:
            self.char_emb_dim = int(config[the_item])
        the_item = 'sw_emb_dim'
        if the_item in config:
            self.sw_emb_dim = int(config[the_item])

        ## read training setting:
        the_item = 'optimizer'
        if the_item in config:
            self.optimizer = config[the_item]
        the_item = 'ave_batch_loss'
        if the_item in config:
            self.average_batch_loss = str2bool(config[the_item])
        the_item = 'status'
        if the_item in config:
            self.status = config[the_item]

        ## read Hyperparameters:
        the_item = 'cnn_layer'
        if the_item in config:
            self.HP_cnn_layer = int(config[the_item])
        the_item = 'iteration'
        if the_item in config:
            self.HP_iteration = int(config[the_item])
        the_item = 'batch_size'
        if the_item in config:
            self.HP_batch_size = int(config[the_item])

        the_item = 'char_hidden_dim'
        if the_item in config:
            self.HP_char_hidden_dim = int(config[the_item])
        the_item = 'sw_hidden_dim'
        if the_item in config:
            self.HP_sw_hidden_dim = int(config[the_item])
        the_item = 'hidden_dim'
        if the_item in config:
            self.HP_hidden_dim = int(config[the_item])
        the_item = 'dropout'
        if the_item in config:
            self.HP_dropout = float(config[the_item])
        the_item = 'lstm_layer'
        if the_item in config:
            self.HP_lstm_layer = int(config[the_item])
        the_item = 'bilstm'
        if the_item in config:
            self.HP_bilstm = str2bool(config[the_item])

        the_item = 'gpu'
        if the_item in config:
            self.HP_gpu = str2bool(config[the_item])
        the_item = 'learning_rate'
        if the_item in config:
            self.HP_lr = float(config[the_item])
        the_item = 'lr_decay'
        if the_item in config:
            self.HP_lr_decay = float(config[the_item])
        the_item = 'clip'
        if the_item in config:
            self.HP_clip = float(config[the_item])
        the_item = 'momentum'
        if the_item in config:
            self.HP_momentum = float(config[the_item])
        the_item = 'l2'
        if the_item in config:
            self.HP_l2 = float(config[the_item])


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
    char_padding_symbol=PADDING
):
    ### Initialization ###
    sw_num = len(sentencepieces)
    in_lines = open(input_file,'r').readlines()
    instance_texts, instance_Ids = [], []
    words, word_Ids = [], []
    chars, char_Ids = [], []
    sws_list, sw_Ids_list = [[] for _ in range(sw_num)], [[] for _ in range(sw_num)]
    sw_fmasks_list = [[] for _ in range(sw_num)]
    sw_bmasks_list = [[] for _ in range(sw_num)]
    labels, label_Ids = [], []

    ### for sequence labeling data format i.e. CoNLL 2003
    for z, line in tqdm(enumerate(in_lines)):
        if len(line) > 2:
            ### get words and labels
            pairs = line.strip().split()
            word = pairs[0]
            if sys.version_info[0] < 3:
                word = word.decode(CHARSET)
            if number_normalized:
                word = normalize_word(word)
            if self.cammel_normalized:
                word = cammel_normalize_word(word)
            label = pairs[-1]
            words.append(word)
            labels.append(label)
            word_Ids.append(word_alphabet.get_index(word))
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
                    if self.cammel_normalized:
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
        else:
            ### append new Instance
            if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)) :
                instance_texts.append([words, chars, sws_list, labels])
                instance_Ids.append([word_Ids, char_Ids, sw_Ids_list, sw_fmasks_list, sw_bmasks_list, label_Ids])
            words, word_Ids = [], []
            chars, char_Ids = [], []
            sws_list, sw_Ids_list = [[] for _ in range(sw_num)], [[] for _ in range(sw_num)]
            sw_fmasks_list, sw_bmasks_list = [[] for _ in range(sw_num)], [[] for _ in range(sw_num)]
            labels, label_Ids = [], []

    if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)) :
        instance_texts.append([words, chars, sws_list, labels])
        instance_Ids.append([word_Ids, char_Ids, sw_Ids_list, sw_fmasks_list, sw_bmasks_list, label_Ids])
        words, word_Ids = [], []
        chars, char_Ids = [], []
        sws_list, sw_Ids_list = [[] for _ in range(sw_num)], [[] for _ in range(sw_num)]
        sw_fmasks_list, sw_bmasks_list = [[] for _ in range(sw_num)], [[] for _ in range(sw_num)]
        labels, label_Ids = [], []
    return instance_texts, instance_Ids


def read_lm_instance(
    input_file,
    word_alphabet, 
    char_alphabet, 
    sw_alphabet_list, 
    number_normalized, 
    cammel_normalized,
    max_sent_length, 
    sentencepieces, 
    char_padding_size=-1, 
    char_padding_symbol=PADDING
):
    ### Initialization ###
    sw_num = len(sentencepieces)
    in_lines = open(input_file,'r').read()
    instance_texts, instance_Ids = [], []
    words, word_Ids = [], []
    chars, char_Ids = [], []
    sws_list, sw_Ids_list = [[] for _ in range(sw_num)], [[] for _ in range(sw_num)]
    sw_fmasks_list = [[] for _ in range(sw_num)]
    sw_bmasks_list = [[] for _ in range(sw_num)]

    ### for sequence labeling data format i.e. CoNLL 2003
    for z, line in tqdm(enumerate(in_lines.split("\n"))):
        # if z == 200:
        #     break
        for word in tokenize(line):
            ### get word
            if number_normalized:
                word = normalize_word(word)
            if cammel_normalized:
                word = cammel_normalize_word(word)
            words.append(word)
            word_Ids.append(word_alphabet.get_index(word))
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
        if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)):
            instance_texts.append([words, chars, sws_list])
            instance_Ids.append([word_Ids, char_Ids, sw_Ids_list, sw_fmasks_list, sw_bmasks_list])
        words, word_Ids = [], []
        chars, char_Ids = [], []
        sws_list, sw_Ids_list = [[] for _ in range(sw_num)], [[] for _ in range(sw_num)]
        sw_fmasks_list, sw_bmasks_list = [[] for _ in range(sw_num)], [[] for _ in range(sw_num)]

    if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)) :
        instance_texts.append([words, chars, sws_list])
        instance_Ids.append([word_Ids, char_Ids, sw_Ids_list, sw_fmasks_list, sw_bmasks_list])
        words, word_Ids = [], []
        chars, char_Ids = [], []
        sws_list, sw_Ids_list = [[] for _ in range(sw_num)], [[] for _ in range(sw_num)]
        sw_fmasks_list, sw_bmasks_list = [[] for _ in range(sw_num)], [[] for _ in range(sw_num)]
    return instance_texts, instance_Ids
