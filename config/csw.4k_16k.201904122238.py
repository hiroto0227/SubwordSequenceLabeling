import os

### Train Description ###

### TrainName ###
config_name = "sw4k.16k.lm4400k"

### NER input ###
ner_corpus_dir = "/home/sekine/SubwordSequenceLabeling/Repository/Chemdner/"
ner_train_path = os.path.join(ner_corpus_dir, "train.bioes")
ner_valid_path = os.path.join(ner_corpus_dir, "valid.bioes")
ner_test_path = os.path.join(ner_corpus_dir, "test.bioes")

### NER output ###
ner_model_dir = "/home/sekine/SubwordSequenceLabeling/Repository/NERModel/"

### LanguageModel input ###
lm_corpus_path = "/home/sekine/SubwordSequenceLabeling/Repository/LargeCorpus/large_corpus_4400k.txt"

### LanguageModel output ###
language_model_dir = "/home/sekine/SubwordSequenceLabeling/Repository/LanguageModel/"

### PretrainGlove ###
word_emb_dir = "/home/sekine/SubwordSequenceLabeling/Repository/GloVe/gv.d50"

### Subword ###
sentence_piece_dirs=["/home/sekine/SubwordSequenceLabeling/Repository/SentencePiece/sp4000.model", 
                     "/home/sekine/SubwordSequenceLabeling/Repository/SentencePiece/sp16000.model"]

### Model parameter ###
word_emb_dim=50
char_emb_dim=30
sw_emb_dim=50
char_hidden_dim=50
sw_hidden_dim=50
hidden_dim=200
dropout=0.5
lstm_layer=1
bilstm=True

### Train config ###
ner_epoch=50
lm_epoch=10
lm_vocab_min_count=10
batch_size=10
optimizer=SGD
lr=0.005
lr_decay=0.0001
momentum=0
l2=1e-8
clip=5.0
ave_batch_loss=False

### Preprocess ###
norm_word_emb=False
norm_char_emb=False
number_normalized=True
cammel_normalized=True

### Hardware ###
gpu=True
