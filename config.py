config_dic = {
    "test": {},
    "train_name": "test",
    "lm_input_path": "./Repository/LargeCorpus/large_corpus_4400k.txt",
    "lm_model_dir": "./Repository/LanguageModel/",
    "sp_path": "./Repository/SentencePiece/sp4000.model",
    "vocab_dir": "./Repository/Vocabulary/",
    "ner_input_dir": "./Repository/Chemdner/",
    "ner_model_dir": "./Repository/NERModel/",

    "word_emb_dim": 50,
    "char_emb_dim": 30,
    "sw_emb_dim": 50,
    "word_hidden_dim": 100,
    "char_hidden_dim": 100,
    "sw_hidden_dim": 100,

    "lm_epoch": 1,
    "ner_epoch": 50,
    "lm_batch_size": 10,
    "ner_batch_size": 10,
    "lm_lr": 0.5,
    "ner_lr": 0.5,
    
    "number_normalize": True,
    "dropout": 0.5,
    "gpu": True
}
