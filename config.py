config_dics = {
    ############################### Base Line (NoLM) #########################################
    "BL.2000k.NoLM": {
        # データに関わるconfig
        "lm_input_path": "./Repository/LargeCorpus/large_corpus_2000k.txt",
        "lm_model_dir": "./Repository/LanguageModel/",
        "sp_path": {},
        "vocab_dir": "./Repository/Vocabulary/",
        "ner_input_dir": "./Repository/Chemdner/",
        "ner_model_dir": "./Repository/NERModel/",
        "cache_dir": "./Repository/Cache/",
        "glove_path": "./Repository/GloVe/gv.d50",
        "number_normalize": True,
        # モデルに関わるconfig
        "word_emb_dim": 50,
        "char_emb_dim": 30,
        "sw_emb_dim": 50,
        "word_hidden_dim": 200,
        "char_hidden_dim": 100,
        "sw_hidden_dim": 100,
        "ner_dropout": 0.5,
        "lm_dropout": 0.5,
        # 学習条件に関わるconfig
        "lm_epoch": 0,
        "ner_epoch": 100,
        "lm_batch_size": 10,
        "ner_batch_size": 10,
        "lm_lr": 5,
        "ner_lr": 0.01,
        "weight_decay": 1e-5,
        "grad_clip": 0,
        "gpu": True
    }, 


    ############################### Base Line (LM) #########################################
    "BL.2000k.LM": {
        # データに関わるconfig
        "lm_input_path": "./Repository/LargeCorpus/large_corpus_2000k.txt",
        "lm_model_dir": "./Repository/LanguageModel/",
        "sp_path": {},
        "vocab_dir": "./Repository/Vocabulary/",
        "ner_input_dir": "./Repository/Chemdner/",
        "ner_model_dir": "./Repository/NERModel/",
        "cache_dir": "./Repository/Cache/",
        "glove_path": "./Repository/GloVe/gv.d50",
        "number_normalize": True,
        # モデルに関わるconfig
        "word_emb_dim": 50,
        "char_emb_dim": 30,
        "sw_emb_dim": 50,
        "word_hidden_dim": 200,
        "char_hidden_dim": 100,
        "sw_hidden_dim": 100,
        "ner_dropout": 0.5,
        "lm_dropout": 0.5,
        # 学習条件に関わるconfig
        "lm_epoch": 1,
        "ner_epoch": 100,
        "lm_batch_size": 10,
        "ner_batch_size": 10,
        "lm_lr": 1,
        "ner_lr": 0.01,
        "weight_decay": 1e-5,
        "grad_clip": 0,
        "gpu": True
    }, 
    
    ############################# SW2k (NoLM) #################################
    "SW2k.2000k.NoLM": {
        # データに関わるconfig
        "lm_input_path": "./Repository/LargeCorpus/large_corpus_2000k.txt",
        "lm_model_dir": "./Repository/LanguageModel/",
        "sp_path": {"SW2k": "./Repository/SentencePiece/sp2000.model"},
        "vocab_dir": "./Repository/Vocabulary/",
        "ner_input_dir": "./Repository/Chemdner/",
        "ner_model_dir": "./Repository/NERModel/",
        "cache_dir": "./Repository/Cache/",
        "glove_path": "./Repository/GloVe/gv.d50",
        "number_normalize": True,
        # モデルに関わるconfig
        "word_emb_dim": 50,
        "char_emb_dim": 30,
        "sw_emb_dim": 50,
        "word_hidden_dim": 200,
        "char_hidden_dim": 100,
        "sw_hidden_dim": 100,
        "ner_dropout": 0.5,
        "lm_dropout": 0.5,
        # 学習条件に関わるconfig
        "lm_epoch": 0,
        "ner_epoch": 100,
        "lm_batch_size": 10,
        "ner_batch_size": 10,
        "lm_lr": 1,
        "ner_lr": 0.01,
        "weight_decay": 1e-5,
        "grad_clip": 0,
        "gpu": True
    }, 

    ############################# SW4k (NoLM) #################################
    "SW4k.2000k.NoLM": {
        # データに関わるconfig
        "lm_input_path": "./Repository/LargeCorpus/large_corpus_2000k.txt",
        "lm_model_dir": "./Repository/LanguageModel/",
        "sp_path": {"SW4k": "./Repository/SentencePiece/sp4000.model",
        "vocab_dir": "./Repository/Vocabulary/",
        "ner_input_dir": "./Repository/Chemdner/",
        "ner_model_dir": "./Repository/NERModel/",
        "cache_dir": "./Repository/Cache/",
        "glove_path": "./Repository/GloVe/gv.d50",
        "number_normalize": True,
        # モデルに関わるconfig
        "word_emb_dim": 50,
        "char_emb_dim": 30,
        "sw_emb_dim": 50,
        "word_hidden_dim": 200,
        "char_hidden_dim": 100,
        "sw_hidden_dim": 100,
        "ner_dropout": 0.5,
        "lm_dropout": 0.5,
        # 学習条件に関わるconfig
        "lm_epoch": 0,
        "ner_epoch": 100,
        "lm_batch_size": 10,
        "ner_batch_size": 10,
        "lm_lr": 1,
        "ner_lr": 0.01,
        "weight_decay": 1e-5,
        "grad_clip": 0,
        "gpu": True
    }, 

    ############################# SW8k (NoLM) #################################
    "SW8k.2000k.NoLM": {
        # データに関わるconfig
        "lm_input_path": "./Repository/LargeCorpus/large_corpus_2000k.txt",
        "lm_model_dir": "./Repository/LanguageModel/",
        "sp_path": {"SW8k": "./Repository/SentencePiece/sp8000.model",
        "vocab_dir": "./Repository/Vocabulary/",
        "ner_input_dir": "./Repository/Chemdner/",
        "ner_model_dir": "./Repository/NERModel/",
        "cache_dir": "./Repository/Cache/",
        "glove_path": "./Repository/GloVe/gv.d50",
        "number_normalize": True,
        # モデルに関わるconfig
        "word_emb_dim": 50,
        "char_emb_dim": 30,
        "sw_emb_dim": 50,
        "word_hidden_dim": 200,
        "char_hidden_dim": 100,
        "sw_hidden_dim": 100,
        "ner_dropout": 0.5,
        "lm_dropout": 0.5,
        # 学習条件に関わるconfig
        "lm_epoch": 0,
        "ner_epoch": 100,
        "lm_batch_size": 10,
        "ner_batch_size": 10,
        "lm_lr": 1,
        "ner_lr": 0.01,
        "weight_decay": 1e-5,
        "grad_clip": 0,
        "gpu": True
    }, 

    ############################# SW16k (NoLM) #################################
    "SW16k.2000k.NoLM": {
        # データに関わるconfig
        "lm_input_path": "./Repository/LargeCorpus/large_corpus_2000k.txt",
        "lm_model_dir": "./Repository/LanguageModel/",
        "sp_path": {"SW16k": "./Repository/SentencePiece/sp16000.model",
        "vocab_dir": "./Repository/Vocabulary/",
        "ner_input_dir": "./Repository/Chemdner/",
        "ner_model_dir": "./Repository/NERModel/",
        "cache_dir": "./Repository/Cache/",
        "glove_path": "./Repository/GloVe/gv.d50",
        "number_normalize": True,
        # モデルに関わるconfig
        "word_emb_dim": 50,
        "char_emb_dim": 30,
        "sw_emb_dim": 50,
        "word_hidden_dim": 200,
        "char_hidden_dim": 100,
        "sw_hidden_dim": 100,
        "ner_dropout": 0.5,
        "lm_dropout": 0.5,
        # 学習条件に関わるconfig
        "lm_epoch": 0,
        "ner_epoch": 100,
        "lm_batch_size": 10,
        "ner_batch_size": 10,
        "lm_lr": 1,
        "ner_lr": 0.01,
        "weight_decay": 1e-5,
        "grad_clip": 0,
        "gpu": True
    }, 

    ############################# SW2k.4k (NoLM) #################################
    "SW2k.4k.2000k.NoLM": {
        # データに関わるconfig
        "lm_input_path": "./Repository/LargeCorpus/large_corpus_2000k.txt",
        "lm_model_dir": "./Repository/LanguageModel/",
        "sp_path": {"SW2k": "./Repository/SentencePiece/sp2000.model", 
                    "SW4k": "./Repository/SentencePiece/sp4000.model"},
        "vocab_dir": "./Repository/Vocabulary/",
        "ner_input_dir": "./Repository/Chemdner/",
        "ner_model_dir": "./Repository/NERModel/",
        "cache_dir": "./Repository/Cache/",
        "glove_path": "./Repository/GloVe/gv.d50",
        "number_normalize": True,
        # モデルに関わるconfig
        "word_emb_dim": 50,
        "char_emb_dim": 30,
        "sw_emb_dim": 50,
        "word_hidden_dim": 200,
        "char_hidden_dim": 100,
        "sw_hidden_dim": 100,
        "ner_dropout": 0.5,
        "lm_dropout": 0.5,
        # 学習条件に関わるconfig
        "lm_epoch": 0,
        "ner_epoch": 100,
        "lm_batch_size": 10,
        "ner_batch_size": 10,
        "lm_lr": 1,
        "ner_lr": 0.01,
        "weight_decay": 1e-5,
        "grad_clip": 0,
        "gpu": True
    }
}

