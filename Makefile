train-sp:
	python3.7 chem_sentencepiece/chem_sentencepiece.py --vocab 4000 --input ./Repository/LargeCorpus/large_corpus_4400k.txt

train-gv:
	python3.7 word2vec/GloVe.py --input ./Repository/LargeCorpus/large_corpus_4400k.txt  --vocab-min 5 --size 50 --iter 50 --window 15 --out ./Repository/GloVe/gv.d50

train-lm:
	python3.7 languagemodel/train.py --config config/csw.4k_16k.config

train-ner:
	CUDA_VISIBLE_DEVICES=0 python3.7 ner/train.py --config config/csw.4k_16k.config
