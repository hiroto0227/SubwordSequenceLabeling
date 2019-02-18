train-sp:
	python3.7 chem_sentencepiece/chem_sentencepiece.py --vocab 2000 --input ./Repository/LargeCorpus/large_corpus.txt

train-gv:
	python3.7 word2vec/GloVe.py --input ./Repository/LargeCorpus/large_corpus.txt --vocab-min 5 --size 50 --iter 50 --window 15 --out ./Repository/GloVe/gv.d50

train-lm:
	python3.7 languagemodel/train.py --config config/fujitsu.test.config

train-ner:
	python3.7 ner/train.py --config ner/config/sw4k.config
