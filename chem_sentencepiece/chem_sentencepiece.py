import re
import os
import shutil
from typing import List
import argparse

import sentencepiece as spm


class ChemSentencePiece:
    spe: spm.SentencePieceProcessor = spm.SentencePieceProcessor()

    def __init__(self):
        pass

    @classmethod
    def load(self, sp_path: str):
        self.spe.Load(sp_path)

    @classmethod    
    def train(self, corpus_path: str, vocab_size: int):
        spm.SentencePieceTrainer.Train("--normalization_rule_name=identity --input={} --model_prefix=sp{}  --vocab_size={}  --model_type=unigram".format(corpus_path, vocab_size, vocab_size))
        shutil.move(f"sp{vocab_size}.model", os.path.join(self.SP_DIR, f"sp{vocab_size}.model"))
        shutil.move(f"sp{vocab_size}.vocab", os.path.join(self.SP_DIR, f"sp{vocab_size}.vocab"))
        os.remove(f"sp{vocab_size}.model")
        os.remove(f"sp{vocab_size}.vocab")
        print(f"sp{vocab_size} moved {self.SP_DIR} !!")

    def tokenize(self, text: str, B_TAG=True) -> List[str]:
        """B_TAGがTrueのときはSubwordの先頭を表す▁を表示する。"""
        tokens: list = []
        if len(text) <= 1:
            if not B_TAG:
                return [text]
            else:
                return [f"▁{text}"]
        else:
            for token in self.spe.EncodeAsPieces(text):
                if not B_TAG:
                    token = token.replace('▁', '')
                tokens.append(token)
            return tokens

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Sentence Piece')
    parser.add_argument('-i', '--input', type=str, help='Input Large Corpus')
    parser.add_argument('-v', '--vocab', type=int, help='Vocaburaly size')
    args = parser.parse_args()
    ChemSentencePiece.train(args.input, args.vocab)
