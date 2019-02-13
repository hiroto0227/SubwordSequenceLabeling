import sys
from typing import List

from tqdm import tqdm
import torch

from lib.charset import CHARSET
from chem_sentencepiece.chem_sentencepiece import ChemSentencePiece
from lib.alphabet import Alphabet


class LargeCorpus:
    sentences: List[str]

    def __init__(self):
        pass
    
    @classmethod
    def load(self, path: str):
        with open(path, "rt", encoding=CHARSET) as f:
            self.sentences = f.read().split("\n")

    def save_as_sptrain(self, sp_model_path: str, outpath: str):
        sp = ChemSentencePiece.load(sp_model_path)
        with open(outpath, "at", encoding=CHARSET) as f:
            for sentence in tqdm(self.sentences):
                f.write(sp.tokenize(sentence))

    def batchify_for_lm(self, batch_size: int) -> torch.LongTensor:
        return torch.zeros(batch_size, batch_size).long() 
