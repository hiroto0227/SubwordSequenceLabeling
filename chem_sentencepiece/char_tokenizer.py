from typing import List


class CharTokenizer:
    def tokenize(self, word: str) -> List[str]:
        return ["▁{}".format(c) if i == 0 else c for i, c in enumerate(word)]