import json
import os
from pathlib import Path
from random import choice
from typing import List, Dict, Iterable

from datasets import tqdm
from nltk.corpus import stopwords
from spacy.tokens import Span


class StopWordDictionary:
    STOP_WORDS = stopwords.words('spanish')

    def __init__(self, stop_words_by_pos: Dict[str, List[str]]):
        self.stop_words_by_pos = stop_words_by_pos

    def random_stop_word(self, pos: str, excluded_word):
        for i in range(3):
            stop_word_choices = self.stop_words_by_pos.get(pos)
            if not stop_word_choices:
                break
            stop_word = choice(stop_word_choices)
            if stop_word != excluded_word:
                return stop_word
        return None

    def save_to_json(self, path: Path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as file:
            json.dump(self.stop_words_by_pos, file)

    @classmethod
    def load_from_json(cls, path: Path):
        with open(path, 'r') as file:
            stop_words_by_pos = json.load(file)
        return cls(stop_words_by_pos)

    @classmethod
    def create_from_sentences(cls, sentences: Iterable[Span]):
        stop_words_by_pos = {}
        for sentence in tqdm(sentences):
            for token in sentence:
                if token.text.lower() in cls.STOP_WORDS:
                    pos_stop_words = stop_words_by_pos.get(token.pos_, set())
                    pos_stop_words.add(token.text)
                    stop_words_by_pos[token.pos_] = pos_stop_words
        stop_words_by_pos = {pos: list(stop_words) for pos, stop_words in stop_words_by_pos.items()}
        return cls(stop_words_by_pos)
