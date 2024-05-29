import json
import os
from pathlib import Path
from typing import Dict, List, Iterable, Optional
from random import choice

from spacy.tokens import Span
from tqdm import tqdm


class Inflector:
    MORPHING_TAGS = ['NOUN', 'VERB', 'PRON', 'DET']

    def __init__(self, inflections: Dict[str, Dict[str, List[str]]]):
        self.inflections = inflections

    def random_inflection(self, lemma: str, tag: str, excluded_word: str) -> Optional[str]:
        tag_inflections = self.inflections.get(tag, {})
        candidate_morphs = [morph for morph in tag_inflections.get(lemma, []) if morph != excluded_word]
        if candidate_morphs:
            return choice(candidate_morphs)
        return None

    def save_to_json(self, path: Path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as file:
            json.dump(self.inflections, file)

    @property
    def tags(self) -> List[str]:
        return list(self.inflections.keys())

    @classmethod
    def load_from_json(cls, path: Path):
        with open(path, 'r') as file:
            inflections = json.load(file)
        return cls(inflections)

    @classmethod
    def create_from_sentences(cls, sentences: Iterable[Span]):
        inflections = {}
        for sentence in tqdm(sentences):
            for token in sentence:
                if token.lemma_ != token.text:
                    tag_inflections = inflections.get(token.tag_, {})
                    inflection = tag_inflections.get(token.lemma_, set())
                    inflection.add(token.text)
                    inflection.add(token.lemma_)
                    tag_inflections[token.lemma_] = inflection
                    inflections[token.tag_] = tag_inflections
        print()
        for tag_inflections in inflections.values():
            for lemma in tag_inflections:
                tag_inflections[lemma] = list(tag_inflections[lemma])
        return cls(inflections)
