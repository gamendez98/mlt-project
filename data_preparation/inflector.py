import json
import os
from pathlib import Path
from typing import Dict, List, Iterable, Optional
from random import choice

from spacy.tokens import Span


class Inflector:
    MORPHING_TAGS = ['NOUN', 'VERB']

    def __init__(self, inflections: Dict[str: Dict[str: List[str]]]):
        self.inflections = inflections

    def random_inflection(self, lemma: str, excluded_word: str) -> Optional[str]:
        candidate_morphs = [morph for morph in self.inflections.get(lemma, []) if morph != excluded_word]
        if candidate_morphs:
            return choice(candidate_morphs)
        return None

    def save_to_json(self, path: Path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as file:
            json.dump(self.inflections, file)

    @classmethod
    def load_from_json(cls, path: Path):
        with open(path, 'r') as file:
            inflections = json.load(file)
        return cls(inflections)

    @classmethod
    def create_from_docs(cls, docs: Iterable[Span]):
        inflections = {tag: {} for tag in cls.MORPHING_TAGS}
        for doc in docs:
            for token in doc:
                if token.tag in cls.MORPHING_TAGS:
                    inflection = inflections[token.tag].get(token.lemma_, [])
                    inflection.append(token.text)
                    inflections[token.tag][token.lemma_] = inflection
        return cls(inflections)
