from typing import List, Set, Optional

import spacy
from spacy import Language
from spacy.tokens import Token, Span
from random import choice

from data_preparation.inflector import Inflector
from data_preparation.stop_word_dictionary import StopWordDictionary

# %%

TRANSFORMATION_METHODS = []


class OutOfCandidatesException(Exception):
    pass


def register_as_transformation(method):
    TRANSFORMATION_METHODS.append(method.__name__)
    return method


def raise_out_of_candidates(method):
    def wrapper(*args, **kwargs):
        candidates = method(*args, **kwargs)
        if not candidates:
            raise OutOfCandidatesException(f'No candidates for case {method.__name__}')
        return candidates

    return wrapper


# %%

PADDING = '@@PADDING@@'
DELETE = '$DELETE'
KEEP = '$KEEP'
APPEND = '$APPEND_'
SPLIT = '$SPLIT_'
REPLACE = 'REPLACE_'


# %%


class SentenceModifier:
    SEMANTIC_TAGS = ['NOUN', 'VERB', 'ADJ', 'ADV']

    def __init__(self, nlp_model: Language, inflector: Inflector, stop_word_dictionary: StopWordDictionary,
                 transformation_rate: float = 0.15, transformations: Optional[List] = None):
        self.nlp = nlp_model
        self.inflector = inflector
        self.stop_word_dictionary = stop_word_dictionary
        self.transformation_rate = transformation_rate
        self.transformations = transformations or TRANSFORMATION_METHODS

    def randomly_transform(self, sentence: Span) -> (List[str], List[Optional[Token]], List[str]):
        words = [token.text for token in sentence]
        tokens = list(sentence)
        labels = [KEEP] * len(sentence)
        number_of_transformations = int(len(sentence) * self.transformation_rate)
        candidate_transformations = list(self.transformations)
        for _ in range(number_of_transformations):
            transformation = choice(candidate_transformations)
            try:
                words, tokens, labels = self.__getattribute__(transformation)(words, tokens, labels)
            except OutOfCandidatesException:
                candidate_transformations.remove(transformation)
                if not candidate_transformations:
                    break
        return words, tokens, labels

    @register_as_transformation
    def transform_morphology(self, words: List[str], tokens: List[Optional[Token]], labels: List[str]):
        candidate_positions = list(self.candidates_for_morphology(words, tokens, labels))
        position = choice(candidate_positions)
        token = tokens[position]
        new_morph = self.inflector.random_inflection(token.lemma_, token.tag_, words[position])
        if new_morph:
            labels[position] = f'{REPLACE}{words[position]}'
            words[position] = new_morph
            tokens[position] = None
        return words, tokens, labels

    @register_as_transformation
    def transform_within_poss(self, words: List[str], tokens: List[Optional[Token]], labels: List[str]):
        position = choice(list(self.candidates_for_within_poss(words, tokens, labels)))
        token = tokens[position]
        new_stop_word = self.stop_word_dictionary.random_stop_word(token.pos_, words[position])
        if new_stop_word:
            labels[position] = f'{REPLACE}{words[position]}'
            words[position] = new_stop_word
            tokens[position] = None
        return words, tokens, labels

    @register_as_transformation
    def transform_elimination(self, words: List[str], tokens: List[Optional[Token]], labels: List[str]):
        position = choice(list(self.candidates_for_elimination(words, tokens, labels)))
        labels[position - 1] = f'{APPEND}{words[position]}'
        labels.pop(position)
        words.pop(position)
        tokens.pop(position)
        return words, tokens, labels

    @register_as_transformation
    def transform_adding(self, words: List[str], tokens: List[Optional[Token]], labels: List[str]):
        position = choice(list(self.un_affected_positions(labels)))
        added_word = choice(self.stop_word_dictionary.STOP_WORDS)
        split_point = position + 1
        words = words[:split_point] + [added_word] + words[split_point:]
        tokens = tokens[:split_point] + [None] + tokens[split_point:]
        labels = labels[:split_point] + [DELETE] + labels[split_point:]
        return words, tokens, labels

    @register_as_transformation
    def transform_token_fusion(self, words: List[str], tokens: List[Optional[Token]], labels: List[str]):
        position = choice(list(self.candidates_for_token_fusion(words, tokens, labels)))
        left_word = words[position]
        words = words[:position] + [words[position] + words[position + 1]] + words[position + 2:]
        labels = labels[:position] + [f'{SPLIT}{len(left_word)}'] + labels[position + 2:]
        tokens = tokens[:position] + [None] + tokens[position + 2:]
        return words, tokens, labels

    @raise_out_of_candidates
    def candidates_for_morphology(self, words: List[str], tokens: List[Optional[Token]], labels: List[str]) -> Set[int]:
        morphable_positions = {i for i, token in enumerate(tokens) if
                               token and token.tag_ in self.inflector.MORPHING_TAGS}
        return self.un_affected_positions(labels) & morphable_positions

    @raise_out_of_candidates
    def candidates_for_within_poss(self, words: List[str], tokens: List[Optional[Token]],
                                   labels: List[str]) -> Set[int]:
        stop_word_positions = {i for i, word in enumerate(words) if word in self.stop_word_dictionary.STOP_WORDS}
        return self.un_affected_positions(labels) & stop_word_positions

    @raise_out_of_candidates
    def candidates_for_elimination(self, words: List[str], tokens: List[Optional[Token]],
                                   labels: List[str]) -> Set[int]:
        non_semantic_positions = {i for i, token in enumerate(tokens) if token and token.tag_ not in self.SEMANTIC_TAGS}
        return self.un_affected_positions(labels) & non_semantic_positions

    @raise_out_of_candidates
    def candidates_for_token_fusion(self, words: List[str], tokens: List[Optional[Token]],
                                    labels: List[str]) -> Set[int]:
        return self.un_affected_positions(labels) - {len(words) - 1}

    @staticmethod
    @raise_out_of_candidates
    def un_affected_positions(labels: List[str]) -> Set[int]:
        affected_positions = {ii for i, label in enumerate(labels) if label != KEEP for ii in
                              [i - 1, i, i + 1]}
        positions = {i for i in range(1, len(labels))}  # the position 0 belongs to padding
        positions -= affected_positions
        return positions
