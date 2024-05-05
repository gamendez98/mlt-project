from typing import List, Set, Optional

import spacy
from spacy.tokens import Doc, Token
from random import choice
from nltk.corpus import stopwords

from data_preparation.inflector import Inflector
from data_preparation.stop_word_dictionary import StopWordDictionary

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

    def __init__(self, nlp_model: str, inflector: Inflector, stop_word_dictionary: StopWordDictionary):
        self.nlp = spacy.load(nlp_model)
        self.inflector = inflector
        self.stop_word_dictionary = stop_word_dictionary

    @staticmethod
    def un_affected_positions(labels: List[str]) -> Set[int]:
        affected_positions = {ii for i, label in enumerate(labels) if label != '$KEEP' for ii in
                              [i - 1, i, i + 1]}
        positions = {i for i in range(1, len(labels))}  # the position 0 belongs to padding
        return positions - affected_positions

    def candidates_for_morphology(self, words: List[str], tokens: List[Optional[Token]], labels: List[str]) -> Set[int]:
        morphable_positions = {i for i, token in enumerate(tokens) if
                               token and token.tag_ in self.inflector.MORPHING_TAGS}
        return self.un_affected_positions(labels) & morphable_positions

    def candidates_for_within_poss(self, words: List[str], tokens: List[Optional[Token]],
                                   labels: List[str]) -> Set[int]:
        stop_word_positions = {i for i, word in enumerate(words) if word in self.stop_word_dictionary.STOP_WORDS}
        return self.un_affected_positions(labels) & stop_word_positions

    def candidates_for_elimination(self, words: List[str], tokens: List[Optional[Token]],
                                   labels: List[str]) -> Set[int]:
        non_semantic_positions = {i for i, token in enumerate(tokens) if token and token.tag_ not in self.SEMANTIC_TAGS}
        return self.un_affected_positions(labels) & non_semantic_positions

    def candidates_for_token_fusion(self, words: List[str], tokens: List[Optional[Token]],
                                    labels: List[str]) -> Set[int]:
        return self.un_affected_positions(labels) - {len(words) - 1}

    def transform_morphology(self, words: List[str], tokens: List[Optional[Token]], labels: List[str]):
        position = choice(list(self.candidates_for_morphology(words, tokens, labels)))
        token = tokens[position]
        new_morph = self.inflector.random_inflection(token.lemma_)
        if new_morph:
            labels[position] = f'{REPLACE}{words[position]}'
            words[position] = new_morph
            tokens[position] = None
        return words, tokens, labels

    def transform_within_poss(self, words: List[str], tokens: List[Optional[Token]], labels: List[str]):
        position = choice(list(self.candidates_for_within_poss(words, tokens, labels)))
        token = tokens[position]
        new_stop_word = self.stop_word_dictionary.random_stop_word(token.pos_)
        if new_stop_word:
            labels[position] = f'{REPLACE}{words[position]}'
            words[position] = new_stop_word
            tokens[position] = None
        return words, tokens, labels

    def transform_elimination(self, words: List[str], tokens: List[Optional[Token]], labels: List[str]):
        position = choice(list(self.candidates_for_elimination(words, tokens, labels)))
        labels[position - 1] = f'{APPEND}{words[position]}'
        labels.pop(position)
        words.pop(position)
        tokens.pop(position)
        return words, tokens, labels

    def transform_adding(self, words: List[str], tokens: List[Optional[Token]], labels: List[str]):
        position = choice(list(self.un_affected_positions(labels)))
        added_word = choice(self.stop_word_dictionary.STOP_WORDS)
        split_point = position + 1
        words = words[:split_point] + [added_word] + words[split_point:]
        tokens = tokens[:split_point] + [None] + tokens[split_point:]
        labels = words[:split_point] + [DELETE] + words[split_point:]
        return words, tokens, labels

    def transform_token_fusion(self, words: List[str], tokens: List[Optional[Token]], labels: List[str]):
        position = choice(list(self.candidates_for_token_fusion(words, tokens, labels)))
        left_word = words[position]
        words = words[:position] + [words[position] + words[position + 1]] + words[position + 2:]
        labels = labels[:position] + [f'{SPLIT}{len(left_word)}'] + labels[position + 2:]
        tokens = tokens[:position] + [None] + tokens[position + 2:]
        return words, tokens, labels
