from typing import List

from abc import ABC, abstractmethod
from typing import List

PADDING = '@@PADDING@@'


class TokenWordAlignment(ABC):
    @abstractmethod
    def clean_token(self, token: str) -> str:
        pass

    @abstractmethod
    def is_word_start(self, token: str) -> bool:
        pass

    def token_to_word_realignment(self, ner_tagged_sentence: List[dict]) -> (List[str], List[str]):
        realigned_words = []
        realigned_labels = []
        accumulated_word = ''
        for tagged_token in ner_tagged_sentence[::-1]:
            token = tagged_token['word']
            label = tagged_token['entity']
            clean_word = self.clean_token(token)
            accumulated_word = clean_word + accumulated_word
            if self.is_word_start(token):
                realigned_words.insert(0, accumulated_word)
                realigned_labels.insert(0, label)
                accumulated_word = ''
        return (
            realigned_words,
            realigned_labels,
        )


class BertTokenWordAlignment(TokenWordAlignment):
    BERT_WORD_PIECE_START = '##'

    def clean_token(self, token: str) -> str:
        return token[2:] if token.startswith(self.BERT_WORD_PIECE_START) else token

    def is_word_start(self, token: str) -> bool:
        return not token.startswith(self.BERT_WORD_PIECE_START)


class RobertaTokenWordAlignment(TokenWordAlignment):
    ROBERTA_WORD_START = '▁'

    def clean_token(self, token: str) -> str:
        return token[1:] if token.startswith(self.ROBERTA_WORD_START) else token

    def is_word_start(self, token: str) -> bool:
        return token.startswith(self.ROBERTA_WORD_START) or token == PADDING
