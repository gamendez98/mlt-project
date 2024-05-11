from typing import List


PADDING = '@@PADDING@@'
DELETE = '$DELETE'
KEEP = '$KEEP'
APPEND = '$APPEND_'
SPLIT = '$SPLIT_'
REPLACE = '$REPLACE_'


class GrammarErrorCorrector:

    @staticmethod
    def correct_all_errors(words: List[str], labels: List[str]):
        pass

    @staticmethod
    def append(words: List[str], label_position: int, word: str):
        new_words = list(words)
        new_words.insert(label_position + 1, word)
        return new_words

    @staticmethod
    def delete(words: List[str], label_position: int):
        new_words = list(words)
        new_words.pop(label_position)
        return new_words

    @staticmethod
    def split(words: List[str], label_position: int, split_position: int):
        pass

    @staticmethod
    def replace(words: List[str], label_position: int, word: str):
        pass
