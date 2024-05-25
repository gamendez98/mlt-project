from typing import List, Set

from nltk import TreebankWordDetokenizer
from transformers import pipeline

from grammar_error_correction.token_realignment import TokenWordAlignment, PADDING

DELETE = '$DELETE'
KEEP = '$KEEP'
APPEND = '$APPEND_'
SPLIT = '$SPLIT_'
REPLACE = '$REPLACE_'


class GrammarErrorCorrector:

    def __init__(self, model_path: str, token_word_alignment: TokenWordAlignment):
        if model_path:
            self.tagging_pipeline = pipeline('ner', model=model_path)
        else:
            self.tagging_pipeline = None
        self.de_tokenizer = TreebankWordDetokenizer()
        self.token_word_alignment = token_word_alignment

    def correct_sentence(self, sentence: str) -> str:
        words, labels = self.get_word_labels(sentence)
        corrected_words, _ = self.correct_label_errors(words, labels)
        return self.de_tokenizer.detokenize(corrected_words[1:])

    def get_word_labels(self, sentence: str) -> (List[str], List[str]):
        ner_tagged_sentence = self.tagging_pipeline(f'{PADDING} {sentence}')
        return self.token_word_alignment.token_to_word_realignment(ner_tagged_sentence)

    def correct_label_errors(self, words: List[str], labels: List[str],
                             labels_to_ignore: Set[str] = None) -> (List[str], List[str]):
        label_position = 0
        labels_to_ignore = labels_to_ignore or {}
        while label_position < len(labels):
            label = labels[label_position]
            if label == KEEP or label in labels_to_ignore:
                pass
            elif label == DELETE:
                words, labels = self.correct_delete(words, labels, label_position)
                label_position -= 1
            elif label.startswith(APPEND):
                words, labels = self.correct_append(words, labels, label_position)
            elif label.startswith(SPLIT):
                words, labels = self.correct_split(words, labels, label_position)
            elif label.startswith(REPLACE):
                words, labels = self.correct_replace(words, labels, label_position)
            label_position += 1
        return words, labels

    @staticmethod
    def correct_append(words: List[str], labels: List[str], label_position: int) -> (List[str], List[str]):
        new_word = labels[label_position].replace(APPEND, '')
        modified_words = words[:label_position + 1] + [new_word] + words[label_position + 1:]
        modified_labels = labels[:label_position] + [KEEP, KEEP] + labels[label_position + 1:]
        modified_labels[label_position] = KEEP
        return modified_words, modified_labels

    @staticmethod
    def correct_delete(words: List[str], labels: List[str], label_position: int) -> (List[str], List[str]):
        modified_words = list(words)
        modified_labels = list(labels)
        modified_words.pop(label_position)
        modified_labels.pop(label_position)
        return modified_words, modified_labels

    @staticmethod
    def correct_split(words: List[str], labels: List[str], label_position: int) -> (List[str], List[str]):
        split_position = int(labels[label_position].replace(SPLIT, ''))
        split_word = [
            words[label_position][:split_position],
            words[label_position][split_position:]
        ]
        modified_words = words[:label_position] + split_word + words[label_position + 1:]
        modified_labels = labels[:label_position] + [KEEP, KEEP] + labels[label_position + 1:]
        return modified_words, modified_labels

    @staticmethod
    def correct_replace(words: List[str], labels: List[str], label_position: int) -> (List[str], List[str]):
        word_replacement = labels[label_position].replace(REPLACE, '')
        modified_words = words[:label_position] + [word_replacement] + words[label_position + 1:]
        modified_labels = labels[:label_position] + [KEEP] + labels[label_position + 1:]
        return modified_words, modified_labels
