from typing import List

PADDING = '@@PADDING@@'
DELETE = '$DELETE'
KEEP = '$KEEP'
APPEND = '$APPEND_'
SPLIT = '$SPLIT_'
REPLACE = '$REPLACE_'


def correct_all_errors(words: List[str], labels: List[str]):
    label_position = 0
    while label_position < len(labels):
        label = labels[label_position]
        if label == KEEP:
            pass
        elif label == DELETE:
            words, labels = correct_delete(words, labels, label_position)
        elif label.startswith(APPEND):
            words, labels = correct_append(words, labels, label_position)
        elif label.startswith(SPLIT):
            words, labels = correct_split(words, labels, label_position)
        elif label.startswith(REPLACE):
            words, labels = correct_replace(words, labels, label_position)
        label_position += 1
    return words


def correct_append(words: List[str], labels: List[str], label_position: int) -> (List[str], List[str]):
    new_word = labels[label_position].replace(APPEND, '')
    modified_words = words[:label_position + 1] + [new_word] + words[label_position + 1:]
    modified_labels = labels[:label_position] + [KEEP, KEEP] + labels[label_position + 1:]
    modified_labels[label_position] = KEEP
    return modified_words, modified_labels


def correct_delete(words: List[str], labels: List[str], label_position: int) -> (List[str], List[str]):
    modified_words = list(words)
    modified_labels = list(labels)
    modified_words.pop(label_position)
    modified_labels.pop(label_position)
    return modified_words, modified_labels


def correct_split(words: List[str], labels: List[str], label_position: int) -> (List[str], List[str]):
    split_position = int(labels[label_position].replace(SPLIT, ''))
    split_word = [
        words[:split_position],
        words[split_position:]
    ]
    modified_words = words[:label_position] + split_word + words[label_position + 1:]
    modified_labels = labels[:label_position] + [KEEP, KEEP] + labels[label_position + 1:]
    return modified_words, modified_labels


def correct_replace(words: List[str], labels: List[str], label_position: int) -> (List[str], List[str]):
    word_replacement = labels[label_position].replace(REPLACE, '')
    modified_words = words[:label_position] + [word_replacement] + words[label_position + 1:]
    modified_labels = labels[:label_position] + [KEEP] + labels[label_position + 1:]
    return modified_words, modified_labels
