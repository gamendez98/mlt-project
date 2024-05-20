import json
import os.path
from collections import Counter
from pathlib import Path
from typing import Iterable, List

import datasets
import pandas as pd
import spacy
import yaml
from spacy import Language
from spacy.tokens import Span
from tqdm import tqdm

from data_preparation.inflector import Inflector
from data_preparation.sentence_modifier import SentenceModifier
from data_preparation.stop_word_dictionary import StopWordDictionary
from grammar_error_correction.grammar_error_correction import GrammarErrorCorrector


def sentence_generator(raw_dataset_path: str, nlp: Language, input_entry_limit: int) -> Iterable[Span]:
    texts = datasets.load_dataset(raw_dataset_path)['train'][:input_entry_limit]['text']
    for doc in nlp.pipe(texts, batch_size=32, n_process=-1):
        for sentence in doc.sents:
            yield sentence


def create_or_load_inflector(inflector_path: Path, raw_dataset_path: str, nlp_model: Language,
                             input_entry_limit: int) -> Inflector:
    if os.path.exists(inflector_path):
        return Inflector.load_from_json(inflector_path)
    print("Creating inflector from sentences...")
    inflector = Inflector.create_from_sentences(sentence_generator(raw_dataset_path, nlp_model, input_entry_limit))
    inflector.save_to_json(inflector_path)
    return inflector


def create_or_load_stop_word_dictionary(stop_word_dictionary_path: Path, raw_dataset_path: str,
                                        nlp_model: Language, input_entry_limit: int) -> StopWordDictionary:
    if os.path.exists(stop_word_dictionary_path):
        return StopWordDictionary.load_from_json(stop_word_dictionary_path)
    print("Creating stop word dictionary from sentences...")
    stop_word_dictionary = StopWordDictionary.create_from_sentences(
        sentence_generator(raw_dataset_path, nlp_model, input_entry_limit))
    stop_word_dictionary.save_to_json(stop_word_dictionary_path)
    return stop_word_dictionary


def create_or_load_sentence_modifier(inflector_path: Path, stop_word_dictionary_path: Path, raw_dataset_path: str,
                                     nlp_model: Language, transformation_rate: float, transformations: List[str],
                                     input_entry_limit: int):
    inflector = create_or_load_inflector(inflector_path, raw_dataset_path, nlp_model, input_entry_limit)
    stop_word_dictionary = create_or_load_stop_word_dictionary(stop_word_dictionary_path, raw_dataset_path, nlp_model,
                                                               input_entry_limit)
    return SentenceModifier(nlp_model, inflector, stop_word_dictionary, transformation_rate, transformations)


def truncate_label_vocab(dataset: List[dict], label_vocab_size: int) -> List[dict]:
    label_counts = Counter(label for entry in dataset for label in entry['labels'])
    label_counts = pd.DataFrame({
        'label': list(label_counts.keys()),
        'label_count': list(label_counts.values())
    })
    label_counts.sort_values('label_count', ascending=False, inplace=True)
    labels_to_keep = set(label_counts.head(label_vocab_size).label)
    print(f'Keeping the {len(labels_to_keep)} most common labels')
    percentage_kept = label_counts.head(label_vocab_size).label_count.sum() / label_counts.label_count.sum()
    print(f'{percentage_kept:.1%} of labels remain unaffected')
    gec = GrammarErrorCorrector(None)
    for entry in dataset:
        words, labels = entry['modified_words'], entry['labels']
        words, labels = gec.correct_label_errors(words, labels, labels_to_ignore=labels_to_keep)
        entry['modified_words'] = words
        entry['labels'] = labels
    return dataset


def create_synthetic_dataset(configuration_path='data_preparation/data_preparation_config.yaml'):
    with open(configuration_path, 'r') as file:
        configuration = yaml.load(file, Loader=yaml.FullLoader)
    raw_dataset_path = configuration['raw_dataset_path']
    nlp_model = spacy.load(configuration['nlp_model'])
    input_entry_limit = configuration.get('input_entry_limit')
    sentence_modifier = create_or_load_sentence_modifier(
        inflector_path=configuration['inflector_path'],
        stop_word_dictionary_path=configuration['stop_word_dictionary_path'],
        raw_dataset_path=raw_dataset_path,
        nlp_model=nlp_model,
        transformation_rate=configuration['transformation_rate'],
        transformations=configuration.get('transformations'),
        input_entry_limit=input_entry_limit
    )
    dataset = []
    print('Generating synthetic dataset')
    for sentence in tqdm(sentence_generator(configuration['raw_dataset_path'], nlp_model, input_entry_limit)):
        original_words = sentence_modifier.sentence_words(sentence)
        words, tokens, labels = sentence_modifier.randomly_transform(sentence)
        dataset.append({
            'original_words': original_words,
            'modified_words': words,
            'labels': labels
        })
    print()
    dataset = truncate_label_vocab(dataset, configuration.get('label_vocab_size'))
    target_path = configuration['target_path']
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, 'w') as file:
        json.dump(dataset, file)


if __name__ == '__main__':
    create_synthetic_dataset()
