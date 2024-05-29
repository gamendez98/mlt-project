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


class Configuration:
    INSTANCE = None

    def __init__(self, configuration_path='data_preparation/data_preparation_config.yaml'):
        with open(configuration_path, 'r') as file:
            self.__dict__ = yaml.load(file, Loader=yaml.FullLoader)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    @classmethod
    def get_instance(cls):
        if not cls.INSTANCE:
            cls.INSTANCE = Configuration()
        return cls.INSTANCE


def sentence_generator(nlp: Language, raw_dataset) -> Iterable[Span]:
    i = 0
    input_entry_limit = Configuration.get_instance().input_entry_limit
    batch_size = min(Configuration.get_instance().spacy_batch_size, input_entry_limit)
    for i in range(0, input_entry_limit, batch_size):
        text_chunk = raw_dataset[i:i + batch_size]['text']
        for doc in nlp.pipe(text_chunk, batch_size=32, n_process=-1):
            for sentence in doc.sents:
                yield sentence
    for doc in nlp.pipe(raw_dataset[i + batch_size:input_entry_limit]['text'], batch_size=32, n_process=-1):
        for sentence in doc.sents:
            yield sentence


def create_or_load_inflector(inflector_path: Path, nlp_model: Language,
                             raw_dataset) -> Inflector:
    if os.path.exists(inflector_path):
        return Inflector.load_from_json(inflector_path)
    print("Creating inflector from sentences...")
    inflector = Inflector.create_from_sentences(sentence_generator(nlp_model, raw_dataset))
    inflector.save_to_json(inflector_path)
    return inflector


def create_or_load_stop_word_dictionary(stop_word_dictionary_path: Path,
                                        nlp_model: Language, raw_dataset: List[str]) -> StopWordDictionary:
    if os.path.exists(stop_word_dictionary_path):
        return StopWordDictionary.load_from_json(stop_word_dictionary_path)
    print("Creating stop word dictionary from sentences...")
    stop_word_dictionary = StopWordDictionary.create_from_sentences(
        sentence_generator(nlp_model, raw_dataset))
    stop_word_dictionary.save_to_json(stop_word_dictionary_path)
    return stop_word_dictionary


def create_or_load_sentence_modifier(inflector_path: Path, stop_word_dictionary_path: Path,
                                     nlp_model: Language, transformation_rate: float, transformations: List[str],
                                     raw_dataset: List[str]):
    inflector = create_or_load_inflector(inflector_path, nlp_model, raw_dataset)
    stop_word_dictionary = create_or_load_stop_word_dictionary(stop_word_dictionary_path, nlp_model, raw_dataset)
    return SentenceModifier(nlp_model, inflector, stop_word_dictionary, transformation_rate, transformations)


def truncate_label_vocab(dataset_directory: Path, label_vocab_size: int):
    label_counts = Counter()
    dataset_files = list(Path(dataset_directory).rglob('*.json'))
    for dataset_file in dataset_files:
        chunk_data = json.load(open(dataset_file, 'r'))
        chunk_label_counts = Counter(label for entry in chunk_data for label in entry['labels'])
        label_counts.update(chunk_label_counts)
    label_counts = pd.DataFrame({
        'label': list(label_counts.keys()),
        'label_count': list(label_counts.values())
    })
    label_counts.sort_values('label_count', ascending=False, inplace=True)
    labels_to_keep = set(label_counts.head(label_vocab_size).label)
    print(f'Keeping the {len(labels_to_keep)} most common labels')
    percentage_kept = label_counts.head(label_vocab_size).label_count.sum() / label_counts.label_count.sum()
    print(f'{percentage_kept:.1%} of labels remain unaffected')
    gec = GrammarErrorCorrector(None, None)
    for dataset_file in dataset_files:
        chunk_data = json.load(open(dataset_file, 'r'))
        for entry in chunk_data:
            words, labels = entry['modified_words'], entry['labels']
            words, labels = gec.correct_label_errors(words, labels, labels_to_ignore=labels_to_keep)
            entry['modified_words'] = words
            entry['labels'] = labels
        with open(dataset_file, 'w') as file:
            json.dump(chunk_data, file)


def create_synthetic_dataset():
    configuration = Configuration.get_instance()
    raw_dataset_path = configuration.raw_dataset_path
    nlp_model = spacy.load(configuration.nlp_model)
    train_texts = datasets.load_dataset(raw_dataset_path)['train']
    sentence_modifier = create_or_load_sentence_modifier(
        inflector_path=configuration.inflector_path,
        stop_word_dictionary_path=configuration.stop_word_dictionary_path,
        nlp_model=nlp_model,
        transformation_rate=configuration.transformation_rate,
        transformations=configuration.get('transformations'),
        raw_dataset=train_texts
    )
    print('Generating synthetic dataset')
    chunk_size = configuration.data_chunk_size
    os.makedirs(configuration.target_path, exist_ok=True)
    dataset = []
    chunk_number = 0
    target_path = configuration.target_path
    for sentence in tqdm(sentence_generator(nlp_model, train_texts)):
        original_words = sentence_modifier.sentence_words(sentence)
        words, tokens, labels = sentence_modifier.randomly_transform(sentence)
        dataset.append({
            'original_words': original_words,
            'modified_words': words,
            'labels': labels
        })
        # dataset = truncate_label_vocab(dataset, configuration.get('label_vocab_size'))
        if dataset and len(dataset) % chunk_size==0:
            file_name = f'chunk_{chunk_number}.json'
            path = os.path.join(target_path, file_name)
            with open(path, 'w') as file:
                json.dump(dataset, file)
            chunk_number += 1
            dataset = []
    if dataset:
        file_name = f'chunk_{chunk_number}.json'
        path = os.path.join(target_path, file_name)
        with open(path, 'w') as file:
            json.dump(dataset, file)
    truncate_label_vocab(target_path, configuration.label_vocab_size)


if __name__ == '__main__':
    create_synthetic_dataset()
