import json
import os.path
from pathlib import Path
from typing import Iterable

import pandas as pd
import spacy
import yaml
from spacy import Language
from spacy.tokens import Span
from tqdm import tqdm

from data_preparation.inflector import Inflector
from data_preparation.sentence_modifier import SentenceModifier
from data_preparation.stop_word_dictionary import StopWordDictionary


def sentence_generator(raw_dataset_path: Path, nlp: Language) -> Iterable[Span]:
    raw_data = pd.read_csv(raw_dataset_path)
    for doc in nlp.pipe(raw_data.text, batch_size=16):
        for sentence in doc.sents:
            yield sentence


def create_or_load_inflector(inflector_path: Path, raw_dataset_path: Path, nlp_model: Language) -> Inflector:
    if os.path.exists(inflector_path):
        return Inflector.load_from_json(inflector_path)
    print("Creating inflector from sentences...")
    inflector = Inflector.create_from_sentences(sentence_generator(raw_dataset_path, nlp_model))
    inflector.save_to_json(inflector_path)
    return inflector


def create_or_load_stop_word_dictionary(stop_word_dictionary_path: Path, raw_dataset_path: Path,
                                        nlp_model: Language) -> StopWordDictionary:
    if os.path.exists(stop_word_dictionary_path):
        return StopWordDictionary.load_from_json(stop_word_dictionary_path)
    print("Creating stop word dictionary from sentences...")
    stop_word_dictionary = StopWordDictionary.create_from_sentences(sentence_generator(raw_dataset_path, nlp_model))
    stop_word_dictionary.save_to_json(stop_word_dictionary_path)
    return stop_word_dictionary


def create_or_load_sentence_modifier(inflector_path: Path, stop_word_dictionary_path: Path, raw_dataset_path: Path,
                                     nlp_model: Language, transformation_rate: float):
    inflector = create_or_load_inflector(inflector_path, raw_dataset_path, nlp_model)
    stop_word_dictionary = create_or_load_stop_word_dictionary(stop_word_dictionary_path, raw_dataset_path, nlp_model)
    return SentenceModifier(nlp_model, inflector, stop_word_dictionary, transformation_rate)


def create_synthetic_dataset(configuration_path='data_preparation/data_preparation_config.yaml'):
    with open(configuration_path, 'r') as file:
        configuration = yaml.load(file, Loader=yaml.FullLoader)
    raw_dataset_path = configuration['raw_dataset_path']
    nlp_model = spacy.load(configuration['nlp_model'])
    sentence_modifier = create_or_load_sentence_modifier(
        inflector_path=configuration['inflector_path'],
        stop_word_dictionary_path=configuration['stop_word_dictionary_path'],
        raw_dataset_path=raw_dataset_path,
        nlp_model=nlp_model,
        transformation_rate=configuration['transformation_rate']
    )
    dataset = []
    print('Generating synthetic dataset')
    for sentence in tqdm(sentence_generator(configuration['raw_dataset_path'], nlp_model)):
        words, tokens, labels = sentence_modifier.randomly_transform(sentence)
        dataset.append({
            'words': words,
            'labels': labels
        })
    print()
    target_path = configuration['target_path']
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, 'w') as file:
        json.dump(dataset, file)


if __name__ == '__main__':
    create_synthetic_dataset()
