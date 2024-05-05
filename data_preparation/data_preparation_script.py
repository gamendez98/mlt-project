import os.path
from pathlib import Path
from typing import Iterable

import spacy
import yaml
from spacy import Language
from spacy.tokens import Span

from data_preparation.inflector import Inflector
from data_preparation.sentence_modifier import SentenceModifier
from data_preparation.stop_word_dictionary import StopWordDictionary


def doc_generator(raw_dataset_path: Path, nlp: Language) -> Iterable[Span]:
    # TODO: read from raw dataset
    for doc in nlp.pipe(["ejemplo uno", "ejemplo 2"]):
        for sentence in doc.sents:
            yield sentence


def create_or_load_inflector(inflector_path: Path, raw_dataset_path: Path, nlp_model: str) -> Inflector:
    if os.path.exists(inflector_path):
        return Inflector.load_from_json(inflector_path)
    nlp = spacy.load(nlp_model)
    inflector = Inflector.create_from_docs(doc_generator(raw_dataset_path, nlp))
    inflector.save_to_json(inflector_path)
    return inflector


def create_or_load_stop_word_dictionary(stop_word_dictionary_path: Path, raw_dataset_path: Path,
                                        nlp_model: str) -> StopWordDictionary:
    if os.path.exists(stop_word_dictionary_path):
        return StopWordDictionary.load_from_json(stop_word_dictionary_path)
    nlp = spacy.load(nlp_model)
    stop_word_dictionary = StopWordDictionary.create_from_docs(doc_generator(raw_dataset_path, nlp))
    stop_word_dictionary.save_to_json(stop_word_dictionary_path)
    return stop_word_dictionary


def create_or_load_sentence_modifier(inflector_path: Path, stop_word_dictionary_path: Path, raw_dataset_path: Path,
                                     nlp_model: str):
    inflector = create_or_load_inflector(inflector_path, raw_dataset_path, nlp_model)
    stop_word_dictionary = create_or_load_stop_word_dictionary(stop_word_dictionary_path, raw_dataset_path, nlp_model)
    return SentenceModifier(nlp_model, inflector, stop_word_dictionary)


def create_synthetic_dataset(configuration_path='data_preparation/data_preparation_config.yaml'):
    with open(configuration_path, 'r') as file:
        configuration = yaml.load(file, Loader=yaml.FullLoader)
    sentence_modifier = create_or_load_sentence_modifier(
        inflector_path=configuration['inflector_path'],
        stop_word_dictionary_path=configuration['stop_word_dictionary_path'],
        raw_dataset_path=configuration['raw_dataset_path'],
        nlp_model=configuration['nlp_model']
    )

