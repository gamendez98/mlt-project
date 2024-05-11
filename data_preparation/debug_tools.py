import json

import pandas as pd
from datasets import tqdm

from grammar_error_correction.grammar_error_correction import correct_all_errors


def print_attributes(doc):
    df = pd.DataFrame({
        "text": [token.text for token in doc],
        "parent": [str(next(token.ancestors, None)) for token in doc],
        "lemma": [token.lemma_ for token in doc],
        "dep": [token.dep_ for token in doc],
        "tag": [token.tag_ for token in doc],
        "pos": [token.pos_ for token in doc],
    })
    print(df)


def print_synthetic_datum(datum):
    print(f'ORIGINAL: {datum["original_words"]}')
    df = pd.DataFrame({
        'modified_words': datum['modified_words'],
        'labels': datum['labels']
    })
    print(df)


def check_datum_correctness(datum):
    modified_words = datum['modified_words']
    original_words = datum['original_words']
    labels = datum['labels']
    corrected_words = correct_all_errors(modified_words, labels)
    is_correct = corrected_words == original_words
    if not is_correct:
        print(f"""
original_words = {original_words}
modified_words = {modified_words}
labels = {labels}
corrected_words = {corrected_words}
        """)
    return is_correct


def check_synthetic_dataset_correctness(synthetic_dataset_path):
    data = json.load(open(synthetic_dataset_path))
    for datum in tqdm(data):
        assert check_datum_correctness(datum)
