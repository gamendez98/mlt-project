# Mlt-project

## Installation

This project was tested using `python3.10`. To test this project create a virtual environment and install the
requirements described in `requirements.txt` using `pip install -r requirements.txt` or
`conda install --yes --file requirements.txt`.

After this you can install the spacy model to be used for the synthetic data generation using
`spacy download <MODEL_NAME>` we recomend to use `es_dep_news_trf` for a spanish dataset

## Dataset creation

To create the synthetic dataset run `python data_preparation/synthetic_dataset_creation.py`. This script will also create
an `Inflector` and a `StopWordDictionary` these objects will be saved as json files.

This script will read the configuration file `data_preparation/synthetic_dataset_config.yaml`. This file describes the
parameters:

- `inflector_path`: Path to save and load the `Inflector` from
- `stop_word_dictionary_path`: Path to save and load the `StopWordDictionary` from
- `raw_dataset_path`: Path to the raw texts
- `label_vocab_size`: maximum number of labels to get on the output dataset
- `input_entry_limit`: number of entries to take from the raw dataset
- `nlp_model`: spacy model name to be used for the data generation
- `transformation_rate`: Fraction of tokens that the script will try to transform and label. The actual proportion of
  transformed tokens may be lower since some times transformations are not possible
- `target_path`: Path to save the resulting synthetic dataset

### Raw data

The raw data-file must be a csv file containing a columns titled `text` that contains all the examples to create the
synthetic dataset

### Synthetic data

The result of running the synthetic dataset script will be a dataset with the structure

```json
[
  {
    "original_words": ["@@PADDING@@", "El", "ama", "los", "gatos", "."],
    "modified_words": ["@@PADDING@@", "El", "ama", "gatos", "."],
    "labels": ["$KEEP", "$KEEP", "$APPEND_los", "$KEEP", "$KEEP"]
  },
  
  {}
  
]
```

Where `original_words` has the sequence of tokens as they came from the raw data, `modified_words` are the tokens after
injecting errors into the sentence and `labels` corresponds to the necessary corrections to go back to the original.

These corrections can be:
- `$KEEP`: No change is needed in this position
- `$DELETE`: The word corresponding to this position has to be removed
- `$APPEND_<word>`: append the token `<word>` after the position of this label 
- `$SPLIT_<place>`: split the word at this position at the character `<place>`
- `$REPLACE_<word>`: replace the word at this position for `<word>`


## Grammar error correction

This repository also provides an easy way to make the corrections described in the synthetic dataset like so:
```python
from grammar_error_correction.grammar_error_correction import correct_all_errors

words= ["@@PADDING@@", "El", "ama", "gatos", "."]
labels= ["$KEEP", "$KEEP", "$APPEND_los", "$KEEP", "$KEEP"]

corrected_words = correct_all_errors(words, labels)
```

This functionality can be used to check the correctness of the dataset

```python
from data_preparation.debug_tools import check_synthetic_dataset_correctness

check_synthetic_dataset_correctness('path/to/syn_data.json')
```

## Model Training

The notebooks and scripts for fine-tuning the transformers models are in the `modelling` folder. First, you have to
generate the training, evaluation and test datasets with `organize_training_data.ipynb`, assuming the synthetic dataset
is already created. Two models are fine-tuned:

1. BETO (Spanish BERT) is finetuned on `bert_finetune.ipynb`.
2. A Spanish RoBERTa is finetuned via two possible ways: a notebook, `roberta_finetune.ipynb`, or its script equivalent, `roberta_finetune_script.py`. Since RoBERTa is way bigger than BETO, we recommend the script approach, so it can train on the background. It may be necessary to uncomment the `model_id` line that refers to `"MMG/xlm-roberta-large-ner-spanish"`, which is the base Hugging Face model. The script can be run with `python roberta_finetune_script.py`.

The evaluation over the test dataset is carried on in `testing_model.ipynb`.

## App

To deploy the app run `python application/app.py`. A notebook version of the app is also available on the same folder.
