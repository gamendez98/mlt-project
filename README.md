# Mlt-project

## Installation

This project was tested using `python3.10`. To test this project create a virtual environment and install the
requirements described in `requirements.txt` using `pip install -r requirements.txt` or
`conda install --yes --file requirements.txt`.

After this you can install the spacy model to be used for the synthetic data generation using
`spacy download <MODEL_NAME>` we recomend to use `es_dep_news_trf` for a spanish dataset

## Dataset creation

To create the synthetic dataset run `python data_preparation/data_preparation_script.py`. This script will also create
an `Inflector` and a `StopWordDictionary` this objects info will also be saved each in their respective paths.

This script will read the configuration file `data_preparation/data_preparation_config.yaml`. This file describes the
parameters:

- `inflector_path`: Path to save and load the `Inflector` from
- `stop_word_dictionary_path`: Path to save and load the `StopWordDictionary` from
- `raw_dataset_path`: Path to the raw texts
- `nlp_model`: spacy model name to be used for the data generation
- `transformation_rate`: Fraction of tokens that the script will try to transform and label. The actual proportion of
  transformed tokens may be lower since some times transformations are not possible
- `target_path`: Path to save the resulting synthetic dataset

### Raw data

The raw data-file must be a csv file containing a columns titled `text` that contains all the examples to create the
synthetic dataset
