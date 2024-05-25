import pandas as pd


def print_gec_ner_pipeline_results(ner_pipeline_results):
    print(pd.DataFrame(ner_pipeline_results).set_index('index')[['word', 'entity', 'score', 'start', 'end']])
