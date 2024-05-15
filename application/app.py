from typing import Union
import sys

import spacy
from fastapi import FastAPI

from grammar_error_correction.grammar_error_correction import GrammarErrorCorrector

app = FastAPI()
MODEL_PATH = 'model/bert_ner_model'
gec = GrammarErrorCorrector(MODEL_PATH)


@app.get("/")
def ok():
    return {"status": "OK"}


@app.post("/correct-sentence")
def get_corrections(sentence: str):
    return {"sentence": gec.correct_sentence(sentence)}


@app.post("/error-corrections")
def get_corrections(sentence: str):
    words, labels = gec.get_word_labels(sentence)
    return {"words": words, 'labels': labels}
