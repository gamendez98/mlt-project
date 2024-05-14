from typing import Union

import spacy
from fastapi import FastAPI

app = FastAPI()

nlp = spacy.load("es_dep_news_trf")


@app.get("/")
def ok():
    return {"status": "OK"}


@app.post("/correct-sentence")
def get_corrections(sentence: str):
    # TODO: add actual correction logic
    return {"sentence": "Esta es la oracion despues de ser corregida"}


@app.post("/error-corrections")
def get_corrections(sentence: str):
    # TODO: add actual correction logic
    doc = nlp(sentence)
    dummy_corrections = ['$APPEND_palabra', '$DELETE', '$SPLIT_2', '$REPLACE_palabra'] + (['$KEEP'] * 100)
    return {"words": ['@@PADDING@@'] + [token.text for token in doc], 'labels': dummy_corrections[:len(doc)]}
