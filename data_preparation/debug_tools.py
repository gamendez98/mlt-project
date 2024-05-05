import pandas as pd


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
