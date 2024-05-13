from typing import Union

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def ok():
    return {"status": "OK"}


@app.get("/items/{item_id}")
def get_corrections(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}