from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from functools import lru_cache

from models import ApiWord, CheckResponse
from completions import check_text, load_model

app = FastAPI()

model, tokenizer, device = load_model()

def check_text_stub(text: str):
    def rep(i):
        if i == 3:
            return -10, [" jumped", " leaps"]
        if i == 5:
            return -10, [" calm"]
        if i == 7:
            return -10, [" dog", " cat", " bird", " fish"]
        return -3, []

    result = []
    for i, w in enumerate(text.split()):
        logprob, replacements = rep(i)
        result.append(ApiWord(text=f" {w}", logprob=logprob, replacements=replacements))
    return result

@lru_cache(maxsize=100)
def cached_check_text(text: str):
    return check_text(text, model, tokenizer, device)

@app.get("/check", response_model=CheckResponse)
def check(text: str):
    return CheckResponse(text=text, words=cached_check_text(text))

app.mount("/", StaticFiles(directory="frontend/public", html=True))
