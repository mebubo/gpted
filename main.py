from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from functools import lru_cache

from models import CheckResponse
from completions import check_text, load_model

app = FastAPI()

model, tokenizer, device = load_model()

@lru_cache(maxsize=100)
def cached_check_text(text: str):
    return check_text(text, model, tokenizer, device)

@app.get("/check", response_model=CheckResponse)
def check(text: str):
    return CheckResponse(text=text, words=cached_check_text(text))

app.mount("/", StaticFiles(directory="frontend/public", html=True))
