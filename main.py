from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

class Word(BaseModel):
    word: str
    start: int
    end: int
    logprob: float
    suggestions: list[str]

class CheckResponse(BaseModel):
    text: str
    words: list[Word]

app = FastAPI()

@app.get("/check", response_model=CheckResponse)
def check(text: str):
    return CheckResponse(text=text, words=[])

# serve files from frontend/public
app.mount("/", StaticFiles(directory="frontend/public", html=True))
