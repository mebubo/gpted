from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from models import CheckResponse, ApiWord
from completions import check_text, load_model

app = FastAPI()

model, tokenizer, device = load_model()

@app.get("/check", response_model=CheckResponse)
def check(text: str):
    return CheckResponse(text=text, words=check_text(text, model, tokenizer, device))

# serve files from frontend/public
app.mount("/", StaticFiles(directory="frontend/public", html=True))
