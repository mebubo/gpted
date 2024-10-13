from dataclasses import dataclass

from pydantic import BaseModel

@dataclass
class Word:
    tokens: list[int]
    text: str
    logprob: float
    context: list[int]

class ApiWord(BaseModel):
    text: str
    logprob: float
    replacements: list[str]

class CheckResponse(BaseModel):
    text: str
    words: list[ApiWord]
