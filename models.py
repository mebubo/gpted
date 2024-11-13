from dataclasses import dataclass

from pydantic import BaseModel

@dataclass
class Word:
    tokens: list[int]
    text: str
    logprob: float
    context: list[int]

class Replacement(BaseModel):
    text: str
    logprob: float

class ApiWord(BaseModel):
    text: str
    logprob: float
    replacements: list[Replacement]

class CheckResponse(BaseModel):
    text: str
    words: list[ApiWord]
