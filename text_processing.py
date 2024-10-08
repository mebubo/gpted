from dataclasses import dataclass
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

type Tokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast

@dataclass
class Word:
    tokens: list[int]
    text: str
    logprob: float
    context: list[int]

def split_into_words(token_probs: list[tuple[int, float]], tokenizer: Tokenizer) -> list[Word]:
    words: list[Word] = []
    current_word: list[int] = []
    current_log_probs: list[float] = []
    current_word_first_token_index: int = 0
    all_tokens: list[int] = [token_id for token_id, _ in token_probs]

    def append_current_word():
        if current_word:
            words.append(Word(current_word,
                              tokenizer.decode(current_word),
                              sum(current_log_probs),
                              all_tokens[:current_word_first_token_index]))

    for i, (token_id, logprob) in enumerate(token_probs):
        token: str = tokenizer.convert_ids_to_tokens([token_id])[0]
        if not token.startswith(chr(9601)) and token.isalpha():
            current_word.append(token_id)
            current_log_probs.append(logprob)
        else:
            append_current_word()
            current_word = [token_id]
            current_log_probs = [logprob]
            current_word_first_token_index = i

    append_current_word()

    return words
