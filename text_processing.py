from dataclasses import dataclass

@dataclass
class Word:
    tokens: list[str]
    text: str
    logprob: float
    first_token_index: int

def split_into_words(token_probs: list[tuple[str, float]]) -> list[Word]:
    words = []
    current_word = []
    current_log_probs = []
    current_word_first_token_index = 0

    for i, (token, logprob) in enumerate(token_probs):
        if not token.startswith(chr(9601)) and token.isalpha():
            current_word.append(token)
            current_log_probs.append(logprob)
        else:
            if current_word:
                words.append(Word(current_word, "".join(current_word), sum(current_log_probs), current_word_first_token_index))
            current_word = [token]
            current_log_probs = [logprob]
            current_word_first_token_index = i

    if current_word:
        words.append(Word(current_word, "".join(current_word), sum(current_log_probs), current_word_first_token_index))

    return words
