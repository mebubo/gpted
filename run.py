#%%
from completions import *
from expand_llm import *
from expand import *

# %%
model, tokenizer, device = load_model()

#%%
# input_text = "The quick brown fox jumpz over"
# input_text = "He asked me to prostate myself before the king"
input_text = "Здравствуйте, я хочу предвыполнить заказ"
inputs: BatchEncoding = tokenize(input_text, tokenizer, device)

#%%
token_probs: list[tuple[int, float]] = calculate_log_probabilities(model, tokenizer, inputs)

#%%
words = split_into_words(token_probs, tokenizer)
log_prob_threshold = -5.0
low_prob_words = [(i, word) for i, word in enumerate(words) if word.logprob < log_prob_threshold]

#%%
contexts = [word.context for _, word in low_prob_words]

#%%
expander = ExpanderOneBatchLLM(model, tokenizer)

#%%
series = []
for i, x in enumerate(contexts):
    series.append(Series(id=i, tokens=x, budget=5.0))

#%%
batch = Batch(items=series)

#%%
stopping_criterion = create_stopping_criterion_llm(tokenizer)

#%%
expanded = expand(batch, expander, stopping_criterion)

# %%
def print_expansions(expansions: ExpansionResultBatch):
    for result in expansions.items:
        for expansion in result.expansions:
            # convert tokens to string
            tokens = [e.token for e in expansion]
            s = tokenizer.decode(tokens)
            print(f"{result.series.id}: {expansion} {s}")

print_expansions(expanded)
# %%
