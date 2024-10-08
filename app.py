#%%
import time
from text_processing import split_into_words, Word
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast, BatchEncoding
from typing import cast

type Tokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast

def load_model_and_tokenizer(model_name: str, device: torch.device) -> tuple[PreTrainedModel, Tokenizer]:
    tokenizer: Tokenizer = AutoTokenizer.from_pretrained(model_name)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    return model, tokenizer

def tokenize(input_text: str, tokenizer: Tokenizer, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inputs: BatchEncoding = tokenizer(input_text, return_tensors="pt").to(device)
    input_ids = cast(torch.Tensor, inputs["input_ids"])
    attention_mask = cast(torch.Tensor, inputs["attention_mask"])
    return input_ids, attention_mask

def calculate_log_probabilities(model: PreTrainedModel, tokenizer: Tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> list[tuple[int, float]]:
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    # B x T x V
    logits: torch.Tensor = outputs.logits[:, :-1, :]
    # B x T x V
    log_probs: torch.Tensor = torch.log_softmax(logits, dim=-1)
    # T - 1
    token_log_probs: torch.Tensor = log_probs[0, range(log_probs.shape[1]), input_ids[0][1:]]
    # T - 1
    tokens: torch.Tensor = input_ids[0][1:]
    return list(zip(tokens.tolist(), token_log_probs.tolist()))


def generate_replacements(model: PreTrainedModel, tokenizer: Tokenizer, contexts: list[list[int]], device: torch.device, num_samples: int = 5) -> list[list[str]]:
    input_ids = torch.tensor(contexts).to(device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.shape[-1] + 5,
            num_return_sequences=num_samples,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )
    all_new_words = []
    for i in range(len(contexts)):
        replacements = []
        for j in range(num_samples):
            generated_ids = outputs[i * num_samples + j][input_ids.shape[-1]:]
            new_word = tokenizer.decode(generated_ids, skip_special_tokens=True).split()[0]
            replacements.append(new_word)
        all_new_words.append(replacements)
    return all_new_words

#%%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "mistralai/Mistral-7B-v0.1"
model, tokenizer = load_model_and_tokenizer(model_name, device)

#%%

input_text = "He asked me to prostrate myself before the king, but I rifused."
input_ids, attention_mask = tokenize(input_text, tokenizer, device)

#%%

token_probs: list[tuple[int, float]] = calculate_log_probabilities(model, tokenizer, input_ids, attention_mask)

#%%

import importlib
import text_processing

importlib.reload(text_processing)
from text_processing import split_into_words, Word

words = split_into_words(token_probs, tokenizer)
log_prob_threshold = -5.0
low_prob_words = [word for word in words if word.logprob < log_prob_threshold]

#%%

start_time = time.time()

contexts = [word.context for word in low_prob_words]
replacements_batch = generate_replacements(model, tokenizer, contexts, device)

for word, replacements in zip(low_prob_words, replacements_batch):
    print(f"Original word: {word.text}, Log Probability: {word.logprob:.4f}")
    print(f"Proposed replacements: {replacements}")

end_time = time.time()
print(f"Total time taken for replacements: {end_time - start_time:.4f} seconds")

# %%
