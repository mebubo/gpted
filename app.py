#%%
import time
from tqdm import tqdm
from text_processing import split_into_words, Word
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast, BatchEncoding
from tokenizers import Encoding
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

def calculate_log_probabilities(model: PreTrainedModel, tokenizer: Tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> list[tuple[str, float]]:
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    # B x T x V
    logits: torch.Tensor = outputs.logits[:, :-1, :]
    # B x T x V
    log_probs: torch.Tensor = torch.log_softmax(logits, dim=-1)
    # T - 1
    token_log_probs: torch.Tensor = log_probs[0, range(log_probs.shape[1]), input_ids[0][1:]]
    # T - 1
    tokens: list[str] = tokenizer.convert_ids_to_tokens(input_ids[0])[1:]
    return list(zip(tokens, token_log_probs.tolist()))


def generate_replacements(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix_tokens: list[int], device: torch.device, num_samples: int = 5) -> list[str]:
    input_context = {"input_ids": torch.tensor([prefix_tokens]).to(device)}
    input_ids = input_context["input_ids"]
    attention_mask = input_context["attention_mask"]
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
    new_words = []
    for i in range(num_samples):
        generated_ids = outputs[i][input_ids.shape[-1]:]
        new_word = tokenizer.decode(generated_ids, skip_special_tokens=True).split()[0]
        new_words.append(new_word)
    return new_words

#%%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "mistralai/Mistral-7B-v0.1"
model, tokenizer = load_model_and_tokenizer(model_name, device)

#%%

input_text = "He asked me to prostrate myself before the king, but I rifused."
input_ids, attention_mask = tokenize(input_text, tokenizer, device)

#%%

token_probs: list[tuple[str, float]] = calculate_log_probabilities(model, tokenizer, input_ids, attention_mask)

#%%

words = split_into_words(token_probs)
log_prob_threshold = -5.0
low_prob_words = [word for word in words if word.logprob < log_prob_threshold]

#%%

start_time = time.time()

for word in tqdm(low_prob_words, desc="Processing words"):
    iteration_start_time = time.time()
    prefix_index = word.first_token_index
    prefix_tokens = tokenizer.convert_tokens_to_ids([token for token, _ in token_probs][:prefix_index + 1])
    replacements = generate_replacements(model, tokenizer, prefix_tokens, device)
    print(f"Original word: {word.text}, Log Probability: {word.logprob:.4f}")
    print(f"Proposed replacements: {replacements}")
    print()
    iteration_end_time = time.time()
    print(f"Time taken for this iteration: {iteration_end_time - iteration_start_time:.4f} seconds")

end_time = time.time()
print(f"Total time taken for the loop: {end_time - start_time:.4f} seconds")

# %%
