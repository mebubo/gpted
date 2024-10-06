#%%
import time
from tqdm import tqdm
from text_processing import split_into_words, Word
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
from pprint import pprint

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

def process_input_text(input_text, tokenizer, device):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    return inputs, input_ids

def calculate_log_probabilities(model, tokenizer, inputs, input_ids):
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    logits = outputs.logits[0, :-1, :]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[range(log_probs.shape[0]), input_ids[0][1:]]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    return list(zip(tokens[1:], token_log_probs.tolist()))


def generate_replacements(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix: str, device: torch.device, num_samples: int = 5) -> list[str]:
    input_context = tokenizer(prefix, return_tensors="pt").to(device)
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
model_name = "mistralai/Mistral-7B-v0.1"
model, tokenizer, device = load_model_and_tokenizer(model_name)

input_text = "He asked me to prostrate myself before the king, but I rifused."
inputs, input_ids = process_input_text(input_text, tokenizer, device)

result = calculate_log_probabilities(model, tokenizer, inputs, input_ids)

words = split_into_words([token for token, _ in result], [logprob for _, logprob in result])
log_prob_threshold = -5.0
low_prob_words = [word for word in words if word.logprob < log_prob_threshold]

#%%

start_time = time.time()

for word in tqdm(low_prob_words, desc="Processing words"):
    iteration_start_time = time.time()
    prefix_index = word.first_token_index
    prefix_tokens = [token for token, _ in result][:prefix_index + 1]
    prefix = tokenizer.convert_tokens_to_string(prefix_tokens)
    replacements = generate_replacements(model, tokenizer, prefix, device)
    print(f"Original word: {word.text}, Log Probability: {word.logprob:.4f}")
    print(f"Proposed replacements: {replacements}")
    print()
    iteration_end_time = time.time()
    print(f"Time taken for this iteration: {iteration_end_time - iteration_start_time:.4f} seconds")

end_time = time.time()
print(f"Total time taken for the loop: {end_time - start_time:.4f} seconds")

# %%
