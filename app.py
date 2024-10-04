#%%
from text_processing import split_into_words, Word
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pprint import pprint

#%%

model_name="mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#%%

input_text = "I just drive to the store to but eggs, but they had some."
input_text = "He asked me to prostate myself before the king, but I rifused."
input_text = "He asked me to prostrate myself before the king, but I rifused."

#%%
inputs = tokenizer(input_text, return_tensors="pt").to(device)
input_ids = inputs["input_ids"]
labels = input_ids

#%%
with torch.no_grad():
    outputs = model(**inputs, labels=labels)

#%%

# Get logits and shift them
logits = outputs.logits[0, :-1, :]

# Calculate log probabilities
log_probs = torch.log_softmax(logits, dim=-1)

# Get the log probability of each token in the sequence
token_log_probs = log_probs[range(log_probs.shape[0]), input_ids[0][1:]]

# Decode tokens and pair them with their log probabilities
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
result = list(zip(tokens[1:], token_log_probs.tolist()))

#%%
for token, logprob in result:
    print(f"Token: {token}, Log Probability: {logprob:.4f}")

# %%
words = []
current_word = []
current_log_probs = []

for token, logprob in result:
    if not token.startswith(chr(9601)) and token.isalpha():
        current_word.append(token)
        current_log_probs.append(logprob)
    else:
        if current_word:
            words.append(("".join(current_word), sum(current_log_probs)))
        current_word = [token]
        current_log_probs = [logprob]

if current_word:
    words.append(("".join(current_word), sum(current_log_probs)))

for word, avg_logprob in words:
    print(f"Word: {word}, Log Probability: {avg_logprob:.4f}")

# %%


words = split_into_words(tokens[1:], token_log_probs)


#%%
def generate_replacements(model, tokenizer, prefix, num_samples=5):
    input_context = tokenizer(prefix, return_tensors="pt").to(device)
    input_ids = input_context["input_ids"]

    new_words = []
    for _ in range(num_samples):
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[-1] + 5,  # generate a few tokens beyond the prefix
                num_return_sequences=1,
                temperature=1.0,
                top_k=50,     # use top-k sampling
                top_p=0.95,   # use nucleus sampling
                do_sample=True
            )

        generated_ids = outputs[0][input_ids.shape[-1]:]  # extract the newly generated part
        new_word = tokenizer.decode(generated_ids, skip_special_tokens=True).split()[0]
        new_words.append(new_word)

    return new_words

# Generate new words for low probability words
for word in low_prob_words:
    prefix_index = word.first_token_index
    prefix_tokens = tokens[:prefix_index + 1]  # include the word itself
    prefix = tokenizer.convert_tokens_to_string(prefix_tokens)

    replacements = generate_replacements(model, tokenizer, prefix)

    print(f"Original word: {word.text}, Log Probability: {word.logprob:.4f}")
    print(f"Proposed replacements: {replacements}")
    print()

# %%
