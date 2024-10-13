#%%
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast, BatchEncoding
from transformers.generation.utils import GenerateOutput
from typing import cast
from dataclasses import dataclass

from models import ApiWord, Word

type Tokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast

def starts_with_space(token: str) -> bool:
    return token.startswith(chr(9601)) or token.startswith(chr(288))

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
        if not starts_with_space(token) and token.isalpha():
            current_word.append(token_id)
            current_log_probs.append(logprob)
        else:
            append_current_word()
            current_word = [token_id]
            current_log_probs = [logprob]
            current_word_first_token_index = i

    append_current_word()

    return words

def load_model_and_tokenizer(model_name: str, device: torch.device) -> tuple[PreTrainedModel, Tokenizer]:
    tokenizer: Tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    return model, tokenizer

def tokenize(input_text: str, tokenizer: Tokenizer, device: torch.device) -> BatchEncoding:
    return tokenizer(input_text, return_tensors="pt").to(device)

def calculate_log_probabilities(model: PreTrainedModel, tokenizer: Tokenizer, inputs: BatchEncoding) -> list[tuple[int, float]]:
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
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

def prepare_inputs(contexts: list[list[int]], tokenizer: Tokenizer, device: torch.device) -> BatchEncoding:
    texts = [tokenizer.decode(context, skip_special_tokens=True) for context in contexts]
    return tokenizer(texts, return_tensors="pt", padding=True).to(device)

def generate_outputs(model: PreTrainedModel, inputs: BatchEncoding, num_samples: int = 5) -> GenerateOutput | torch.LongTensor:
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=4,
            num_return_sequences=num_samples,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            do_sample=True
            # num_beams=num_samples
        )
    return outputs

def extract_replacements(outputs: GenerateOutput | torch.LongTensor, tokenizer: Tokenizer, num_inputs: int, input_len: int, num_samples: int = 5) -> list[list[str]]:
    all_new_words = []
    for i in range(num_inputs):
        replacements = set()
        for j in range(num_samples):
            generated_ids = outputs[i * num_samples + j][input_len:]
            new_word = tokenizer.convert_ids_to_tokens(generated_ids.tolist())[0]
            if starts_with_space(new_word):
                replacements.add(" " +new_word[1:])
        all_new_words.append(sorted(list(replacements)))
    return all_new_words

#%%

def load_model() -> tuple[PreTrainedModel, Tokenizer, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model_name = "mistralai/Mistral-7B-v0.1"
    model_name = "unsloth/Llama-3.2-1B"
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    return model, tokenizer, device

def check_text(input_text: str, model: PreTrainedModel, tokenizer: Tokenizer, device: torch.device) -> list[ApiWord]:
#%%
    inputs: BatchEncoding = tokenize(input_text, tokenizer, device)

    #%%
    token_probs: list[tuple[int, float]] = calculate_log_probabilities(model, tokenizer, inputs)

    #%%
    words = split_into_words(token_probs, tokenizer)
    log_prob_threshold = -5.0
    low_prob_words = [(i, word) for i, word in enumerate(words) if word.logprob < log_prob_threshold]

    #%%
    contexts = [word.context for _, word in low_prob_words]
    inputs = prepare_inputs(contexts, tokenizer, device)
    input_ids = inputs["input_ids"]

    #%%
    num_samples = 10
    start_time = time.time()
    outputs = generate_outputs(model, inputs, num_samples)
    end_time = time.time()
    print(f"Total time taken for replacements: {end_time - start_time:.4f} seconds")

    #%%
    replacements = extract_replacements(outputs, tokenizer, input_ids.shape[0], input_ids.shape[1], num_samples)

    low_prob_words_with_replacements = { i: (w, r) for (i, w), r in zip(low_prob_words, replacements) }

    result = []
    for i, word in enumerate(words):
        if i in low_prob_words_with_replacements:
            result.append(ApiWord(text=word.text, logprob=word.logprob, replacements=low_prob_words_with_replacements[i][1]))
        else:
            result.append(ApiWord(text=word.text, logprob=word.logprob, replacements=[]))
    return result
