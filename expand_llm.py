import torch
from expand import *
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast, BatchEncoding
from dataclasses import dataclass
import time

type Tokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast

def find_next_tokens(model: PreTrainedModel, inputs: BatchEncoding, threshold: float) -> list[list[tuple[int, float]]]:
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    print("Running inference")
    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    print(f"Inference done, took {time.time() - start_time} seconds")
    start_time = time.time()
    logits: torch.Tensor = outputs.logits[:, -1, :]
    log_probs: torch.Tensor = torch.log_softmax(logits, dim=-1)
    print(f"Log probs done, took {time.time() - start_time} seconds")
    start_time = time.time()
    result = []
    print(f"Resulting tensor: {log_probs.shape}")
    for probs in log_probs:
        # Filter out low probability tokens for efficiency
        above_threshold = torch.where(probs > threshold)
        filtered_indices = above_threshold[0]
        filtered_probs = probs[filtered_indices]
        result.append([(idx.item(), prob.item()) for idx, prob in zip(filtered_indices, filtered_probs)])
    print(f"Result done, took {time.time() - start_time} seconds")
    return result

def prepare_inputs(contexts: list[list[int]], tokenizer: Tokenizer, device: torch.device) -> BatchEncoding:
    texts = [tokenizer.decode(context, skip_special_tokens=True) for context in contexts]
    return tokenizer(texts, return_tensors="pt", padding=True).to(device)

@dataclass
class LLMBatchExpander(BatchExpander):
    model: PreTrainedModel
    tokenizer: Tokenizer
    threshold: float

    def expand(self, batch: Batch) -> BatchCandidates:
        inputs = prepare_inputs([s.get_all_tokens() for s in batch.items], self.tokenizer, self.model.device)
        next_tokens = find_next_tokens(self.model, inputs, self.threshold)
        start_time = time.time()
        results = []
        print(f"Batch size: {len(batch.items)}, next tokens size: {len(next_tokens)}")
        for s, next_tokens in zip(batch.items, next_tokens):
            expansions = [Expansion(token=token, cost=cost) for token, cost in next_tokens]
            results.append(TokenCandidates(series=s, expansions=expansions))
        print()
        print(f"Token candidates done, took {time.time() - start_time} seconds")
        return BatchCandidates(items=results)

def create_stopping_criterion_llm(tokenizer: Tokenizer) -> Callable[[Series, Expansion], bool]:
    def stopping_criterion(series: Series, expansion: Expansion) -> bool:
        d = default_completion_criterion(series, expansion)
        if d:
            return d
        token_str = tokenizer.decode([expansion.token])
        starts_with_space = token_str.startswith(" ")
        # print(f"-----{token_str}-----, {starts_with_space=}")
        is_first_token = len(series.expansions) == 0
        if is_first_token and not starts_with_space:
            return True
        if not is_first_token and starts_with_space:
            return True
        return False
    return stopping_criterion
