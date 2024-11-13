import torch
from expand import *
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast, BatchEncoding
from dataclasses import dataclass

type Tokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast

def find_next_tokens(model: PreTrainedModel, inputs: BatchEncoding, tokenizer: Tokenizer) -> list[list[tuple[int, float]]]:
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits: torch.Tensor = outputs.logits[:, -1, :]
    log_probs: torch.Tensor = torch.log_softmax(logits, dim=-1)
    result = []
    for probs in log_probs:
        result.append([(i, p.item()) for i, p in enumerate(probs)])
    return result

def prepare_inputs(contexts: list[list[int]], tokenizer: Tokenizer, device: torch.device) -> BatchEncoding:
    texts = [tokenizer.decode(context, skip_special_tokens=True) for context in contexts]
    return tokenizer(texts, return_tensors="pt", padding=True).to(device)

@dataclass
class ExpanderOneBatchLLM:
    model: PreTrainedModel
    tokenizer: Tokenizer

    def expand(self, batch: Batch) -> ExpansionOneResultBatch:
        inputs = prepare_inputs([s.get_all_tokens() for s in batch.items], self.tokenizer, self.model.device)
        next_tokens = find_next_tokens(self.model, inputs, self.tokenizer)
        results = []
        for s, next_tokens in zip(batch.items, next_tokens):
            expansions = [Expansion(token=token, cost=cost) for token, cost in next_tokens]
            results.append(ExpansionOneResult(series=s, expansions=expansions))
        return ExpansionOneResultBatch(items=results)

def create_stopping_criterion_llm(tokenizer: Tokenizer) -> Callable[[Series, Expansion], bool]:
    def stopping_criterion(series: Series, expansion: Expansion) -> bool:
        d = default_completion_criterion(series, expansion)
        if d:
            return d
        token_str = tokenizer.decode([expansion.token])
        starts_with_space = token_str.startswith(" ")
        print(f"-----{token_str}-----, {starts_with_space=}")
        is_first_token = len(series.expansions) == 0
        if is_first_token and not starts_with_space:
            return True
        if not is_first_token and starts_with_space:
            return True
        return False
    return stopping_criterion
