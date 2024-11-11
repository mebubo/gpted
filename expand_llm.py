from expand import *
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast, BatchEncoding
from dataclasses import dataclass
from completions import prepare_inputs, find_next_tokens

type Tokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast

@dataclass
class ExpanderOneBatchLLM:
    model: PreTrainedModel
    tokenizer: Tokenizer

    def expand(self, batch: Batch) -> ExpansionOneResultBatch:
        inputs = prepare_inputs([s.tokens for s in batch.items], self.tokenizer, self.model.device)
        next_tokens = find_next_tokens(self.model, inputs, self.tokenizer)
        results = []
        for s, next_tokens in zip(batch.items, next_tokens):
            expansions = [Expansion(token=token, cost=logprob) for token, logprob in next_tokens if logprob + s.get_remaining_budget() >= 0]
            results.append(ExpansionOneResult(series=s, expansions=expansions))
        return ExpansionOneResultBatch(items=results)
