from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol

@dataclass
class Series:
    id: int
    tokens: list[int]
    budget: float

@dataclass
class Batch:
    items: list[Series]

@dataclass
class ExpansionOne:
    token: int
    cost: float

@dataclass
class ExpansionOneResult:
    series: Series
    expansions: list[ExpansionOne]

@dataclass
class ExpansionOneResultBatch:
    items: list[ExpansionOneResult]

# A fundamental operation that we can implement both using an LLM and using a list of hardcoded sequences, for testing
class ExpanderOneBatch(Protocol):
    def expand(self, batch: Batch) -> ExpansionOneResultBatch: ...

@dataclass
class ExpansionResult:
    series: Series
    expansions: list[list[int]]

@dataclass
class ExpansionResultBatch:
    items: list[ExpansionResult]

def compute_new_series(result: ExpansionOneResult) -> list[Series]:
    results = []
    for expansion in result.expansions:
        results.append(Series(id=result.series.id, tokens=result.series.tokens + [expansion.token], budget=result.series.budget + expansion.cost))
    return results

def compute_expansions(original_series: list[Series], expanded_series: list[Series]) -> ExpansionResultBatch:
    # check that ids in original_series are unique
    assert len(original_series) == len({s.id for s in original_series})
    # group original series by id
    original_series_by_id = {s.id: s for s in original_series}
    # group expanded series by id
    expanded_series_by_id: dict[int, list[list[int]]] = defaultdict(list)
    for s in expanded_series:
        expanded_series_by_id[s.id].append(s.tokens)
    results = []
    for id, s in original_series_by_id.items():
        expansions = expanded_series_by_id[id]
        # subtract the original series from each expansion
        l = len(s.tokens)
        trimmed_expansions = [e[l:] for e in expansions if len(e) > l]
        expansion_result = ExpansionResult(series=s, expansions=trimmed_expansions)
        results.append(expansion_result)
    return ExpansionResultBatch(items=results)

# A compound operation that we can implement generically, relying on an ExpanderOneBatch
def expand(batch: Batch, expander: ExpanderOneBatch) -> ExpansionResultBatch:
    completed_series: list[Series] = []
    current_batch = batch
    while len(current_batch.items) > 0:
        print(f"Expanding {len(current_batch.items)} series: {current_batch.items}")
        current_batch_items = []
        expanded = expander.expand(current_batch)
        for item in expanded.items:
            if len(item.expansions) == 0:
                completed_series.append(item.series)
            else:
                current_batch_items.extend(compute_new_series(item))
        current_batch = Batch(items=current_batch_items)
    return compute_expansions(batch.items, completed_series)
