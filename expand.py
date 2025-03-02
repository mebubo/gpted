from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Protocol, Self

@dataclass
class Expansion:
    token: int
    cost: float

@dataclass
class Series:
    id: int
    tokens: list[int]
    budget: float
    expansions: list[Expansion] = field(default_factory=list)

    def get_all_tokens(self) -> list[int]:
        return self.tokens + [e.token for e in self.expansions]

    def get_remaining_budget(self) -> float:
        return self.budget + sum(e.cost for e in self.expansions)

@dataclass
class Batch:
    items: list[Series]

@dataclass
class TokenCandidates:
    series: Series
    expansions: list[Expansion]

@dataclass
class BatchCandidates:
    items: list[TokenCandidates]

# A fundamental operation that we can implement both using an LLM and using a list of hardcoded sequences, for testing
class BatchExpander(Protocol):
    def expand(self, batch: Batch) -> BatchCandidates: ...

@dataclass
class CompletedSequence:
    series: Series
    expansions: list[list[Expansion]]

@dataclass
class CompletedBatch:
    items: list[CompletedSequence]

def compute_new_series(result: TokenCandidates, stopping_criterion: Callable[[Series, Expansion], bool]) -> tuple[list[Series], list[Series]]:
    new_series_batch = []
    for expansion in result.expansions:
        if not stopping_criterion(result.series, expansion):
            new_series = Series(
                id=result.series.id,
                tokens=result.series.tokens,
                expansions=result.series.expansions + [expansion],
                budget=result.series.budget
            )
            new_series_batch.append(new_series)
    completed_series = [result.series] if len(new_series_batch) == 0 else []
    return new_series_batch, completed_series

def compute_expansions(original_series: list[Series], expanded_series: list[Series]) -> CompletedBatch:
    # check that ids in original_series are unique
    assert len(original_series) == len({s.id for s in original_series})
    # group original series by id
    original_series_by_id = {s.id: s for s in original_series}
    # group expanded series by id
    expanded_series_by_id: dict[int, list[list[Expansion]]] = defaultdict(list)
    for s in expanded_series:
        if len(s.expansions) != 0:
            expanded_series_by_id[s.id].append(s.expansions)
    results = []
    for id, s in original_series_by_id.items():
        expansions = expanded_series_by_id[id]
        expansion_result = CompletedSequence(series=s, expansions=expansions)
        results.append(expansion_result)
    return CompletedBatch(items=results)

def default_completion_criterion(series: Series, expansion: Expansion) -> bool:
    return series.get_remaining_budget() + expansion.cost < 0

# A compound operation that we can implement generically, relying on a BatchExpander
def expand(batch: Batch, expander: BatchExpander, completion_criterion: Callable[[Series, Expansion], bool] = default_completion_criterion) -> CompletedBatch:
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
                new_series, completed = compute_new_series(item, completion_criterion)
                completed_series.extend(completed)
                current_batch_items.extend(new_series)
        current_batch = Batch(items=current_batch_items)
    return compute_expansions(batch.items, completed_series)
