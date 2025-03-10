from dataclasses import dataclass
from expand import Series, BatchExpander, Expansion, Batch, TokenCandidates, BatchCandidates, CompletedSequence, CompletedBatch, expand

possible_sequences = [
    [1, 21, 31, 41],
    [1, 21, 31, 42],
    [1, 21, 32, 41, 51],
    [1, 22, 33, 41],
    [1, 22, 34, 41],
]

def expand_series(series: Series) -> list[Expansion]:
    all_tokens = series.get_all_tokens()
    l = len(all_tokens)
    items = [s[l] for s in possible_sequences if s[:l] == all_tokens and len(s) > l]
    candidates = [Expansion(token=l, cost=-1.0) for l in dict.fromkeys(items)]
    return candidates

class PredefinedSequenceExpander(BatchExpander):
    def expand(self, batch: Batch) -> BatchCandidates:
        result = []
        for s in batch.items:
            expansions = expand_series(s)
            result.append(TokenCandidates(series=s, expansions=expansions))
        return BatchCandidates(items=result)

expander = PredefinedSequenceExpander()

def test_expander_zero_budget():
    s = Series(id=0, tokens=[1], budget=0.0)
    expanded = expander.expand(Batch(items=[s]))
    expected = BatchCandidates(
        items=[TokenCandidates(series=s, expansions=[
            Expansion(token=21, cost=-1.0),
            Expansion(token=22, cost=-1.0),
        ])]
    )
    assert expected == expanded

def test_expander_budget_one():
    s = Series(id=0, tokens=[1], budget=1.0)
    expanded = expander.expand(Batch(items=[s]))
    expected = BatchCandidates(
        items=[TokenCandidates(series=s, expansions=[
            Expansion(token=21, cost=-1.0),
            Expansion(token=22, cost=-1.0),
        ])]
    )
    assert expected == expanded

def test_expander_budget_two():
    s = Series(id=0, tokens=[1], budget=2.0)
    expanded = expander.expand(Batch(items=[s]))
    expected = BatchCandidates(
        items=[TokenCandidates(series=s, expansions=[
            Expansion(token=21, cost=-1.0),
            Expansion(token=22, cost=-1.0),
        ])]
    )
    assert expected == expanded

def test_expander_budget_one_no_expansion():
    s = Series(id=0, tokens=[1, 20], budget=1.0)
    expanded = expander.expand(Batch(items=[s]))
    expected = BatchCandidates(
        items=[TokenCandidates(series=s, expansions=[])]
    )
    assert expected == expanded

def test_expander_budget_one_two_tokens():
    s = Series(id=0, tokens=[1, 22], budget=1.0)
    expanded = expander.expand(Batch(items=[s]))
    expected = BatchCandidates(
        items=[TokenCandidates(series=s, expansions=[
            Expansion(token=33, cost=-1.0),
            Expansion(token=34, cost=-1.0),
        ])]
    )
    assert expected == expanded

def test_expander_budget_one_two_tokens_two_series():
    s1 = Series(id=0, tokens=[1, 21, 31], budget=1.0)
    s2 = Series(id=1, tokens=[1, 22], budget=1.0)
    expanded = expander.expand(Batch(items=[s1, s2]))
    expected = BatchCandidates(
        items=[
            TokenCandidates(series=s1, expansions=[
                Expansion(token=41, cost=-1.0),
                Expansion(token=42, cost=-1.0),
            ]),
            TokenCandidates(series=s2, expansions=[
                Expansion(token=33, cost=-1.0),
                Expansion(token=34, cost=-1.0),
            ])
        ]
    )
    assert expected == expanded

def test_expand_01():
    batch = Batch(items=[
        Series(id=0, tokens=[1, 21], budget=1.0),
        Series(id=1, tokens=[1, 22], budget=1.0),
    ])
    expanded = expand(batch, expander)
    assert expanded == CompletedBatch(items=[
        CompletedSequence(
            series=Series(id=0, tokens=[1, 21], budget=1.0),
            expansions=[
                [Expansion(token=31, cost=-1.0)],
                [Expansion(token=32, cost=-1.0)],
            ]
        ),
        CompletedSequence(
            series=Series(id=1, tokens=[1, 22], budget=1.0),
            expansions=[
                [Expansion(token=33, cost=-1.0)],
                [Expansion(token=34, cost=-1.0)],
            ]
        ),
    ])

def test_expand_02():
    batch = Batch(items=[
        Series(id=0, tokens=[1, 21], budget=2.0),
        Series(id=1, tokens=[1, 22], budget=1.0),
    ])
    expanded = expand(batch, expander)
    assert expanded == CompletedBatch(items=[
        CompletedSequence(
            series=Series(id=0, tokens=[1, 21], budget=2.0),
            expansions=[
                [Expansion(token=31, cost=-1.0), Expansion(token=41, cost=-1.0)],
                [Expansion(token=31, cost=-1.0), Expansion(token=42, cost=-1.0)],
                [Expansion(token=32, cost=-1.0), Expansion(token=41, cost=-1.0)],
            ]
        ),
        CompletedSequence(
            series=Series(id=1, tokens=[1, 22], budget=1.0),
            expansions=[
                [Expansion(token=33, cost=-1.0)],
                [Expansion(token=34, cost=-1.0)],
            ]
        ),
    ])

def test_expand_03():
    batch = Batch(items=[
        Series(id=0, tokens=[1, 21], budget=3.0),
        Series(id=1, tokens=[1, 22], budget=0.0),
    ])
    expanded = expand(batch, expander)
    assert expanded == CompletedBatch(items=[
        CompletedSequence(
            series=Series(id=0, tokens=[1, 21], budget=3.0),
            expansions=[
                [Expansion(token=31, cost=-1.0), Expansion(token=41, cost=-1.0)],
                [Expansion(token=31, cost=-1.0), Expansion(token=42, cost=-1.0)],
                [Expansion(token=32, cost=-1.0), Expansion(token=41, cost=-1.0), Expansion(token=51, cost=-1.0)],
            ]
        ),
        CompletedSequence(
            series=Series(id=1, tokens=[1, 22], budget=0.0),
            expansions=[],
        ),
    ])
