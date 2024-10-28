from functools import reduce
from typing import Callable


def combine[T](items: list[T], combine_fn: Callable[[T, T], T | None]) -> list[T]:
    def fold_fn(acc: list[T], item: T) -> list[T]:
        if not acc:
            return [item]

        combined = combine_fn(acc[-1], item)
        if combined is not None:
            return [*acc[:-1], combined]
        return [*acc, item]

    result = reduce(fold_fn, items, [])

    return result
