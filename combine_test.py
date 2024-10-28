import pytest
from combine import combine


def test_empty_list():
    assert combine([], lambda x, y: x + y) == []

def test_single_item():
    assert combine(["hello"], lambda x, y: x + y) == ["hello"]

def test_two_items():
    assert combine(["hello", "world"], lambda x, y: x + y) == ["helloworld"]

def test_sum():
    assert combine([1, 2, 3, 4], lambda x, y: x + y) == [10]


def test_add_if_even():
    def add_if_even(x: int, y: int) -> int | None:
        if (x + y) % 2 == 0:
            return x + y
        return None

    assert combine([1, 3, 1, 4], add_if_even) == [4, 1, 4]
    assert combine([1, 3, 2, 4], add_if_even) == [10]


def test_join_if_same_letter():
    def join_if_same_letter(x: str, y: str) -> str | None:
        if x[0] == y[0]:
            return x + y
        return None

    assert combine(["hello", "hi", "home", "world", "welcome"], join_if_same_letter) == ["hellohihome", "worldwelcome"]


def test_no_combinations():
    def never_combine(x: int, y: int) -> None:
        return None

    input_list = [1, 2, 3, 4]
    assert combine(input_list, never_combine) == input_list
