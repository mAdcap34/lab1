
from grader_contracts.python_basics import (
    PositiveIntegerInput,
    TextInput,
    VectorPairInput,
)

from python_basics import (
    count_vowels,
    has_unique_characters,
    count_one_bits,
    multiplicative_persistence,
    mse,
    prime_factorization,
    pyramid,
    is_balanced_number,
)


def test_count_vowels():
    assert count_vowels(TextInput(value="Hello World")) == 3
    assert count_vowels(TextInput(value="AEIOU")) == 5
    assert count_vowels(TextInput(value="rhythm")) == 0
    assert count_vowels(TextInput(value="")) == 0


def test_has_unique_characters():
    assert has_unique_characters(TextInput(value="python")) is True
    assert has_unique_characters(TextInput(value="hello")) is False
    assert has_unique_characters(TextInput(value="")) is True
    assert has_unique_characters(TextInput(value="Aa")) is True


def test_count_one_bits():
    assert count_one_bits(PositiveIntegerInput(value=13)) == 3
    assert count_one_bits(PositiveIntegerInput(value=1)) == 1
    assert count_one_bits(PositiveIntegerInput(value=8)) == 1


def test_multiplicative_persistence():
    assert multiplicative_persistence(
        PositiveIntegerInput(value=39)
    ) == 3
    assert multiplicative_persistence(
        PositiveIntegerInput(value=4)
    ) == 0
    assert multiplicative_persistence(
        PositiveIntegerInput(value=999)
    ) == 4


def test_mse():
    assert mse(VectorPairInput(predicted=[1, 2], expected=[3, 4])) == 4.0
    assert mse(VectorPairInput(predicted=[1, 2], expected=[1, 2])) == 0.0


def test_prime_factorization():
    assert prime_factorization(
        PositiveIntegerInput(value=86240)
    ) == "(2**5)(5)(7**2)(11)"
    assert prime_factorization(
        PositiveIntegerInput(value=13)
    ) == "(13)"
    assert prime_factorization(
        PositiveIntegerInput(value=1)
    ) == ""


def test_pyramid():
    assert pyramid(PositiveIntegerInput(value=14)) == 3
    assert pyramid(PositiveIntegerInput(value=1)) == 1
    assert pyramid(PositiveIntegerInput(value=2)) == "It is impossible"


def test_is_balanced_number():
    assert is_balanced_number(
        PositiveIntegerInput(value=1234006)
    ) is True
    assert is_balanced_number(
        PositiveIntegerInput(value=123456)
    ) is False
    assert is_balanced_number(
        PositiveIntegerInput(value=121)
    ) is True
    assert is_balanced_number(
        PositiveIntegerInput(value=11)
    ) is True
