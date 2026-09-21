"""Unit tests for Python programming skills and procedural reasoning in LITTLE."""

from little.core.models import BeliefStatus
from little.language.parser import LearningEngine
from little.memory.store import MemoryStore


def test_python_procedural_math_and_algorithms():
    store = MemoryStore(":memory:")
    engine = LearningEngine(store)

    # 1. Fibonacci
    res_fib = engine.ask("What is the fibonacci of 25?")
    assert res_fib.status == BeliefStatus.SUPPORTED
    assert res_fib.answer == 75025

    # 2. Primality test (104729 is the 10,000th prime)
    res_prime_true = engine.ask("Is 104729 prime?")
    assert res_prime_true.status == BeliefStatus.SUPPORTED
    assert res_prime_true.answer is True

    # Composite number
    res_prime_false = engine.ask("Is 104730 prime?")
    assert res_prime_false.status == BeliefStatus.SUPPORTED
    assert res_prime_false.answer is False

    # 3. String reverse
    res_rev = engine.ask('What is the reverse of "antigravity"?')
    assert res_rev.status == BeliefStatus.SUPPORTED
    assert res_rev.answer == "ytivargitna"

    # 4. Palindrome check
    res_pal_true = engine.ask('Is "racecar" a palindrome?')
    assert res_pal_true.status == BeliefStatus.SUPPORTED
    assert res_pal_true.answer is True

    res_pal_false = engine.ask('Is "python" a palindrome?')
    assert res_pal_false.status == BeliefStatus.SUPPORTED
    assert res_pal_false.answer is False
