"""Integration tests verifying end-to-end AST Code Synthesis and Sandbox Verification in Conversational Chat."""

from little.core.kernel import CognitiveKernel
from little.memory.store import MemoryStore


def test_conversational_code_synthesis_prime_and_fibonacci():
    """Verify chat queries requesting code synthesize valid, verified Python functions."""
    memory = MemoryStore(":memory:", seed_ontology=True)
    kernel = CognitiveKernel(memory=memory)

    # 1. Ask for prime check function
    reply_prime = kernel.chat("Write a python function to check if a number is prime")
    assert "def is_prime(" in reply_prime
    assert "VERIFIED" in reply_prime
    assert "sandbox" in reply_prime.lower()
    assert "```python" in reply_prime

    # 2. Ask for fibonacci
    reply_fib = kernel.chat("Can you generate code for fibonacci?")
    assert "def fibonacci(" in reply_fib
    assert "VERIFIED" in reply_fib
    assert "sandbox" in reply_fib.lower()


def test_conversational_code_synthesis_binary_search_and_gcd():
    """Verify algorithmic search and math primitives synthesize in chat."""
    memory = MemoryStore(":memory:", seed_ontology=True)
    kernel = CognitiveKernel(memory=memory)

    # 1. Binary search
    reply_bs = kernel.chat("Generate python code for binary search")
    assert "def binary_search(" in reply_bs
    assert "while low <= high:" in reply_bs
    assert "VERIFIED" in reply_bs

    # 2. Greatest Common Divisor
    reply_gcd = kernel.chat("Write code for greatest common divisor")
    assert "def gcd(" in reply_gcd
    assert "VERIFIED" in reply_gcd


def test_conversational_code_synthesis_string_and_list_manipulation():
    """Verify palindrome, reverse string, and list utilities synthesize and verify."""
    memory = MemoryStore(":memory:", seed_ontology=True)
    kernel = CognitiveKernel(memory=memory)

    # 1. Palindrome check
    reply_pal = kernel.chat("Synthesize python code to check if a string is a palindrome")
    assert "def is_palindrome(" in reply_pal
    assert "VERIFIED" in reply_pal

    # 2. Flatten 2D list
    reply_flat = kernel.chat("Write a function to flatten a list")
    assert "def flatten(" in reply_flat
    assert "VERIFIED" in reply_flat


def test_conversational_distinction_between_eval_and_synthesis():
    """Verify LITTLE correctly distinguishes between computing an answer vs synthesizing code."""
    memory = MemoryStore(":memory:", seed_ontology=True)
    kernel = CognitiveKernel(memory=memory)

    # Question about prime truth value (computational math query)
    res_val = kernel.chat("Is 104729 prime?")
    assert "yes" in res_val.lower() or "true" in res_val.lower()
    assert "def is_prime" not in res_val

    # Explicit code generation request
    res_code = kernel.chat("Write a function to check if a number is prime")
    assert "def is_prime(" in res_code
    assert "VERIFIED" in res_code
