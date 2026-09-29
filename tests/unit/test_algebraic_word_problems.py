"""Unit tests for algebraic word problem solver."""

from little.procedural.algebraic_word_problem import AlgebraicWordProblemSolver


def test_algebraic_age_problem_with_temporal_shift():
    solver = AlgebraicWordProblemSolver()
    problem = "Alice is twice as old as Bob. In 5 years, the sum of their ages will be 40. How old is Alice?"
    
    assert solver.is_algebraic_problem(problem) is True
    sol = solver.solve(problem)
    assert sol is not None
    assert sol.status == "SOLVED"
    assert "Alice is 20 years old" in sol.result
    # Verify trace steps
    assert any("Cramer's Rule" in step for step in sol.steps)
    assert any("Variables:" in step for step in sol.steps)


def test_algebraic_age_problem_offset():
    solver = AlgebraicWordProblemSolver()
    problem = "David is 4 years older than Sarah. The sum of their ages is 28. How old is Sarah?"
    
    sol = solver.solve(problem)
    assert sol is not None
    assert sol.status == "SOLVED"
    assert "Sarah is 12 years old" in sol.result


def test_algebraic_cost_linear_system():
    solver = AlgebraicWordProblemSolver()
    problem = "A pen costs 3 dollars more than a notebook. 2 pens and 3 notebooks cost 21 dollars. What is the cost of a pen?"
    
    sol = solver.solve(problem)
    assert sol is not None
    assert sol.status == "SOLVED"
    assert "cost of a pen is $6" in sol.result
