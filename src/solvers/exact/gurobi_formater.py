"""Converts a solved GurobiSolver into the shared SolverResult format (utils/solution.py),
so the exact model's output can be compared fairly against the heuristic's — same cost
breakdown, same fields, per the terminal output format documented in CLAUDE.md.
"""

from __future__ import annotations

from solvers.exact.gurobi import GurobiSolver
from utils.solution import Solution, SolverResult


class GurobiFormatter:
    def __init__(self, solver: GurobiSolver):
        self.solver = solver

    def to_solver_result(self) -> SolverResult:
        if self.solver.model is None or self.solver.model.SolCount == 0:
            raise RuntimeError("Gurobi solver has no incumbent solution to format.")

        assignment = self.solver.extract_assignment()
        solution = Solution(self.solver.data_manager, assignment)

        return SolverResult(
            solution=solution,
            runtime=self.solver.runtime,
            initial_fitness=None,
            evaluation_count=int(self.solver.model.IterCount),
        )
