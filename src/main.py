"""Runs Gurobi and N heuristic repetitions on one or more instance sets and saves the results.

Usage: python src/main.py [instance] [repetitions] [time_limit] [set]
    set: a number ("3"), a range ("1-5") or "all"; defaults: pequeno 5 300 1
"""

from __future__ import annotations

import sys

from solvers.exact.gurobi import GurobiSolver
from solvers.exact.gurobi_formater import GurobiFormatter
from solvers.heuristics.memetic_solver import MemeticSolver
from utils.data_generator import N_SETS_PER_INSTANCE
from utils.data_manager import DataManager
from utils.results_representation import ResultsReporter
from utils.solution import SolverResult

DEFAULT_INSTANCE = "pequeno"
DEFAULT_REPETITIONS = 5
DEFAULT_TIME_LIMIT = 300.0
DEFAULT_SET_SELECTOR = "1"


def parse_args(argv: list[str]) -> tuple[str, int, float, str]:
    instance_name = argv[1] if len(argv) > 1 else DEFAULT_INSTANCE
    repetitions = int(argv[2]) if len(argv) > 2 else DEFAULT_REPETITIONS
    time_limit = float(argv[3]) if len(argv) > 3 else DEFAULT_TIME_LIMIT
    set_selector = argv[4] if len(argv) > 4 else DEFAULT_SET_SELECTOR
    return instance_name, repetitions, time_limit, set_selector


def parse_set_selector(value: str, total_sets: int = N_SETS_PER_INSTANCE) -> list[int]:
    value = value.strip().lower()

    if value == "all":
        return list(range(1, total_sets + 1))

    if "-" in value:
        start_str, end_str = value.split("-", 1)
        start, end = int(start_str), int(end_str)
        if not (1 <= start <= end <= total_sets):
            raise ValueError(
                f"Intervalo de conjuntos inválido: '{value}' (esperado entre 1 e {total_sets})"
            )
        return list(range(start, end + 1))

    set_number = int(value)
    if not (1 <= set_number <= total_sets):
        raise ValueError(
            f"Conjunto inválido: '{value}' (esperado um número entre 1 e {total_sets}, "
            f"um intervalo tipo '1-5', ou 'all')"
        )
    return [set_number]


def run_gurobi(data_manager: DataManager, time_limit: float) -> SolverResult | None:
    solver = GurobiSolver(data_manager, time_limit=time_limit, verbose=False)
    solver.solve()
    if solver.model.SolCount == 0:
        return None
    return GurobiFormatter(solver).to_solver_result()


def run_heuristic_repetitions(data_manager: DataManager, repetitions: int, time_limit: float) -> list[SolverResult]:
    results = []
    for i in range(repetitions):
        solver = MemeticSolver(data_manager)  # no fixed seed: independent repetitions
        result = solver.solve(time_limit=time_limit)
        print(
            f"  > Repetição {i + 1:02d}/{repetitions:02d} concluída "
            f"| Fitness: {result.total_cost:<10.1f}| Tempo: {result.runtime:.4f} s"
        )
        results.append(result)
    return results


def run_scenario(
    instance_name: str, set_number: int, repetitions: int, time_limit: float, index: int, total: int
) -> None:
    """Runs both solvers on one set and writes CSVs/plots to results/<instance>/set_<NN>/."""
    scenario_path = f"{instance_name}/set_{set_number:02d}"

    print(
        f"[{index}/{total}] Rodando {instance_name.upper()} — conjunto {set_number:02d} "
        f"({repetitions} repetições, {time_limit:.0f}s por solver)..."
    )

    data_manager = DataManager(scenario_path)
    gurobi_result = run_gurobi(data_manager, time_limit)
    heuristic_results = run_heuristic_repetitions(data_manager, repetitions, time_limit)

    reporter = ResultsReporter(scenario_path, gurobi_result, heuristic_results)
    reporter.export_csv()
    reporter.plot_all()

    gurobi_display = f"{gurobi_result.total_cost:.1f}" if gurobi_result is not None else "N/A"
    best_display = f"{reporter.best_result.total_cost:.1f}" if reporter.best_result is not None else "N/A"
    best_feasible_display = (
        f"{reporter.best_feasible_result.best_feasible_fitness:.1f}"
        if reporter.best_feasible_result is not None else "N/A (nenhuma viável encontrada)"
    )
    print(
        f"[{index}/{total}] Conjunto {set_number:02d} concluído | Gurobi: {gurobi_display} "
        f"| Heur. (Melhor): {best_display} | Heur. (Melhor Viável): {best_feasible_display}"
    )
    print(f"          -> CSVs e gráficos em results/{scenario_path}/")
    print()


def main() -> None:
    instance_name, repetitions, time_limit, set_selector_raw = parse_args(sys.argv)
    set_numbers = parse_set_selector(set_selector_raw)
    total = len(set_numbers)

    for index, set_number in enumerate(set_numbers, start=1):
        run_scenario(instance_name, set_number, repetitions, time_limit, index, total)


if __name__ == "__main__":
    main()
