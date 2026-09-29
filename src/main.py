"""Runs the exact (Gurobi) and heuristic (memetic) solvers on one or more generated
sets of an instance size and compares them.

Usage:
    python src/main.py [instance] [repetitions] [time_limit] [set]

    instance       name of a data_base/<instance>/ folder (default: pequeno)
    repetitions    independent heuristic runs, for statistical robustness (default: 5)
    time_limit     seconds each solver gets — same budget for Gurobi and for every
                   heuristic repetition (default: 300.0)
    set            which of the N_SETS_PER_INSTANCE generated sets to run (see
                   data_generator.py): a single number ("3"), an inclusive range
                   ("1-5"), or "all" (default: "1", i.e. just the first set)

Each selected set is a fully independent run against its own data_base/<instance>/
set_<NN>/room.csv + patient.csv, reported to its own results/<instance>/set_<NN>/ —
running several sets back to back does not aggregate them into one combined report
(each set keeps the same per-set CSVs/plots this always produced); that's a deliberate
scope choice for now, not an oversight.

Terminal output per set is a condensed "[i/total] ..." progress line (start, then
completion with Gurobi's cost, the heuristic's best, and its best capacity-feasible
solution — see ResultsReporter.best_feasible_result) plus the existing per-repetition
heuristic lines, rather than the full cost-breakdown table
(ResultsReporter.print_summary_table) — that table is still there and unchanged, just
not printed by this loop, since dumping it once per set made a "1-5"/"all" run
unreadable. The CSVs/plots it would summarize are written all the same
(export_csv + plot_all, still called every set); call print_summary_table() yourself
(e.g. in a script or REPL against a ResultsReporter you construct) if you want it for
one specific set.

The standard comparison metric is `evaluation_count` — how many times a candidate
solution's cost was computed (Gurobi's simplex iteration count vs. the heuristic's GA +
local-search evaluations combined) — since that reflects search effort in a way that's
comparable across both solvers even when one of them (Gurobi, on the larger instances)
barely gets past its initial setup within the time budget.
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
    """Parses the 4th CLI argument into the list of set numbers to run: a single
    number ("3"), an inclusive range ("1-5"), or "all" (every set, 1..total_sets)."""
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
        # No fixed seed: each repetition draws its own randomness (Python seeds from OS
        # entropy when seed=None), rather than reproducing the same RNG trajectory every
        # time main.py runs. Section 3.1.4's determinism preference is about the input
        # data (data_generator.py's per-instance seeds keep room.csv/patient.csv fixed
        # across runs) — it isn't meant to extend to the heuristic's own search
        # randomness, and a fixed seed=i risked baking in one particular repetition's
        # quirks (e.g. an early-converging run whose result doesn't change no matter
        # what stagnation-handling behavior is tested) rather than giving genuinely
        # independent samples for the mean/std reported across repetitions.
        solver = MemeticSolver(data_manager)
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
    """Runs the full Gurobi-vs-heuristic comparison for a single generated set,
    reading data_base/<instance_name>/set_<NN>/ and writing to
    results/<instance_name>/set_<NN>/ — everything main() used to do directly, before
    it could loop over more than one set (see the module docstring's --set option).

    Terminal output is deliberately condensed to a [index/total] progress line per
    set (plus the existing per-repetition heuristic lines, still useful live feedback
    within a set) instead of the full cost-breakdown table print_summary_table()
    normally prints — that table stays fine for one run at a time, but dumping it once
    per set made a "1-5"/"all" run unreadable. CSVs and plots are unaffected: still
    the exact same files ResultsReporter.report_all() would have produced."""
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
