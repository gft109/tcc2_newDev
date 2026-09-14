"""Runs the exact (Gurobi) and heuristic (memetic) solvers on one instance and compares
them.

Usage:
    python src/main.py [instance] [repetitions]

    instance      name of a data_base/<instance>/ folder (default: pequeno)
    repetitions   independent heuristic runs, for statistical robustness (default: 5)

Both solvers run under the same fixed time ceiling (DEFAULT_TIME_LIMIT, 5 minutes) —
this is no longer a CLI argument, because it stopped being the right basis for
comparing the two solvers: the heuristic's own termination criteria (max generations,
patience-based early stopping — see memetic_solver.py) mean its actual runtime varies
run to run and is no longer a variable under the user's control. The standard
comparison metric is instead `evaluation_count` — how many times a candidate
solution's cost was computed (Gurobi's simplex iteration count vs. the heuristic's
GA + local-search evaluations combined) — since that reflects search effort in a way
that's comparable across both solvers regardless of how much wall-clock time either
one actually used.
"""

from __future__ import annotations

import sys

from solvers.exact.gurobi import GurobiSolver
from solvers.exact.gurobi_formater import GurobiFormatter
from solvers.heuristics.memetic_solver import MemeticSolver
from utils.data_manager import DataManager
from utils.results_representation import ResultsReporter
from utils.solution import SolverResult

DEFAULT_INSTANCE = "pequeno"
DEFAULT_REPETITIONS = 5
DEFAULT_TIME_LIMIT = 300.0


def parse_args(argv: list[str]) -> tuple[str, int]:
    instance_name = argv[1] if len(argv) > 1 else DEFAULT_INSTANCE
    repetitions = int(argv[2]) if len(argv) > 2 else DEFAULT_REPETITIONS
    return instance_name, repetitions


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


def main() -> None:
    instance_name, repetitions = parse_args(sys.argv)
    time_limit = DEFAULT_TIME_LIMIT

    print("=" * 72)
    print("=== INICIANDO AS SIMULAÇÕES ===")
    print(f"Instância selecionada: {instance_name.upper()}")
    print(f"Repetições solicitadas: {repetitions}")
    print(f"Teto de tempo (fixo):  {time_limit} s")
    print(f"Diretório dos dados:   data_base/{instance_name}")
    print(f"Diretório de saída:    results/{instance_name}")
    print("=" * 72)
    print()

    data_manager = DataManager(instance_name)

    print("[1/2] Executando Modelo Exato (Gurobi) com limite de tempo...")
    gurobi_result = run_gurobi(data_manager, time_limit)
    if gurobi_result is not None:
        print(f"  > Gurobi concluído  | Fitness: {gurobi_result.total_cost:<10.1f}| Tempo: {gurobi_result.runtime:.4f} s")
    else:
        print("  > Gurobi não encontrou nenhuma solução viável dentro do limite de tempo.")
    print()

    print(f"[2/2] Executando Algoritmo Memético ({repetitions} rodadas independentes)...")
    heuristic_results = run_heuristic_repetitions(data_manager, repetitions, time_limit)
    print()

    reporter = ResultsReporter(instance_name, gurobi_result, heuristic_results)
    reporter.report_all()


if __name__ == "__main__":
    main()
