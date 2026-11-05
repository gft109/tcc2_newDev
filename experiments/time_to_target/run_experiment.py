"""Time-to-target: how long each method takes to reach the other method's final result.

The protocol runs in results/ log the best solution per generation (heuristic) and only the
final incumbent (Gurobi), not the time of each improvement, so the times are measured here:
- heuristic: on medio/grande/muito_grande the best of the repaired initial population already
  beats Gurobi's final Z in every protocol run, so its time-to-target is at most the time to
  build and repair the initial population (as in MemeticSolver.solve), measured for 3 seeds
  per set, 4 processes in parallel as in the protocol;
- Gurobi: pequeno is solved again with the protocol ceiling, logging every new incumbent, to
  find when it reaches the mean Z of the heuristic runs. On the larger instances its final Z is
  above every heuristic run, so it never reaches the target within the ceiling.

Usage: python experiments/time_to_target/run_experiment.py
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from multiprocessing import Pool

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pandas as pd

from utils.data_manager import DataManager
from utils.solution import Solution

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PROTOCOL_DIR = os.path.join(PROJECT_ROOT, "results")

TIME_LIMITS = {"pequeno": 30.0, "medio": 120.0, "grande": 480.0, "muito_grande": 1920.0}
N_SETS = 10
SEEDS = (1, 2, 3)
HEURISTIC_WORKERS = 4
GUROBI_WORKERS = 2


def protocol_set_dir(instance: str, set_number: int) -> str:
    return os.path.join(PROTOCOL_DIR, instance, f"set_{set_number:02d}")


def gurobi_final_z(instance: str, set_number: int) -> float:
    return float(pd.read_csv(os.path.join(protocol_set_dir(instance, set_number), "gurobi_run.csv"))["total_cost"].iloc[0])


def heuristic_final_z(instance: str, set_number: int) -> list[float]:
    return pd.read_csv(os.path.join(protocol_set_dir(instance, set_number), "heuristic_runs.csv"))["total_cost"].tolist()


def protocol_runs_reaching_gurobi(instance: str) -> tuple[int, int, int]:
    """(runs whose initial population beats Gurobi's final Z, runs that reach it at any
    generation, total runs), from the saved protocol CSVs."""
    initial_beats = reaches = total = 0
    for set_number in range(1, N_SETS + 1):
        target = gurobi_final_z(instance, set_number)
        runs = pd.read_csv(os.path.join(protocol_set_dir(instance, set_number), "heuristic_runs.csv"))
        convergence = pd.read_csv(os.path.join(protocol_set_dir(instance, set_number), "convergence.csv"))
        for _, run in runs.iterrows():
            total += 1
            initial_beats += bool(run["initial_fitness"] < target)
            history = convergence[convergence["iteration"] == run["iteration"]]["best_fitness"]
            reaches += bool((history <= target).any())
    return initial_beats, reaches, total


def time_initial_population(args: tuple[str, int, int]) -> dict:
    instance, set_number, seed = args
    from solvers.heuristics.memetic_solver import MemeticSolver

    data_manager = DataManager(f"{instance}/set_{set_number:02d}")
    solver = MemeticSolver(data_manager, seed=seed)
    start = time.perf_counter()
    population = solver.genetic.create_initial_population()
    for individual in population:
        if individual.capacity_violation > 0:
            solver.repair.repair(individual)
    population.sort(key=Solution.sort_key)
    elapsed = time.perf_counter() - start
    return {
        "instance": instance,
        "set": set_number,
        "seed": seed,
        "initial_population_seconds": elapsed,
        "initial_best_z": population[0].objective_value,
        "gurobi_final_z": gurobi_final_z(instance, set_number),
    }


def gurobi_incumbents(set_number: int) -> list[dict]:
    from gurobipy import GRB

    from solvers.exact.gurobi import GurobiSolver

    solver = GurobiSolver(DataManager(f"pequeno/set_{set_number:02d}"), time_limit=TIME_LIMITS["pequeno"])
    solver.build_model()
    incumbents: list[tuple[float, float]] = []

    def log_incumbent(model, where):
        if where == GRB.Callback.MIPSOL:
            incumbents.append((model.cbGet(GRB.Callback.RUNTIME), model.cbGet(GRB.Callback.MIPSOL_OBJ)))

    solver.model.optimize(log_incumbent)
    return [
        {"set": set_number, "seconds": seconds, "z": z, "final_z": solver.model.ObjVal}
        for seconds, z in incumbents
    ]


def summarize(heuristic: pd.DataFrame, incumbents: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for instance, ceiling in TIME_LIMITS.items():
        initial_beats, reaches, total = protocol_runs_reaching_gurobi(instance)
        row = {
            "instance": instance,
            "time_limit": ceiling,
            "protocol_runs": total,
            "runs_initial_population_beats_gurobi": initial_beats,
            "runs_reaching_gurobi": reaches,
        }

        per_set = heuristic[heuristic["instance"] == instance].groupby("set")["initial_population_seconds"].mean()
        if initial_beats == total:
            row["heuristic_ttt_mean"] = per_set.mean()
            row["heuristic_ttt_min_set"] = per_set.min()
            row["heuristic_ttt_max_set"] = per_set.max()

        gurobi_times = []
        for set_number in range(1, N_SETS + 1):
            target = statistics.mean(heuristic_final_z(instance, set_number))
            if instance == "pequeno":
                set_incumbents = incumbents[incumbents["set"] == set_number]
                hits = set_incumbents[set_incumbents["z"] <= target]["seconds"]
                gurobi_times.append(hits.min() if not hits.empty else None)
            else:
                gurobi_times.append(None if gurobi_final_z(instance, set_number) > target else 0.0)
        reached = [t for t in gurobi_times if t is not None]
        row["gurobi_sets_reaching_heuristic_mean"] = len(reached)
        if instance == "pequeno" and reached:
            row["gurobi_ttt_mean"] = statistics.mean(reached)
            row["gurobi_ttt_min_set"] = min(reached)
            row["gurobi_ttt_max_set"] = max(reached)
        if instance == "pequeno":
            finals = incumbents.groupby("set")["final_z"].first()
            row["gurobi_rerun_sets_same_final_z"] = sum(
                abs(finals[s] - gurobi_final_z(instance, s)) < 1e-6 for s in finals.index
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    jobs = [(instance, s, seed) for instance in reversed(TIME_LIMITS) for s in range(1, N_SETS + 1) for seed in SEEDS]
    with Pool(HEURISTIC_WORKERS) as pool:
        heuristic = pd.DataFrame(pool.map(time_initial_population, jobs))
    heuristic.to_csv(os.path.join(RESULTS_DIR, "heuristic_initial_population.csv"), index=False)
    print("População inicial da heurística medida")

    with Pool(GUROBI_WORKERS) as pool:
        incumbents = pd.DataFrame([row for rows in pool.map(gurobi_incumbents, range(1, N_SETS + 1)) for row in rows])
    incumbents.to_csv(os.path.join(RESULTS_DIR, "gurobi_pequeno_incumbents.csv"), index=False)
    print("Soluções incumbentes do Gurobi (pequeno) registradas")

    summary = summarize(heuristic, incumbents)
    summary.to_csv(os.path.join(RESULTS_DIR, "time_to_target_summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
