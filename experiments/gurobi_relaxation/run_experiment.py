"""Experiment: can the LP relaxation (GurobiSolver(relaxation=True)) give a cheap lower
bound on the larger instances?

Usage: python experiments/gurobi_relaxation/run_experiment.py [time_limit=300] [instances...]
"""

from __future__ import annotations

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pandas as pd
from gurobipy import GRB

from solvers.exact.gurobi import GurobiSolver
from utils.data_manager import DataManager

_STATUS_NAMES = {
    GRB.OPTIMAL: "OPTIMAL",
    GRB.TIME_LIMIT: "TIME_LIMIT",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.UNBOUNDED: "UNBOUNDED",
}

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")

DEFAULT_TIME_LIMIT = 300.0
DEFAULT_INSTANCES = ["grande/set_01", "muito_grande/set_01"]


def _existing_mip_lower_bound(instance_path: str) -> float | None:
    """Lower bound from the saved MIP run (results/<instance>/gurobi_run.csv), for reference."""
    path = os.path.join(RESULTS_ROOT, instance_path, "gurobi_run.csv")
    if not os.path.exists(path):
        return None
    row = pd.read_csv(path).iloc[0]
    value = row.get("lower_bound")
    return None if pd.isna(value) else float(value)


def run_relaxation(instance_path: str, time_limit: float) -> dict:
    print(f"=== {instance_path} ===")
    data_manager = DataManager(instance_path)

    solver = GurobiSolver(data_manager, time_limit=time_limit, relaxation=True, verbose=False)

    build_start = time.perf_counter()
    solver.build_model()
    build_time = time.perf_counter() - build_start
    # update() so NumConstrs counts the pending constraints
    solver.model.update()
    n_vars = solver.model.NumVars
    n_constrs = solver.model.NumConstrs
    print(f"  build_model: {build_time:.2f}s ({n_vars} variáveis, {n_constrs} restrições)")

    solver.solve()
    status_name = _STATUS_NAMES.get(solver.status, str(solver.status))
    relaxation_bound = solver.model.ObjVal if solver.model.SolCount > 0 else None
    print(
        f"  solve: {solver.runtime:.2f}s | status={status_name} "
        f"| bound (ObjVal)={relaxation_bound}"
    )

    mip_bound = _existing_mip_lower_bound(instance_path)
    print(f"  lower_bound do MIP já salvo (results/{instance_path}/gurobi_run.csv): {mip_bound}")
    print()

    return {
        "instance": instance_path,
        "n_vars": n_vars,
        "n_constrs": n_constrs,
        "build_time_seconds": build_time,
        "solve_time_seconds": solver.runtime,
        "status": status_name,
        "relaxation_bound": relaxation_bound,
        "existing_mip_lower_bound": mip_bound,
    }


def main() -> None:
    time_limit = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TIME_LIMIT
    instances = sys.argv[2:] if len(sys.argv) > 2 else DEFAULT_INSTANCES

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 72)
    print("=== EXPERIMENTO: RELAXAÇÃO LINEAR DO GUROBI COMO LOWER BOUND ===")
    print(f"Instâncias:   {instances}")
    print(f"Tempo limite: {time_limit} s por instância")
    print("=" * 72)
    print()

    rows = [run_relaxation(instance_path, time_limit) for instance_path in instances]

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "relaxation_results.csv"), index=False)
    print(f"Resultados em {RESULTS_DIR}/relaxation_results.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
