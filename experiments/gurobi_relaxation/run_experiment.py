"""Experiment (NOT a production change): is GurobiSolver(relaxation=True) — the LP
relaxation of the exact model (all variables continuous in [0,1] instead of binary) —
a practical way to get a cheap lower bound on "grande"/"muito_grande", where the full
MIP solve barely leaves presolve in a 300s budget (see chat analysis: lower_bound
stuck at 0.0, gap=100% on "grande")?

This does NOT go through GurobiFormatter.to_solver_result() / extract_assignment():
those read var.X > 0.5 to recover an integer room assignment, which is meaningless for
a relaxed (fractional) solve — there is no feasible "solution" to extract, only a
bound (model.ObjVal, which for a solved LP has no gap — it IS the lower bound).
src/solvers/exact/gurobi.py and gurobi_formater.py are NOT touched; this script reads
GurobiSolver's existing public attributes/properties directly instead.

Reports, per instance: how long just building the model takes (build_model() is pure
Python looping, scales with instance size independent of var_type — see chat analysis
of "grande" taking ~12s to build 1.8M variables), then the relaxation solve's runtime,
status, and resulting bound, compared against the existing MIP lower_bound already on
file in results/<instance>/set_01/gurobi_run.csv (read for reference only).

Usage:
    python experiments/gurobi_relaxation/run_experiment.py [time_limit] [instances...]

    time_limit   seconds for the relaxation solve itself (default: 300.0)
    instances    space-separated instance paths (default: grande/set_01 muito_grande/set_01)
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
    """Reads lower_bound from the already-committed results/<instance>/gurobi_run.csv
    (the real MIP run), if present, purely for side-by-side reference in the report —
    does not re-run anything."""
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
    # build_model() adds constraints via addConstr() without a trailing update() of
    # its own (optimize() triggers one internally) — call it here too so NumConstrs
    # reflects the pending constraints instead of reading as 0.
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
