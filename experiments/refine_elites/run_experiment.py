"""A/B test: also apply local search to the elites (refine_elites=True) vs. not.

Usage: python experiments/refine_elites/run_experiment.py [repetitions=5] [time_limit=300]
       [instance_path=grande/set_01]
"""

from __future__ import annotations

import os
import statistics
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from solvers.heuristics.memetic_solver import MemeticSolver
from utils.data_manager import DataManager

INSTANCE_PATH = "grande/set_01"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

DEFAULT_REPETITIONS = 5
DEFAULT_TIME_LIMIT = 300.0


def run_condition(
    refine_elites: bool, label: str, data_manager: DataManager, repetitions: int, time_limit: float
) -> list[dict]:
    rows = []
    for i in range(repetitions):
        solver = MemeticSolver(data_manager, refine_elites=refine_elites)
        result = solver.solve(time_limit=time_limit)
        solution = result.solution
        generations = len(result.convergence_history) - 1
        rows.append({
            "condition": label,
            "repetition": i + 1,
            "fitness": solution.fitness,
            "objective_value": solution.objective_value,
            "capacity_violation": solution.capacity_violation,
            "generations": generations,
            "evaluation_count": result.evaluation_count,
            "best_feasible_fitness": result.best_feasible_fitness,
        })
        print(
            f"[{label}] rep {i + 1}/{repetitions} | gerações={generations} "
            f"| fitness={solution.fitness:.1f} | objetivo (sem penalidade)={solution.objective_value:.1f} "
            f"| violação={solution.capacity_violation}"
        )
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    def mean_std(series: pd.Series) -> tuple[float, float]:
        values = series.dropna().tolist()
        if not values:
            return float("nan"), float("nan")
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        return mean, std

    rows = []
    for condition, group in raw.groupby("condition"):
        fitness_mean, fitness_std = mean_std(group["fitness"])
        objective_mean, objective_std = mean_std(group["objective_value"])
        violation_mean, violation_std = mean_std(group["capacity_violation"])
        generations_mean, generations_std = mean_std(group["generations"])
        rows.append({
            "condition": condition,
            "generations_mean": generations_mean,
            "generations_std": generations_std,
            "fitness_mean": fitness_mean,
            "fitness_std": fitness_std,
            "objective_value_mean": objective_mean,
            "objective_value_std": objective_std,
            "capacity_violation_mean": violation_mean,
            "capacity_violation_std": violation_std,
        })
    return pd.DataFrame(rows)


def plot_comparison(summary: pd.DataFrame, instance_path: str, file_slug: str) -> None:
    conditions = summary["condition"].tolist()
    x = range(len(conditions))
    colors = ["steelblue", "darkorange"]

    fig, (ax_obj, ax_viol) = plt.subplots(1, 2, figsize=(11, 5))

    ax_obj.bar(x, summary["objective_value_mean"], yerr=summary["objective_value_std"], color=colors, capsize=5)
    ax_obj.set_xticks(list(x))
    ax_obj.set_xticklabels(conditions)
    ax_obj.set_ylabel("Fitness sem penalidade de capacidade (objective_value)")
    ax_obj.set_title("Qualidade da solução")

    ax_viol.bar(x, summary["capacity_violation_mean"], yerr=summary["capacity_violation_std"], color=colors, capsize=5)
    ax_viol.set_xticks(list(x))
    ax_viol.set_xticklabels(conditions)
    ax_viol.set_ylabel("Violação de capacidade (leitos-dia, média)")
    ax_viol.set_title("Viabilidade")

    fig.suptitle(f"refine_elites=False vs. True — instância {instance_path}")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"refine_elites_ab_comparison_{file_slug}.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    repetitions = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REPETITIONS
    time_limit = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TIME_LIMIT
    instance_path = sys.argv[3] if len(sys.argv) > 3 else INSTANCE_PATH
    file_slug = instance_path.replace("/", "_")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    data_manager = DataManager(instance_path)

    print("=" * 72)
    print("=== A/B: refine_elites=False (padrão) VS. True ===")
    print(f"Instância:    {instance_path}")
    print(f"Repetições:   {repetitions} por condição")
    print(f"Tempo limite: {time_limit} s por rodada")
    print("=" * 72)
    print()

    rows = []
    rows += run_condition(False, "padrao", data_manager, repetitions, time_limit)
    rows += run_condition(True, "refine_elites", data_manager, repetitions, time_limit)

    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(RESULTS_DIR, f"refine_elites_ab_raw_{file_slug}.csv"), index=False)

    summary = summarize(raw)
    summary.to_csv(os.path.join(RESULTS_DIR, f"refine_elites_ab_summary_{file_slug}.csv"), index=False)
    plot_comparison(summary, instance_path, file_slug)

    print()
    print(f"Resultados em {RESULTS_DIR}/")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
