"""A/B test: population size scaled with instance size vs. the fixed default (30).

Usage: python experiments/scaled_population/run_experiment.py [repetitions=5] [time_limit=300]
       [instance_path=grande/set_01]
"""

from __future__ import annotations

import math
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

POPULATION_PCT = 0.30
ELITE_PCT = 0.03
REFINE_PCT = 0.09


def scaled_params(n_patients: int) -> dict:
    population_size = math.ceil(POPULATION_PCT * n_patients)
    elite_count = math.ceil(ELITE_PCT * n_patients)
    refine_count = math.ceil(REFINE_PCT * n_patients)
    return {
        "population_size": population_size,
        "elite_fraction": elite_count / population_size,
        "local_search_fraction": refine_count / population_size,
        "elite_count": elite_count,
        "refine_count": refine_count,
    }


def run_condition(
    label: str, solver_kwargs: dict, data_manager: DataManager, repetitions: int, time_limit: float
) -> list[dict]:
    rows = []
    for i in range(repetitions):
        solver = MemeticSolver(data_manager, **solver_kwargs)
        result = solver.solve(time_limit=time_limit)
        solution = result.solution
        generations = len(result.convergence_history) - 1
        rows.append({
            "condition": label,
            "repetition": i + 1,
            "population_size": solver.genetic.population_size,
            "generations": generations,
            "fitness": solution.fitness,
            "objective_value": solution.objective_value,
            "capacity_violation": solution.capacity_violation,
            "evaluation_count": result.evaluation_count,
            "best_feasible_fitness": result.best_feasible_fitness,
        })
        print(
            f"[{label}] rep {i + 1}/{repetitions} | pop={solver.genetic.population_size} "
            f"| gerações={generations} | fitness={solution.fitness:.1f} "
            f"| objetivo (sem penalidade)={solution.objective_value:.1f} "
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
            "population_size": group["population_size"].iloc[0],
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

    fig, (ax_gen, ax_obj, ax_viol) = plt.subplots(1, 3, figsize=(15, 5))

    ax_gen.bar(x, summary["generations_mean"], yerr=summary["generations_std"], color=colors, capsize=5)
    ax_gen.set_xticks(list(x))
    ax_gen.set_xticklabels(conditions)
    ax_gen.set_ylabel("Gerações completadas (média)")
    ax_gen.set_title("Gerações")

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

    fig.suptitle(f"população fixa (30) vs. escalada — instância {instance_path}")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"scaled_population_ab_comparison_{file_slug}.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    repetitions = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REPETITIONS
    time_limit = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TIME_LIMIT
    instance_path = sys.argv[3] if len(sys.argv) > 3 else INSTANCE_PATH
    file_slug = instance_path.replace("/", "_")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    data_manager = DataManager(instance_path)
    n_patients = len(data_manager.patients)
    scaled = scaled_params(n_patients)

    print("=" * 72)
    print("=== A/B: POPULAÇÃO FIXA (30) VS. ESCALADA COM O Nº DE PACIENTES ===")
    print(f"Instância:    {instance_path} ({n_patients} pacientes)")
    print(f"Repetições:   {repetitions} por condição")
    print(f"Tempo limite: {time_limit} s por rodada")
    print(
        f"Escalado: population_size={scaled['population_size']} "
        f"(elites={scaled['elite_count']}, refinados={scaled['refine_count']})"
    )
    print("=" * 72)
    print()

    rows = []
    rows += run_condition("fixo", {}, data_manager, repetitions, time_limit)
    rows += run_condition(
        "escalado",
        {
            "population_size": scaled["population_size"],
            "elite_fraction": scaled["elite_fraction"],
            "local_search_fraction": scaled["local_search_fraction"],
        },
        data_manager, repetitions, time_limit,
    )

    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(RESULTS_DIR, f"scaled_population_ab_raw_{file_slug}.csv"), index=False)

    summary = summarize(raw)
    summary.to_csv(os.path.join(RESULTS_DIR, f"scaled_population_ab_summary_{file_slug}.csv"), index=False)
    plot_comparison(summary, instance_path, file_slug)

    print()
    print(f"Resultados em {RESULTS_DIR}/")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
