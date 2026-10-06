"""W_cap sweep on medio/set_01: fitness vs. capacity violation for each penalty weight.

Usage: python experiments/capacity/run_experiment.py [repetitions=3] [time_limit=60]
       python experiments/capacity/run_experiment.py plot   # only redraws the chart from the saved summary
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
from utils.solution import WEIGHTS

INSTANCE_PATH = "medio/set_01"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

W_CAP_VALUES = [200, 400, 600, 800, 1000, 1400, 1900, 2500, 3300, 5000]

DEFAULT_REPETITIONS = 3
DEFAULT_TIME_LIMIT = 60.0


def run_sweep(repetitions: int, time_limit: float) -> pd.DataFrame:
    data_manager = DataManager(INSTANCE_PATH)
    original_w_cap = WEIGHTS["W_CAP"]
    rows = []

    try:
        for w_cap in W_CAP_VALUES:
            WEIGHTS["W_CAP"] = w_cap
            print(f"W_CAP={w_cap}: {repetitions} repetições de {time_limit:.0f}s...")
            for i in range(repetitions):
                solver = MemeticSolver(data_manager)
                result = solver.solve(time_limit=time_limit)
                solution = result.solution
                rows.append({
                    "w_cap": w_cap,
                    "repetition": i + 1,
                    "fitness": solution.fitness,
                    "objective_value": solution.objective_value,
                    "capacity_violation": solution.capacity_violation,
                    "capacity_cost": solution.capacity_cost,
                })
                print(
                    f"  > rep {i + 1}/{repetitions} | fitness={solution.fitness:.1f} "
                    f"| objetivo (sem penalidade)={solution.objective_value:.1f} "
                    f"| violação={solution.capacity_violation}"
                )
    finally:
        # restore W_CAP even on error
        WEIGHTS["W_CAP"] = original_w_cap

    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    def mean_std(series: pd.Series) -> tuple[float, float]:
        values = series.tolist()
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        return mean, std

    rows = []
    for w_cap, group in raw.groupby("w_cap"):
        fitness_mean, fitness_std = mean_std(group["fitness"])
        objective_mean, objective_std = mean_std(group["objective_value"])
        violation_mean, violation_std = mean_std(group["capacity_violation"])
        rows.append({
            "w_cap": w_cap,
            "fitness_mean": fitness_mean,
            "fitness_std": fitness_std,
            "objective_value_mean": objective_mean,
            "objective_value_std": objective_std,
            "capacity_violation_mean": violation_mean,
            "capacity_violation_std": violation_std,
        })
    return pd.DataFrame(rows).sort_values("w_cap").reset_index(drop=True)


def plot_fitness_vs_capacity_penalty(summary: pd.DataFrame) -> None:
    """Cost vs. W_cap, with mean capacity violation on a second y-axis."""
    x = range(len(summary))
    x_labels = [str(w) for w in summary["w_cap"]]

    fig, ax_fitness = plt.subplots(figsize=(9, 6))

    ax_fitness.errorbar(
        x, summary["objective_value_mean"], yerr=summary["objective_value_std"],
        color="steelblue", marker="o", linewidth=2, capsize=4,
        label=r"Função objetivo $Z$ (média ± DP)",
    )
    ax_fitness.set_xlabel(r"Peso da penalidade de capacidade ($W_{cap}$)")
    ax_fitness.set_ylabel(r"Função objetivo $Z$", color="steelblue")
    ax_fitness.tick_params(axis="y", labelcolor="steelblue")
    ax_fitness.set_xticks(list(x))
    ax_fitness.set_xticklabels(x_labels)

    ax_violation = ax_fitness.twinx()
    ax_violation.plot(
        x, summary["capacity_violation_mean"], color="firebrick", marker="D",
        linestyle="--", linewidth=2, label=r"Violação de capacidade $V_{cap}$ (média)",
    )
    ax_violation.set_ylabel(r"Violação de capacidade $V_{cap}$ (pacientes-dia)", color="firebrick")
    ax_violation.tick_params(axis="y", labelcolor="firebrick")
    ax_violation.set_ylim(bottom=0)

    current_default = WEIGHTS["W_CAP"]
    if current_default in summary["w_cap"].values:
        default_x = summary.index[summary["w_cap"] == current_default][0]
        ax_fitness.axvline(default_x, color="gray", linestyle=":", linewidth=1.2, zorder=0)
        ax_fitness.text(
            default_x, ax_fitness.get_ylim()[1], f" valor adotado ({current_default})",
            rotation=90, va="top", ha="left", fontsize=8, color="gray",
        )

    lines_1, labels_1 = ax_fitness.get_legend_handles_labels()
    lines_2, labels_2 = ax_violation.get_legend_handles_labels()
    ax_fitness.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", bbox_to_anchor=(0.12, 1.0))

    ax_fitness.set_title(f"Calibração do peso de capacidade — instância {INSTANCE_PATH}")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fitness_vs_capacity_penalty.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot_fitness_vs_capacity_penalty(pd.read_csv(os.path.join(RESULTS_DIR, "capacity_sweep_summary.csv")))
        return

    repetitions = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REPETITIONS
    time_limit = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TIME_LIMIT

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 72)
    print("=== EXPERIMENTO: PENALIDADE DE CAPACIDADE (W_cap) VS. FITNESS ===")
    print(f"Instância:    {INSTANCE_PATH}")
    print(f"Valores W_cap: {W_CAP_VALUES}")
    print(f"Repetições:   {repetitions} por valor")
    print(f"Tempo limite: {time_limit} s por rodada")
    print("=" * 72)
    print()

    raw = run_sweep(repetitions, time_limit)
    raw.to_csv(os.path.join(RESULTS_DIR, "capacity_sweep_raw.csv"), index=False)

    summary = summarize(raw)
    summary.to_csv(os.path.join(RESULTS_DIR, "capacity_sweep_summary.csv"), index=False)

    plot_fitness_vs_capacity_penalty(summary)

    print()
    print(f"Resultados em {RESULTS_DIR}/")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
