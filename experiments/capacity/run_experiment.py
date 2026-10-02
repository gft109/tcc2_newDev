"""Capacity-penalty sweep: how does W_CAP (the heuristic's soft capacity-penalty
weight — see CLAUDE.md design decision #1, capacity is a hard constraint for Gurobi
but soft for the heuristic) trade off fitness against solution quality, measured as
the number of capacity constraints still violated (capacity_violation)?

Runs the memetic heuristic on a single fixed instance (medio/set_01) across a range of
W_CAP values, several repetitions each, and plots cost vs. W_CAP (mean ± std,
capacity_violation on a second axis). Gurobi is intentionally not part of this
experiment: it doesn't use W_CAP at all (capacity is always a hard constraint there),
so its result wouldn't change across the sweep.

The chart plots `objective_value` (specialty + transfer + gender cost), NOT raw
`fitness` — fitness = objective_value + W_CAP * capacity_violation, so raw fitness at
W_CAP=5000 and W_CAP=200 aren't on a comparable scale even for the same underlying
solution quality: the penalty term itself dwarfs everything else once W_CAP gets
large, which would make the chart mostly show "how big is W_CAP" rather than "how
good is the solution". Subtracting the penalty back out (objective_value = fitness -
capacity_cost, where capacity_cost = W_CAP * capacity_violation) puts every point on
the same footing. Both `fitness` and `objective_value` are still saved to the CSVs —
only the plot is restricted to the comparable one.

Standalone and self-contained under experiments/ — does not import from or get
imported by main.py, and writes only under experiments/capacity/results/, never under
results/<instancia>/. The one shared piece of state it touches is
utils.solution.WEIGHTS["W_CAP"], mutated only for the duration of this script's own
process (restored before exit) — main.py running separately is unaffected, since it
reads WEIGHTS fresh in whatever process it's running in.

Usage:
    python experiments/capacity/run_experiment.py [repetitions] [time_limit]

    repetitions   independent heuristic runs per W_CAP value (default: 3)
    time_limit    seconds per run, same meaning as main.py's (default: 60.0)
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
"""Sweep points. Includes 0 (capacity effectively unpenalized) and the current
project default (1000, see CLAUDE.md's weights section) so it's directly visible
where "today's setting" sits on the curve."""

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
        # Restored even on error/interrupt — nothing about this experiment should
        # leak into whatever else imports utils.solution.WEIGHTS afterward, in this
        # same process (e.g. re-running this script's functions from a REPL).
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
    """The chart asked for: cost vs. W_CAP, with capacity_violation (mean) on a
    second y-axis so the trade-off this experiment is about — solution quality vs.
    actually satisfying capacity — is visible in one plot rather than two separate
    ones. Plots objective_value (mean ± std), NOT raw fitness — see the module
    docstring for why raw fitness isn't comparable across different W_CAP values."""
    x = range(len(summary))
    x_labels = [str(w) for w in summary["w_cap"]]

    fig, ax_fitness = plt.subplots(figsize=(9, 6))

    ax_fitness.errorbar(
        x, summary["objective_value_mean"], yerr=summary["objective_value_std"],
        color="steelblue", marker="o", linewidth=2, capsize=4,
        label="Fitness sem penalidade de capacidade (média ± DP)",
    )
    ax_fitness.set_xlabel("Peso da penalidade de capacidade (W_cap)")
    ax_fitness.set_ylabel(
        "Fitness sem penalidade de capacidade (especialidade + transferência + gênero)",
        color="steelblue",
    )
    ax_fitness.tick_params(axis="y", labelcolor="steelblue")
    ax_fitness.set_xticks(list(x))
    ax_fitness.set_xticklabels(x_labels)

    ax_violation = ax_fitness.twinx()
    ax_violation.plot(
        x, summary["capacity_violation_mean"], color="firebrick", marker="D",
        linestyle="--", linewidth=2, label="Violação de capacidade (média)",
    )
    ax_violation.set_ylabel("Leitos-dia de capacidade excedida (média)", color="firebrick")
    ax_violation.tick_params(axis="y", labelcolor="firebrick")
    ax_violation.set_ylim(bottom=0)

    current_default = WEIGHTS["W_CAP"]
    if current_default in summary["w_cap"].values:
        default_x = summary.index[summary["w_cap"] == current_default][0]
        ax_fitness.axvline(default_x, color="gray", linestyle=":", linewidth=1.2, zorder=0)
        ax_fitness.text(
            default_x, ax_fitness.get_ylim()[1], f" W_cap atual ({current_default})",
            rotation=90, va="top", ha="left", fontsize=8, color="gray",
        )

    lines_1, labels_1 = ax_fitness.get_legend_handles_labels()
    lines_2, labels_2 = ax_violation.get_legend_handles_labels()
    ax_fitness.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    ax_fitness.set_title(
        f"Fitness (sem penalidade de capacidade) vs. peso de penalidade — instância {INSTANCE_PATH}"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fitness_vs_capacity_penalty.png"), dpi=150)
    plt.close(fig)


def main() -> None:
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
