"""A/B test (NOT a production change): does an escalating, infrequent "big shake" —
replace a growing fraction of the non-elite population with brand-new random
individuals once stagnation persists well past the existing boosted-mutation trigger
— reduce capacity_violation / change fitness on "grande/set_01", versus the current
MemeticSolver unmodified?

ShakeMemeticSolver is a LOCAL subclass defined only in this file, overriding solve()
with the shake inserted (everything else — __init__, _refine, _track_best_feasible —
inherited unchanged from MemeticSolver). src/solvers/heuristics/memetic_solver.py is
NOT touched. This is a side-by-side comparison to decide whether the shake is worth
promoting into the real solver — see the chat analysis for the design rationale
(escalation thresholds, why a stagnation-triggered shake should avoid the pitfall that
made the already-existing-but-disabled big_mutation_probability (constant rate) hurt
more than help).

Shake design:
  - Reuses stale_generations (already tracked). Once stale_generations >= patience +
    shake_interval, shake level 1 fires ONCE: a fraction of the non-elite population
    is replaced with create_random_individual() (fresh random individuals, same
    constructor already used for the existing "immigrants" mechanism). Level 2 fires
    at patience + 2*shake_interval, etc. — escalating, not repeating every generation.
  - Resets to level 0 whenever stale_generations itself resets (a real improvement
    was found) — same reset condition as the existing boosted-mutation trigger.
  - Fraction replaced grows with level: SHAKE_FRACTIONS = (0.5, 0.7, 0.9), capped at
    the last value for any further escalation.

Usage:
    python experiments/shake/run_experiment.py [repetitions] [time_limit]

    repetitions   independent runs per condition (baseline, shake) (default: 3)
    time_limit    seconds per run — same budget main.py uses by default (300.0),
                  for an apples-to-apples comparison against results/grande/set_01/
                  already on file
"""

from __future__ import annotations

import os
import random
import statistics
import sys
import time

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
from utils.solution import Solution, SolverResult

INSTANCE_PATH = "grande/set_01"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

DEFAULT_REPETITIONS = 3
DEFAULT_TIME_LIMIT = 300.0

SHAKE_INTERVAL = 2
"""Calibrated for grande/set_01, NOT a general-purpose default: a smoke test found
this instance only completes ~11-15 total generations in a 300s run (population_size
30 + local search scaled to 1600 patients makes each generation expensive) — the
originally-proposed shake_interval=10 (patience=5 + shake_interval=10 => first shake
at 15 stale generations) NEVER fired at all in that budget, since 15 stale generations
alone would exceed the whole run's generation count. With shake_interval=2 (first
shake at 7 stale generations), it does fire within the available budget — e.g. one
test run fired at generation 9 and the population found a markedly better solution
immediately after (145000 -> 134125, see chat analysis). Any real adoption of this
mechanism would need shake_interval (and maybe patience) scaled per instance size,
the same way local_search.py's EVALUATIONS_PER_PATIENT already scales its own cap —
a fixed constant across instance sizes doesn't transfer."""
SHAKE_FRACTIONS = (0.5, 0.7, 0.9)


class ShakeMemeticSolver(MemeticSolver):
    """MemeticSolver + an escalating population shake on persistent stagnation. See
    module docstring for the design. Only solve() is overridden; __init__, _refine,
    _track_best_feasible are inherited as-is."""

    def __init__(self, *args, shake_interval: int = SHAKE_INTERVAL, shake_fractions=SHAKE_FRACTIONS, **kwargs):
        super().__init__(*args, **kwargs)
        self.shake_interval = shake_interval
        self.shake_fractions = shake_fractions
        self.shake_events: list[tuple[int, int, float]] = []  # (generation, level, fraction)

    def solve(self, time_limit: float) -> SolverResult:
        start = time.perf_counter()
        deadline = start + time_limit
        population_size = self.genetic.population_size

        population = self.genetic.create_initial_population()
        population.sort(key=Solution.sort_key)

        best = population[0].copy()
        initial_fitness = best.fitness
        convergence_history = [best.fitness]
        local_search_evaluations = 0
        generations = 0

        best_feasible: Solution | None = None
        best_feasible_generation: int | None = None
        best_feasible, best_feasible_generation = self._track_best_feasible(
            population, 0, best_feasible, best_feasible_generation
        )

        elite_count = max(1, round(population_size * self.elite_fraction))
        stale_generations = 0
        shake_level_fired = 0
        self.shake_events = []

        while time.perf_counter() < deadline:
            generations += 1

            offspring = [individual.copy() for individual in population[:elite_count]]
            non_elite_indices = list(range(elite_count, population_size))

            is_boosted = stale_generations >= self.patience

            while len(offspring) < population_size:
                parent_a = self.genetic.tournament_select(population)
                parent_b = self.genetic.tournament_select(population)
                child = self.genetic.crossover(parent_a, parent_b)
                if is_boosted:
                    self.genetic.mutate_fraction(child, self.boosted_mutation_fraction)
                else:
                    self.genetic.mutate(child, rate=self.mutation_rate)
                offspring.append(child)

            n_immigrants = min(len(non_elite_indices), round(len(offspring) * self.immigrant_fraction))
            for index in self._rng.sample(non_elite_indices, n_immigrants):
                offspring[index] = self.genetic.create_random_individual()

            # --- escalating shake on persistent stagnation ---
            current_shake_level = 0
            if stale_generations >= self.patience:
                current_shake_level = (stale_generations - self.patience) // self.shake_interval
            if current_shake_level > shake_level_fired and current_shake_level >= 1:
                fraction = self.shake_fractions[min(current_shake_level - 1, len(self.shake_fractions) - 1)]
                n_shake = round(len(non_elite_indices) * fraction)
                shake_indices = self._rng.sample(non_elite_indices, min(n_shake, len(non_elite_indices)))
                for index in shake_indices:
                    offspring[index] = self.genetic.create_random_individual()
                shake_level_fired = current_shake_level
                self.shake_events.append((generations, current_shake_level, fraction))
            # --- end shake ---

            offspring, local_search_evaluations = self._refine(
                offspring, elite_count, non_elite_indices, deadline, local_search_evaluations
            )
            offspring.sort(key=Solution.sort_key)
            population = offspring

            best_feasible, best_feasible_generation = self._track_best_feasible(
                population, generations, best_feasible, best_feasible_generation
            )

            if population[0].sort_key() < best.sort_key():
                best = population[0].copy()
                stale_generations = 0
                shake_level_fired = 0
            else:
                stale_generations += 1
            convergence_history.append(best.fitness)

        runtime = time.perf_counter() - start
        evaluation_count = generations * population_size + local_search_evaluations

        return SolverResult(
            solution=best,
            runtime=runtime,
            initial_fitness=initial_fitness,
            evaluation_count=evaluation_count,
            convergence_history=convergence_history,
            best_feasible_fitness=best_feasible.fitness if best_feasible is not None else None,
            best_feasible_generation=best_feasible_generation,
        )


def run_condition(
    solver_cls, label: str, data_manager: DataManager, repetitions: int, time_limit: float, **solver_kwargs
) -> list[dict]:
    rows = []
    for i in range(repetitions):
        solver = solver_cls(data_manager, **solver_kwargs)
        result = solver.solve(time_limit=time_limit)
        solution = result.solution
        shake_events = getattr(solver, "shake_events", [])
        rows.append({
            "condition": label,
            "repetition": i + 1,
            "fitness": solution.fitness,
            "objective_value": solution.objective_value,
            "capacity_violation": solution.capacity_violation,
            "evaluation_count": result.evaluation_count,
            "best_feasible_fitness": result.best_feasible_fitness,
            "n_shakes": len(shake_events),
            "shake_events": str(shake_events),
        })
        print(
            f"[{label}] rep {i + 1}/{repetitions} | fitness={solution.fitness:.1f} "
            f"| objetivo (sem penalidade)={solution.objective_value:.1f} "
            f"| violação={solution.capacity_violation} | shakes disparados={len(shake_events)}"
        )
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    def mean_std(series: pd.Series) -> tuple[float, float]:
        values = series.tolist()
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        return mean, std

    rows = []
    for condition, group in raw.groupby("condition"):
        fitness_mean, fitness_std = mean_std(group["fitness"])
        objective_mean, objective_std = mean_std(group["objective_value"])
        violation_mean, violation_std = mean_std(group["capacity_violation"])
        rows.append({
            "condition": condition,
            "fitness_mean": fitness_mean,
            "fitness_std": fitness_std,
            "objective_value_mean": objective_mean,
            "objective_value_std": objective_std,
            "capacity_violation_mean": violation_mean,
            "capacity_violation_std": violation_std,
            "n_shakes_mean": mean_std(group["n_shakes"])[0],
        })
    return pd.DataFrame(rows)


def plot_comparison(summary: pd.DataFrame, instance_path: str, file_slug: str) -> None:
    conditions = summary["condition"].tolist()
    x = range(len(conditions))

    fig, (ax_obj, ax_viol) = plt.subplots(1, 2, figsize=(11, 5))

    ax_obj.bar(x, summary["objective_value_mean"], yerr=summary["objective_value_std"],
               color=["steelblue", "darkorange"], capsize=5)
    ax_obj.set_xticks(list(x))
    ax_obj.set_xticklabels(conditions)
    ax_obj.set_ylabel("Fitness sem penalidade de capacidade (objective_value)")
    ax_obj.set_title("Qualidade da solução")

    ax_viol.bar(x, summary["capacity_violation_mean"], yerr=summary["capacity_violation_std"],
                color=["steelblue", "darkorange"], capsize=5)
    ax_viol.set_xticks(list(x))
    ax_viol.set_xticklabels(conditions)
    ax_viol.set_ylabel("Violação de capacidade (leitos-dia, média)")
    ax_viol.set_title("Viabilidade")

    fig.suptitle(f"Baseline vs. shake escalonado — instância {instance_path}")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"shake_ab_comparison_{file_slug}.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    repetitions = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REPETITIONS
    time_limit = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TIME_LIMIT
    instance_path = sys.argv[3] if len(sys.argv) > 3 else INSTANCE_PATH
    shake_interval = int(sys.argv[4]) if len(sys.argv) > 4 else SHAKE_INTERVAL
    file_slug = instance_path.replace("/", "_")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    data_manager = DataManager(instance_path)

    print("=" * 72)
    print("=== A/B: SHAKE ESCALONADO POR ESTAGNAÇÃO VS. BASELINE ===")
    print(f"Instância:    {instance_path}")
    print(f"Repetições:   {repetitions} por condição")
    print(f"Tempo limite: {time_limit} s por rodada")
    print(f"shake_interval={shake_interval}, shake_fractions={SHAKE_FRACTIONS}")
    print("=" * 72)
    print()

    rows = []
    rows += run_condition(MemeticSolver, "baseline", data_manager, repetitions, time_limit)
    rows += run_condition(
        ShakeMemeticSolver, "shake", data_manager, repetitions, time_limit, shake_interval=shake_interval
    )

    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(RESULTS_DIR, f"shake_ab_raw_{file_slug}.csv"), index=False)

    summary = summarize(raw)
    summary.to_csv(os.path.join(RESULTS_DIR, f"shake_ab_summary_{file_slug}.csv"), index=False)
    plot_comparison(summary, instance_path, file_slug)

    print()
    print(f"Resultados em {RESULTS_DIR}/")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
