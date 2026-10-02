"""A/B test (NOT a production change): does calibrating local_search_fraction per
instance — instead of MemeticSolver's fixed default (0.3) regardless of size — to
target at least `target_generations` completed GA generations improve outcomes on
"grande/set_01", where the fixed fraction currently yields only ~11-15 generations in
a 300s run?

Why that's the bottleneck (see chat analysis): each generation's _refine() phase runs
local_search_fraction * population_size individuals (9 of 30, by default) to near-full
VNS convergence. LocalSearch's own module docstring measures that convergence cost at
~150-190 evaluations/patient on "grande" (1600 patients) — a few seconds per
individual — so 9 of them dominate a generation's wall-clock time, leaving only
~11-15 generations total in 300s. Too few for population-level mechanisms
(crossover-driven exploration, stagnation-escape triggers) to do much.

AdaptiveRefineMemeticSolver is a LOCAL subclass defined only in this file. It measures
(calibrates) this instance's actual per-individual local-search cost with one
discarded probe call at the start of solve(), then derives the local_search_fraction
that should let the run complete target_generations generations in the given
time_limit — capped at the constructor-provided default, never raised above it (no
evidence more refinement helps instances that already exceed the target comfortably,
e.g. pequeno/medio). src/solvers/heuristics/memetic_solver.py is NOT touched; this is
a side-by-side comparison to decide whether it's worth promoting into the real solver.

Usage:
    python experiments/generations/run_experiment.py [repetitions] [time_limit] [instance_path] [target_generations]

    repetitions         independent runs per condition (baseline, adaptive) (default: 5)
    time_limit          seconds per run — same budget main.py uses by default (300.0)
    instance_path       default: grande/set_01
    target_generations  minimum generations the adaptive condition aims for (default: 25)
"""

from __future__ import annotations

import os
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

DEFAULT_REPETITIONS = 5
DEFAULT_TIME_LIMIT = 300.0
DEFAULT_TARGET_GENERATIONS = 25


class AdaptiveRefineMemeticSolver(MemeticSolver):
    """MemeticSolver that calibrates local_search_fraction at the start of solve()
    (see module docstring) instead of using a fixed value regardless of instance
    size. Only __init__ and solve() are overridden; _refine() and
    _track_best_feasible() are inherited unchanged and read self.local_search_fraction
    as usual — calibration just adjusts that attribute before the main loop starts."""

    def __init__(
        self, *args, target_generations: int = DEFAULT_TARGET_GENERATIONS, calibration_fraction: float = 0.1,
        max_calibration_time: float = 15.0, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.target_generations = target_generations
        self.calibration_fraction = calibration_fraction
        self.max_calibration_time = max_calibration_time
        self.calibrated_fraction: float | None = None
        self.calibration_cost: float | None = None

    def _calibrate_local_search_fraction(self, population: list[Solution], time_limit: float) -> float:
        """Times one discarded local-search call (on a copy of the current best, so
        the real population is untouched) to estimate this instance's per-individual
        VNS cost, then derives the largest local_search_fraction that should let the
        run complete target_generations generations in time_limit seconds — capped at
        the constructor-provided fraction, never raised above it."""
        probe = population[0].copy()
        calibration_budget = min(self.max_calibration_time, time_limit * self.calibration_fraction)

        start = time.perf_counter()
        self.local_search.run(probe, time_budget=calibration_budget)
        self.calibration_cost = time.perf_counter() - start

        if self.calibration_cost <= 0:
            return self.local_search_fraction

        generation_time_budget = time_limit / self.target_generations
        n_individuals = max(1, int(generation_time_budget / self.calibration_cost))
        computed_fraction = n_individuals / self.genetic.population_size
        return min(self.local_search_fraction, computed_fraction)

    def solve(self, time_limit: float) -> SolverResult:
        start = time.perf_counter()
        deadline = start + time_limit
        population_size = self.genetic.population_size

        population = self.genetic.create_initial_population()
        population.sort(key=Solution.sort_key)

        # Calibration: a small, bounded slice of the budget spent measuring this
        # instance's actual local-search cost, then local_search_fraction is adjusted
        # in place before the main loop — the inherited _refine() reads it as usual.
        self.calibrated_fraction = self._calibrate_local_search_fraction(population, time_limit)
        self.local_search_fraction = self.calibrated_fraction

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
        generations_completed = len(result.convergence_history) - 1
        calibrated_fraction = getattr(solver, "calibrated_fraction", None)
        calibration_cost = getattr(solver, "calibration_cost", None)
        rows.append({
            "condition": label,
            "repetition": i + 1,
            "fitness": solution.fitness,
            "objective_value": solution.objective_value,
            "capacity_violation": solution.capacity_violation,
            "generations": generations_completed,
            "evaluation_count": result.evaluation_count,
            "best_feasible_fitness": result.best_feasible_fitness,
            "calibrated_fraction": calibrated_fraction,
            "calibration_cost": calibration_cost,
        })
        extra = (
            f" | fração calibrada={calibrated_fraction:.3f} (custo medido={calibration_cost:.2f}s)"
            if calibrated_fraction is not None else ""
        )
        print(
            f"[{label}] rep {i + 1}/{repetitions} | gerações={generations_completed} "
            f"| fitness={solution.fitness:.1f} | objetivo (sem penalidade)={solution.objective_value:.1f} "
            f"| violação={solution.capacity_violation}{extra}"
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


def plot_comparison(summary: pd.DataFrame, instance_path: str, file_slug: str, target_generations: int) -> None:
    conditions = summary["condition"].tolist()
    x = range(len(conditions))
    colors = ["steelblue", "darkorange"]

    fig, (ax_gen, ax_obj, ax_viol) = plt.subplots(1, 3, figsize=(15, 5))

    ax_gen.bar(x, summary["generations_mean"], yerr=summary["generations_std"], color=colors, capsize=5)
    ax_gen.axhline(target_generations, color="gray", linestyle=":", linewidth=1.2)
    ax_gen.text(0, target_generations, f" meta: {target_generations}", va="bottom", fontsize=8, color="gray")
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

    fig.suptitle(f"Baseline vs. local_search_fraction adaptativo — instância {instance_path}")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"generations_ab_comparison_{file_slug}.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    repetitions = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REPETITIONS
    time_limit = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TIME_LIMIT
    instance_path = sys.argv[3] if len(sys.argv) > 3 else INSTANCE_PATH
    target_generations = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_TARGET_GENERATIONS
    file_slug = instance_path.replace("/", "_")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    data_manager = DataManager(instance_path)

    print("=" * 72)
    print("=== A/B: LOCAL_SEARCH_FRACTION ADAPTATIVO (META DE GERAÇÕES) VS. BASELINE ===")
    print(f"Instância:          {instance_path}")
    print(f"Repetições:         {repetitions} por condição")
    print(f"Tempo limite:       {time_limit} s por rodada")
    print(f"Meta de gerações:   {target_generations}")
    print("=" * 72)
    print()

    rows = []
    rows += run_condition(MemeticSolver, "baseline", data_manager, repetitions, time_limit)
    rows += run_condition(
        AdaptiveRefineMemeticSolver, "adaptativo", data_manager, repetitions, time_limit,
        target_generations=target_generations,
    )

    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(RESULTS_DIR, f"generations_ab_raw_{file_slug}.csv"), index=False)

    summary = summarize(raw)
    summary.to_csv(os.path.join(RESULTS_DIR, f"generations_ab_summary_{file_slug}.csv"), index=False)
    plot_comparison(summary, instance_path, file_slug, target_generations)

    print()
    print(f"Resultados em {RESULTS_DIR}/")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
