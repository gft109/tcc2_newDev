"""A/B test: calibrate local_search_fraction per instance to reach target_generations,
vs. the fixed default.

Usage: python experiments/generations/run_experiment.py [repetitions=5] [time_limit=300]
       [instance_path=grande/set_01] [target_generations=25]
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
    """MemeticSolver that sets local_search_fraction from a timing probe at the start of solve()."""

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
        """Times one discarded local-search call and derives the largest fraction (capped at
        the default) that still fits target_generations in the time limit."""
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

        # spend a small slice of the budget measuring local-search cost
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
