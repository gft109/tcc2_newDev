"""Memetic solver: GA loop with VND local search (Lamarckian) on part of each generation.

Runs until the time limit. After `patience` generations without improvement, mutation
switches to a fixed `boosted_mutation_fraction` until the next improvement."""

from __future__ import annotations

import random
import time

from solvers.heuristics.genetic import GeneticAlgorithm
from solvers.heuristics.local_search import LocalSearch
from utils.data_manager import DataManager
from utils.solution import Solution, SolverResult


class MemeticSolver:
    def __init__(
        self,
        data_manager: DataManager,
        population_size: int = 30,
        tournament_size: int = 3,
        crossover_rate: float = 0.9,
        room_inherit_probability: float = 0.5,
        mutation_rate: float = 0.05,
        boosted_mutation_fraction: float = 0.35,
        feasible_seed_fraction: float = 0.2,
        immigrant_fraction: float = 0.1,
        elite_fraction: float = 0.1,
        local_search_fraction: float = 0.3,
        refine_elites: bool = True,
        max_local_search_candidates: int = 10,
        max_local_search_evaluations: int | None | str = "auto",
        patience: int = 5,
        seed: int | None = None,
    ):
        self.data_manager = data_manager
        self.genetic = GeneticAlgorithm(
            data_manager,
            population_size=population_size,
            tournament_size=tournament_size,
            crossover_rate=crossover_rate,
            room_inherit_probability=room_inherit_probability,
            mutation_rate=mutation_rate,
            feasible_seed_fraction=feasible_seed_fraction,
            seed=seed,
        )
        self.local_search = LocalSearch(
            data_manager,
            max_candidates=max_local_search_candidates,
            max_evaluations=max_local_search_evaluations,
            seed=seed,
        )
        self.elite_fraction = elite_fraction
        self.local_search_fraction = local_search_fraction
        self.refine_elites = refine_elites
        self.immigrant_fraction = immigrant_fraction
        self.patience = patience
        self.mutation_rate = mutation_rate
        self.boosted_mutation_fraction = boosted_mutation_fraction
        self._rng = random.Random(seed)

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

        # `best` may violate capacity; this keeps the best capacity-feasible one seen.
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
            best_feasible_solution=best_feasible,
        )

    @staticmethod
    def _track_best_feasible(
        population: list[Solution],
        generation: int,
        best_feasible: Solution | None,
        best_feasible_generation: int | None,
    ) -> tuple[Solution | None, int | None]:
        """Keeps the lowest-fitness capacity-feasible individual seen so far and its generation."""
        for individual in population:
            if individual.capacity_violation == 0 and (
                best_feasible is None or individual.fitness < best_feasible.fitness
            ):
                best_feasible = individual.copy()
                best_feasible_generation = generation
        return best_feasible, best_feasible_generation

    def _refine(self, offspring, elite_count, non_elite_indices, deadline, local_search_evaluations):
        """Applies local search to the elites (if refine_elites) and a sampled fraction of the
        rest, splitting the remaining time evenly among them."""
        n_refine_non_elite = min(len(non_elite_indices), round(len(offspring) * self.local_search_fraction))
        refine_indices = self._rng.sample(non_elite_indices, n_refine_non_elite)
        if self.refine_elites:
            refine_indices = list(range(elite_count)) + refine_indices

        remaining_to_process = len(refine_indices)
        for index in refine_indices:
            remaining_time = deadline - time.perf_counter()
            if remaining_time <= 0:
                break
            per_individual_budget = remaining_time / remaining_to_process
            offspring[index] = self.local_search.run(offspring[index], time_budget=per_individual_budget)
            local_search_evaluations += self.local_search.last_run_evaluations
            remaining_to_process -= 1

        return offspring, local_search_evaluations
