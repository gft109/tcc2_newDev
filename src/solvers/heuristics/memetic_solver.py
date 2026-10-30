"""Memetic solver: GA loop with VND local search (Lamarckian) on part of each generation.

Runs until the time limit. After `patience` generations without improvement, mutation
switches to a fixed `boosted_mutation_fraction` until the next improvement.

repair_mode (capacity repair, see utils/repair.py; its time counts within the time limit):
    "always" - every individual is repaired as soon as it is created, and again after the
               VND, so the population is always feasible (default).
    "final"  - the GA stops early enough to repair the final population and returns the
               feasible solution with the lowest Z.
    "none"   - no repair; the best solution may violate capacity."""

from __future__ import annotations

import random
import time

from solvers.heuristics.genetic import GeneticAlgorithm
from solvers.heuristics.local_search import LocalSearch
from utils.data_manager import DataManager
from utils.repair import CapacityRepair
from utils.solution import Solution, SolverResult

REPAIR_MODES = ("none", "final", "always")


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
        repair_mode: str = "always",
        seed: int | None = None,
    ):
        if repair_mode not in REPAIR_MODES:
            raise ValueError(f"repair_mode must be one of {REPAIR_MODES}, got '{repair_mode}'")
        self.data_manager = data_manager
        self.repair_mode = repair_mode
        self.repair = CapacityRepair(data_manager, max_candidates=max_local_search_candidates, seed=seed)
        self._repair_time = 0.0
        self._repair_evaluations = 0
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
        self._repair_time = 0.0
        self._repair_evaluations = 0

        population = self.genetic.create_initial_population()
        if self.repair_mode == "always":
            for individual in population:
                self._timed_repair(individual)
        population.sort(key=Solution.sort_key)

        # "final" stops the GA early by the time it took to repair the initial best, whose
        # violation is far above that of the final population.
        loop_deadline = deadline
        if self.repair_mode == "final":
            reserve_start = time.perf_counter()
            self.repair.repair(population[0].copy())
            reserve = time.perf_counter() - reserve_start
            self._repair_time += reserve
            self._repair_evaluations += self.repair.last_run_evaluations
            loop_deadline = deadline - reserve

        best = population[0].copy()
        initial_fitness = best.fitness
        convergence_history = [best.fitness]
        capacity_violation_history = [best.capacity_violation]
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
        # "always": the VND stops early by the longest post-VND repair seen, and new offspring
        # are only created if their pre-VND repair (longest seen) still fits before that;
        # otherwise the remaining time goes to the VND on the current, already feasible,
        # population. Either way the repairs of the last generation finish within the limit.
        pre_vnd_repair_reserve = 0.0
        post_vnd_repair_reserve = 0.0

        while time.perf_counter() < loop_deadline:
            generations += 1
            non_elite_indices = list(range(elite_count, population_size))
            refine_only = time.perf_counter() >= loop_deadline - pre_vnd_repair_reserve

            if refine_only:
                offspring = [individual.copy() for individual in population]
            else:
                offspring = [individual.copy() for individual in population[:elite_count]]

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

                if self.repair_mode == "always":
                    repair_time_before = self._repair_time
                    for individual in offspring[elite_count:]:
                        self._timed_repair(individual)
                    pre_vnd_repair_reserve = max(pre_vnd_repair_reserve, self._repair_time - repair_time_before)

            offspring, local_search_evaluations = self._refine(
                offspring, elite_count, non_elite_indices, loop_deadline, local_search_evaluations
            )
            if self.repair_mode == "always":
                # VND is driven by F and almost never adds violation; this keeps it guaranteed
                repair_time_before = self._repair_time
                for individual in offspring:
                    self._timed_repair(individual)
                post_vnd_repair_reserve = max(post_vnd_repair_reserve, self._repair_time - repair_time_before)
                loop_deadline = deadline - post_vnd_repair_reserve
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
            capacity_violation_history.append(best.capacity_violation)

        pre_repair_objective = None
        pre_repair_capacity_violation = None
        if self.repair_mode == "final":
            pre_repair_objective = best.objective_value
            pre_repair_capacity_violation = best.capacity_violation
            best = self._repair_final(best, population, best_feasible, deadline)

        runtime = time.perf_counter() - start
        evaluation_count = (
            generations * population_size + local_search_evaluations + self._repair_evaluations
        )

        return SolverResult(
            solution=best,
            pre_repair_objective=pre_repair_objective,
            pre_repair_capacity_violation=pre_repair_capacity_violation,
            repair_time=self._repair_time if self.repair_mode != "none" else None,
            runtime=runtime,
            initial_fitness=initial_fitness,
            evaluation_count=evaluation_count,
            convergence_history=convergence_history,
            capacity_violation_history=capacity_violation_history,
            best_feasible_fitness=best_feasible.fitness if best_feasible is not None else None,
            best_feasible_generation=best_feasible_generation,
            best_feasible_solution=best_feasible,
        )

    def _timed_repair(self, individual: Solution) -> None:
        if individual.capacity_violation == 0:
            return
        repair_start = time.perf_counter()
        self.repair.repair(individual)
        self._repair_time += time.perf_counter() - repair_start
        self._repair_evaluations += self.repair.last_run_evaluations

    def _repair_final(
        self, best: Solution, population: list[Solution], best_feasible: Solution | None, deadline: float
    ) -> Solution:
        """Repairs the overall best (always) and then the final population in F order while
        time remains; returns the feasible solution with the lowest Z, including the best
        feasible individual seen during the search."""
        repaired: list[Solution] = []
        for index, individual in enumerate([best] + population):
            if index > 0 and time.perf_counter() >= deadline:
                break
            candidate = individual.copy()
            self._timed_repair(candidate)
            repaired.append(candidate)
        if best_feasible is not None:
            repaired.append(best_feasible.copy())
        return min(repaired, key=lambda solution: solution.objective_value)

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
