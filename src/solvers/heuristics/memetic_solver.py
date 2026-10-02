"""Orchestrates the memetic algorithm (GA + VNS local search) for the PBA, per Section
2.7/3.4 of the TCC: the GA explores globally via its population, while VNS refines
individuals into local optima, applied Lamarckian-style to a sampled fraction of each
generation's offspring — the refined genome is what the next generation's crossover
sees. This is the standard memetic-algorithm pattern (local search once per GA
iteration, on a subset of the population), viable to run every generation now that
local_search.py's descent uses don't-look bits instead of naive full sweeps (see its
module docstring) instead of starving the GA of generations.

The GA loop's only stopping condition is the time budget (`time_limit` passed to
solve()) — it does NOT stop early on stagnation. `patience` instead governs a mutation
trigger, not termination: a population that looks stuck keeps running and keeps
getting periodic diversity injections for as long as time allows, rather than giving up
once `patience` generations pass without improvement.

Once `stale_generations >= patience`, offspring generation switches once from the usual
per-patient `mutation_rate` (GeneticAlgorithm.mutate) to a guaranteed
`boosted_mutation_fraction` of patients (GeneticAlgorithm.mutate_fraction) — a step, not
a ramp — until the next improvement resets `stale_generations` back to baseline, which
also un-triggers the boost. This is "triggered" mutation (increase mutation pressure
once the population looks stuck, drop it back down once it's improving again), and
deliberately switches from a probability to a guaranteed fraction rather than just
raising the rate: a per-patient coin flip can, purely by chance, touch far fewer
patients than intended, which defeats the point of a boost meant to actually escape a
stuck population. Since the loop never stops on stagnation, this can trigger and
un-trigger repeatedly over a single run — every time the population stalls again after
an improvement, it gets another boost.
"""

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

        # Tracked separately from `best`: under COMPARISON_MODE="fitness_only"
        # (utils/solution.py), sort_key() is plain fitness, so `best` can end up
        # capacity-infeasible (it won by total cost, not by being viable). This instead
        # remembers the best individual with capacity_violation == 0 ever seen in any
        # generation, even if the search later moved on to a cheaper-but-infeasible one.
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

            # Triggered mutation (see module docstring): stays at the normal
            # per-patient mutation_rate until stale_generations reaches patience, then
            # switches once to a guaranteed boosted_mutation_fraction of patients and
            # stays there until the next improvement resets stale_generations back
            # down. patience no longer stops the loop — it can trigger and
            # re-trigger as many times as the time budget allows.
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

            # Random immigrants: replace a few non-elite offspring with brand-new
            # random individuals each generation. Crossover alone tends to converge
            # the population toward a shrinking set of building blocks over many
            # generations; injecting fresh, unrelated individuals keeps real
            # diversity in circulation instead of only recombining what's already
            # there.
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
        """Scans a generation's population for capacity-feasible individuals
        (capacity_violation == 0) and keeps the lowest-fitness one seen so far, plus
        the generation it first appeared in. Called on the initial population
        (generation 0) and after every generation thereafter."""
        for individual in population:
            if individual.capacity_violation == 0 and (
                best_feasible is None or individual.fitness < best_feasible.fitness
            ):
                best_feasible = individual.copy()
                best_feasible_generation = generation
        return best_feasible, best_feasible_generation

    def _refine(self, offspring, elite_count, non_elite_indices, deadline, local_search_evaluations):
        """Refines a sampled fraction of non-elite offspring, plus the elites too
        (refine_elites=True by default since 2026-10-01) via local search, overwriting
        each in place — the refined genome is what next generation's crossover will
        see. Each individual's time budget is an equal share of whatever time remains
        before `deadline`, recomputed as the queue shrinks.

        refine_elites defaulted to False until an A/B test (5 reps, 300s, pequeno/
        medio/grande) found True won on every metric at every size — objective_value
        improved 2%/23%/45% respectively, capacity_violation dropped to ~0 on
        medio/grande — with no instance where it was worse. Without it, an elite is
        completely frozen (never re-evaluated by local search) for as long as it keeps
        winning its slot, which can be many generations even when generations are
        abundant (pequeno/medio): crossover/mutation are blind (never check for
        improvement, see GeneticAlgorithm.mutate), so nothing else in the loop can
        polish that individual further — only local search can, and elites were the
        one group it never touched. The win grows with instance size because fewer
        total generations (large instances: ~12-15 in 300s, see local_search.py's
        EVALUATIONS_PER_PATIENT) means each one frozen without refinement is a bigger
        fraction of the whole run wasted."""
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
