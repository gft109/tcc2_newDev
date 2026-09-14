"""Orchestrates the memetic algorithm (GA + VNS local search) for the PBA, per Section
2.7/3.4 of the TCC: the GA explores globally via its population, while VNS refines
individuals into local optima. Returns a SolverResult in the same format used by
gurobi_formater.py, so the two solvers can be compared directly.

`local_search_mode` selects how the two are coupled — a prototyping knob to compare
against the literature (see local_search.py's module docstring for the Ceschia &
Schaerf background on why an unbounded per-call cost is dangerous inside a population
loop in the first place):

- "lamarckian" (default): local search runs on a sampled fraction of each generation's
  offspring and OVERWRITES them in place — the refined genome is what the next
  generation's crossover sees. This is the standard memetic-algorithm pattern (local
  search applied once per GA iteration to a subset of the population) and, now that
  local_search.py's descent uses don't-look bits instead of naive full sweeps (see its
  module docstring), cheap enough per call to run every generation without starving the
  GA of generations the way the old sweep-based descent did.
- "pool": Zhou et al. (MOGANS) / Sitepu et al.-style. Local search runs on COPIES of
  the sampled offspring, producing extra candidates that are only ADDED to the
  population; a single survivor selection (sort by sort_key, keep the best
  population_size) then decides who actually continues, un-refined originals included.
  This preserves un-refined genetic material for future crossover even when its
  refined counterpart wins a slot — at the same per-generation time cost as
  "lamarckian" (the coupling to generation count doesn't go away, only the
  overwrite-in-place behavior does).
- "end_of_run": no local search during the GA loop at all — every generation is pure
  crossover/mutation/selection, mirroring Belciug & Gorunescu's fixed-generation-count
  GA. A single local search pass then runs once, over the final population (always
  including its current elites), spending however much of the time budget is left when
  the GA loop stops — which is governed by `end_of_run_generations` (a hard cap,
  defaulting to the instance's own room count), not by a pre-carved time slice:
  empirically, spreading the same leftover time across more individuals than needed
  produces shallower, less-converged refinements than giving fewer individuals (elites
  + a sample) the time to reach a true local optimum, even if that leaves time on the
  table. Kept available for comparison, but no longer the default: in practice its
  GA-only loop plateaus after very few generations regardless of how much time is
  left, since nothing but crossover/mutation/random-immigrants drives improvement —
  leaving most of the time budget unused and the final population worse than
  "lamarckian"'s per-generation refinement produces.

The GA loop's only stopping condition is the time budget (`time_limit` passed to
solve(), and `end_of_run_generations` as an additional cap for "end_of_run" mode only)
— it does NOT stop early on stagnation. `patience` instead governs a mutation trigger,
not termination (see below): a population that looks stuck keeps running and keeps
getting periodic diversity injections for as long as time allows, rather than giving up
once `patience` generations pass without improvement. This was a deliberate reversal of
an earlier version where `patience` doubled as the stopping condition — that made
runs on fast-converging instances/seeds end far short of the time budget, and coupled
two conceptually different things (how long to keep trying vs. when to try harder) into
one number.

Once `stale_generations >= patience`, offspring generation switches once from the usual
per-patient `mutation_rate` (GeneticAlgorithm.mutate) to a guaranteed
`boosted_mutation_fraction` of patients (GeneticAlgorithm.mutate_fraction) — a step, not
a ramp — until the next improvement resets `stale_generations` back to baseline, which
also un-triggers the boost. This is "triggered" mutation (increase mutation pressure
once the population looks stuck, drop it back down once it's improving again), and
deliberately switches from a probability to a guaranteed fraction rather than just
raising the rate: a per-patient coin flip can, purely by chance, touch far fewer
patients than intended, which defeats the point of a boost meant to actually escape a
stuck population. Since the loop no longer stops on stagnation, this can trigger and
un-trigger repeatedly over a single run — every time the population stalls again after
an improvement, it gets another boost. Unlike `big_mutation_*` (a fixed, always-on
per-individual chance of a much larger jump, tested and found to hurt more than help at
a constant rate — see genetic.py), this only ever applies while stagnation has actually
been observed.

`capacity_weight_start`/`capacity_weight_end`/`capacity_weight_ramp_generations` exist
to support a dynamic penalty for capacity_violation (Joines & Houck, 1994's classic GA
constraint-handling technique: penalty grows with generation number, so early
generations explore more freely through mildly-infeasible territory and later ones
increasingly enforce feasibility) — `solve()` ramps the shared `WEIGHTS["W_CAP"]`
linearly from `capacity_weight_start` to `capacity_weight_end` over
`capacity_weight_ramp_generations` generations, then holds at the ceiling. This is safe
to do mid-run — unlike W_TRANSF/W_GEN/W_SPEC, which Solution bakes into incremental
accumulators the moment a mutation is applied (so changing them mid-run would desync a
solution's accumulated cost between old- and new-weight contributions), W_CAP is only
ever read live via the `capacity_cost` property, so every comparison always reflects
the current weight with no possibility of drift. The defaults (`start == end == 400`)
make this a no-op — a flat W_CAP=400 throughout, matching the exact model's weights —
which is back to the original static behavior after A/B testing found the dynamic
ramp's generation-count-based horizon didn't scale well across instance sizes (good on
medio, which completes enough generations for the ramp to finish; worse than static on
grande, which doesn't). Raise `capacity_weight_end` above `capacity_weight_start` to
re-enable the ramp."""

from __future__ import annotations

import random
import time

from solvers.heuristics.genetic import GeneticAlgorithm
from solvers.heuristics.local_search import LocalSearch
from utils.data_manager import DataManager
from utils.solution import WEIGHTS, Solution, SolverResult


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
        big_mutation_probability: float = 0.0,
        big_mutation_fraction: float = 0.25,
        immigrant_fraction: float = 0.1,
        elite_fraction: float = 0.1,
        local_search_fraction: float = 0.3,
        refine_elites: bool = False,
        max_local_search_candidates: int = 10,
        max_shake_level: int = 0,
        shake_fraction: float = 0.05,
        max_local_search_evaluations: int | None | str = "auto",
        local_search_mode: str = "lamarckian",
        end_of_run_fraction: float = 0.3,
        end_of_run_generations: int | None = None,
        patience: int = 5,
        capacity_weight_start: float = 400.0,
        capacity_weight_end: float = 400.0,
        capacity_weight_ramp_generations: int = 10,
        seed: int | None = None,
    ):
        if local_search_mode not in ("lamarckian", "pool", "end_of_run"):
            raise ValueError(f"unknown local_search_mode: {local_search_mode!r}")
        if end_of_run_generations is None:
            end_of_run_generations = len(data_manager.rooms)
        self.data_manager = data_manager
        self.genetic = GeneticAlgorithm(
            data_manager,
            population_size=population_size,
            tournament_size=tournament_size,
            crossover_rate=crossover_rate,
            room_inherit_probability=room_inherit_probability,
            mutation_rate=mutation_rate,
            big_mutation_probability=big_mutation_probability,
            big_mutation_fraction=big_mutation_fraction,
            seed=seed,
        )
        self.local_search = LocalSearch(
            data_manager,
            max_candidates=max_local_search_candidates,
            max_shake_level=max_shake_level,
            shake_fraction=shake_fraction,
            max_evaluations=max_local_search_evaluations,
            seed=seed,
        )
        self.elite_fraction = elite_fraction
        self.local_search_fraction = local_search_fraction
        self.refine_elites = refine_elites
        self.immigrant_fraction = immigrant_fraction
        self.local_search_mode = local_search_mode
        self.end_of_run_fraction = end_of_run_fraction
        self.end_of_run_generations = end_of_run_generations
        self.patience = patience
        self.mutation_rate = mutation_rate
        self.boosted_mutation_fraction = boosted_mutation_fraction
        self.capacity_weight_start = capacity_weight_start
        self.capacity_weight_end = capacity_weight_end
        self.capacity_weight_ramp_generations = capacity_weight_ramp_generations
        self._rng = random.Random(seed)

    def solve(self, time_limit: float) -> SolverResult:
        start = time.perf_counter()
        deadline = start + time_limit
        population_size = self.genetic.population_size

        use_fixed_generations = self.local_search_mode == "end_of_run" and self.end_of_run_generations is not None

        ga_deadline = deadline
        if self.local_search_mode == "end_of_run" and not use_fixed_generations:
            ga_deadline = start + time_limit * (1 - self.end_of_run_fraction)

        # Dynamic capacity penalty (see module docstring): starts low so early
        # generations can explore through mildly-infeasible territory, ramps up to the
        # ceiling by capacity_weight_ramp_generations. Safe to mutate the shared
        # WEIGHTS dict mid-run because capacity_cost is read live, not baked into an
        # incremental accumulator (unlike W_TRANSF/W_GEN/W_SPEC).
        WEIGHTS["W_CAP"] = self.capacity_weight_start

        population = self.genetic.create_initial_population()
        population.sort(key=Solution.sort_key)

        initial_best_individual = population[0].copy()
        best = population[0].copy()
        convergence_history = [self._reporting_fitness(best)]
        local_search_evaluations = 0
        generations = 0

        elite_count = max(1, round(population_size * self.elite_fraction))
        stale_generations = 0

        while (
            not use_fixed_generations or generations < self.end_of_run_generations
        ) and time.perf_counter() < ga_deadline:
            generations += 1
            if self.capacity_weight_ramp_generations > 0:
                ramp_progress = min(1.0, generations / self.capacity_weight_ramp_generations)
            else:
                ramp_progress = 1.0
            WEIGHTS["W_CAP"] = self.capacity_weight_start + ramp_progress * (
                self.capacity_weight_end - self.capacity_weight_start
            )

            elites = population[:elite_count]
            offspring = [individual.copy() for individual in elites]
            non_elite_indices = list(range(elite_count, population_size))

            # Triggered mutation (see module docstring): stays at the normal
            # per-patient mutation_rate until stale_generations reaches patience, then
            # switches once to a guaranteed boosted_mutation_fraction of patients and
            # stays there until the next improvement resets stale_generations back
            # down. patience no longer stops the loop (see module docstring) — it can
            # trigger and re-trigger as many times as the time budget allows.
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

            if self.local_search_mode == "lamarckian":
                offspring, local_search_evaluations = self._refine_lamarckian(
                    offspring, elite_count, non_elite_indices, ga_deadline, local_search_evaluations
                )
                offspring.sort(key=Solution.sort_key)
                population = offspring
            elif self.local_search_mode == "pool":
                population, local_search_evaluations = self._refine_pool(
                    offspring, elite_count, non_elite_indices, population_size, ga_deadline, local_search_evaluations
                )
            else:  # "end_of_run": no local search during the GA loop itself
                offspring.sort(key=Solution.sort_key)
                population = offspring

            if population[0].sort_key() < best.sort_key():
                best = population[0].copy()
                stale_generations = 0
            else:
                stale_generations += 1
            convergence_history.append(self._reporting_fitness(best))

        if self.local_search_mode == "end_of_run":
            population, local_search_evaluations = self._refine_end_of_run(
                population, elite_count, deadline, local_search_evaluations
            )
            population.sort(key=Solution.sort_key)
            if population[0].sort_key() < best.sort_key():
                best = population[0].copy()
            convergence_history.append(self._reporting_fitness(best))

        # Hold at the ceiling once the run ends, so every downstream read of
        # best.fitness/total_cost reflects one consistent, settled weight rather than
        # whatever point the ramp happened to be at.
        WEIGHTS["W_CAP"] = self.capacity_weight_end
        initial_fitness = self._reporting_fitness(initial_best_individual)

        runtime = time.perf_counter() - start
        evaluation_count = generations * population_size + local_search_evaluations

        return SolverResult(
            solution=best,
            runtime=runtime,
            initial_fitness=initial_fitness,
            evaluation_count=evaluation_count,
            convergence_history=convergence_history,
        )

    def _reporting_fitness(self, solution: Solution) -> float:
        """Cost for convergence tracking (convergence_history, initial_fitness) —
        always evaluated at capacity_weight_end, regardless of where WEIGHTS['W_CAP']
        currently sits mid-ramp. Search decisions (sort_key comparisons) correctly use
        the live, ramping weight; but if convergence_history read `best.fitness` at
        whatever weight was active that generation, the SAME best solution would show
        a rising reported cost purely because the weight grew under it — a solution
        that never got worse would look like it did. Pinning every reported point to
        the same fixed weight keeps the history a true measure of solution-quality
        progress instead of a mix of search-quality progress and the ramp's own
        movement."""
        return solution.objective_value + self.capacity_weight_end * solution.capacity_violation

    def _select_refine_indices(self, elite_count: int, non_elite_indices: list[int], pool_size: int) -> list[int]:
        """Shared sampling policy for "lamarckian"/"pool": a fraction of the non-elite
        offspring, plus the elites too if refine_elites is set."""
        n_refine_non_elite = min(len(non_elite_indices), round(pool_size * self.local_search_fraction))
        refine_indices = self._rng.sample(non_elite_indices, n_refine_non_elite)
        if self.refine_elites:
            refine_indices = list(range(elite_count)) + refine_indices
        return refine_indices

    def _refine_lamarckian(self, offspring, elite_count, non_elite_indices, deadline, local_search_evaluations):
        """Refines a sample of offspring and overwrites them in place — the refined
        genome is what next generation's crossover will see (see module docstring)."""
        refine_indices = self._select_refine_indices(elite_count, non_elite_indices, len(offspring))

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

    def _refine_pool(self, offspring, elite_count, non_elite_indices, population_size, deadline, local_search_evaluations):
        """Zhou et al. (MOGANS) / Sitepu et al.-style: local search runs on COPIES,
        producing extra candidates that only enter the next generation by winning a
        population-wide survivor selection alongside the un-refined offspring (see
        module docstring)."""
        refine_indices = self._select_refine_indices(elite_count, non_elite_indices, len(offspring))

        extra_candidates = []
        remaining_to_process = len(refine_indices)
        for index in refine_indices:
            remaining_time = deadline - time.perf_counter()
            if remaining_time <= 0:
                break
            per_individual_budget = remaining_time / remaining_to_process
            refined = self.local_search.run(offspring[index].copy(), time_budget=per_individual_budget)
            local_search_evaluations += self.local_search.last_run_evaluations
            extra_candidates.append(refined)
            remaining_to_process -= 1

        combined = offspring + extra_candidates
        combined.sort(key=Solution.sort_key)
        return combined[:population_size], local_search_evaluations

    def _refine_end_of_run(self, population, elite_count, deadline, local_search_evaluations):
        """Single local-search pass over the final population, spending whatever time is
        left before `deadline` — the entire remainder of the run's time budget, since
        the GA loop stopped by generation count/patience rather than by time. Always
        includes the current elites — there is no future generation left to preserve
        their un-refined genome for, so refine_elites doesn't gate this (see module
        docstring)."""
        non_elite_indices = list(range(elite_count, len(population)))
        n_refine_non_elite = min(len(non_elite_indices), round(len(population) * self.local_search_fraction))
        refine_indices = list(range(elite_count)) + self._rng.sample(non_elite_indices, n_refine_non_elite)

        remaining_to_process = len(refine_indices)
        for index in refine_indices:
            remaining_time = deadline - time.perf_counter()
            if remaining_time <= 0:
                break
            per_individual_budget = remaining_time / remaining_to_process
            population[index] = self.local_search.run(population[index], time_budget=per_individual_budget)
            local_search_evaluations += self.local_search.last_run_evaluations
            remaining_to_process -= 1

        return population, local_search_evaluations
