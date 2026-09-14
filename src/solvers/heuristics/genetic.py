"""Genetic Algorithm component of the memetic solver for the PBA: population
initialization (semi-greedy, Section 3.3 of the TCC), tournament selection, crossover
and mutation. The population-level generational loop and the interleaving with VNS
local search live in memetic_solver.py.
"""

from __future__ import annotations

import random

from utils.data_manager import DataManager
from utils.solution import Solution


class GeneticAlgorithm:
    def __init__(
        self,
        data_manager: DataManager,
        population_size: int = 30,
        tournament_size: int = 3,
        crossover_rate: float = 0.9,
        room_inherit_probability: float = 0.5,
        mutation_rate: float = 0.05,
        big_mutation_probability: float = 0.0,
        big_mutation_fraction: float = 0.25,
        seed: int | None = None,
    ):
        self.data_manager = data_manager
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.room_inherit_probability = room_inherit_probability
        self.mutation_rate = mutation_rate
        self.big_mutation_probability = big_mutation_probability
        self.big_mutation_fraction = big_mutation_fraction
        self._rng = random.Random(seed)

        self._all_room_ids = list(data_manager.rooms.keys())
        self._rooms_by_specialty: dict[str, list[int]] = {}
        for room_id, room in data_manager.rooms.items():
            self._rooms_by_specialty.setdefault(room.specialty, []).append(room_id)

    def _candidate_rooms_for(self, required_specialty: str) -> list[int]:
        return self._rooms_by_specialty.get(required_specialty) or self._all_room_ids

    # -- population initialization (Section 3.3: "Solução Inicial") --

    def create_initial_population(self) -> list[Solution]:
        return [self.create_random_individual() for _ in range(self.population_size)]

    def create_random_individual(self) -> Solution:
        """Semi-greedy random construction: each patient gets a single randomly chosen
        room (matching their required specialty when possible) for their entire stay.
        Capacity is ignored here on purpose — it's a soft constraint evaluated (and
        pressured out) by the search, not something the constructor enforces."""
        assignment: dict[int, list[int]] = {}
        for patient_id, patient in self.data_manager.patients.items():
            room_id = self._rng.choice(self._candidate_rooms_for(patient.required_specialty))
            assignment[patient_id] = [room_id] * patient.los
        return Solution(self.data_manager, assignment)

    # -- selection --

    def tournament_select(self, population: list[Solution]) -> Solution:
        contenders = self._rng.sample(population, min(self.tournament_size, len(population)))
        return min(contenders, key=Solution.sort_key)

    # -- crossover --

    def crossover(self, parent_a: Solution, parent_b: Solution) -> Solution:
        """Room-block crossover: patients are grouped by which room they occupy on the
        first day of their stay, in a randomly chosen "primary" parent (the other parent
        is "secondary" — which one is primary is re-rolled on every call, so it's fair
        across many generations even though any single call is asymmetric). Each
        room-group is inherited as a whole block: either every patient who shared that
        room in the primary parent comes across together, or none of them do. Patients
        whose primary-parent room wasn't selected fall back to their own sequence in the
        secondary parent, unchanged.

        This replaces a plain per-patient 50/50 choice, which picked up or dropped
        individual patients regardless of their room-mates and reliably re-introduced
        gender mixing that neither parent actually had (confirmed by testing a
        gender-aware weighting at the per-patient level, which made no measurable
        difference — the room-mate interaction was being destroyed by the crossover
        granularity itself, not by bad luck in the coin flips). Every patient's own room
        sequence is still always inherited whole from a single parent — this only
        changes which parent "wins" a given patient, never splices within their stay.

        Known limitation: grouping uses the patient's first-day room, so someone who
        transfers mid-stay (N3) technically belongs to two room-groups (before/after),
        and only the first is used for the inherit/fallback decision — their post-
        transfer room-mates can still end up from the other parent. Transfers are the
        rarest, most heavily penalized move, so this affects few patients, and even for
        those it's strictly less exposure than the old per-patient scheme had for
        everyone."""
        if self._rng.random() > self.crossover_rate:
            return (parent_a if self._rng.random() < 0.5 else parent_b).copy()

        primary, secondary = (parent_a, parent_b) if self._rng.random() < 0.5 else (parent_b, parent_a)

        patients_by_first_day_room: dict[int, list[int]] = {}
        for patient_id in self.data_manager.patients:
            first_room = primary.get_room_sequence(patient_id)[0]
            patients_by_first_day_room.setdefault(first_room, []).append(patient_id)

        selected_rooms = {
            room_id for room_id in patients_by_first_day_room if self._rng.random() < self.room_inherit_probability
        }

        assignment: dict[int, list[int]] = {}
        for room_id, patient_ids in patients_by_first_day_room.items():
            source = primary if room_id in selected_rooms else secondary
            for patient_id in patient_ids:
                assignment[patient_id] = source.get_room_sequence(patient_id)

        return Solution(self.data_manager, assignment)

    # -- mutation --

    def mutate(self, solution: Solution, rate: float | None = None) -> None:
        """Blind (non-improving) mutation for diversity — distinct from VNS, this never
        checks whether a move helps. Every offspring gets the light per-patient version,
        at `rate` if given or `self.mutation_rate` otherwise: each patient independently
        has that probability of being mutated (see _mutate_patient for what "mutated"
        means). With `big_mutation_probability` chance (0 by default — tested and found
        to hurt more than help at a constant, always-on rate, see MemeticSolver's
        immigrant_fraction for the mechanism that actually earned its keep), the whole
        offspring instead gets `mutate_fraction` at `big_mutation_fraction` — a bigger,
        guaranteed-size perturbation rather than a per-patient coin flip."""
        if self._rng.random() < self.big_mutation_probability:
            self.mutate_fraction(solution, self.big_mutation_fraction)
        else:
            self._light_mutation(solution, rate if rate is not None else self.mutation_rate)

    def _light_mutation(self, solution: Solution, rate: float) -> None:
        for patient_id in self.data_manager.patients:
            if self._rng.random() < rate:
                self._mutate_patient(solution, patient_id)

    def mutate_fraction(self, solution: Solution, fraction: float) -> None:
        """Mutates an exact fraction of patients (rounded, at least 1) rather than a
        per-patient probability — a probabilistic rate can, purely by chance, touch far
        fewer patients than intended, which matters when the caller specifically wants
        a guaranteed bigger jump (constant-rate big_mutation_*, or MemeticSolver's
        stagnation-triggered mutation boost — see its module docstring)."""
        patient_ids = list(self.data_manager.patients.keys())
        n_to_mutate = max(1, round(len(patient_ids) * fraction))
        for patient_id in self._rng.sample(patient_ids, n_to_mutate):
            self._mutate_patient(solution, patient_id)

    def _mutate_patient(self, solution: Solution, patient_id: int) -> None:
        """Picks one of three move types for this patient, mirroring VNS's
        neighborhoods (local_search.py) instead of always doing the same full-stay
        reassignment — so mutation doesn't necessarily erase whatever partial-transfer
        structure N3/N4 refinement already built into a parent:
        - N1-style (_mutate_full_stay): the whole stay to one new room (the original,
          and still the only option for a patient with no one to swap with).
        - N3-style (_mutate_partial): only a random suffix of the stay, preserving the
          prefix — introduces or moves a transfer instead of flattening one away.
        - N4-style (_mutate_overlap_swap): swap rooms with a random overlapping
          patient, only on the days they actually share — touches two patients per
          mutation instead of one, and leaves the rest of both stays untouched."""
        move = self._rng.choice((self._mutate_full_stay, self._mutate_partial, self._mutate_overlap_swap))
        move(solution, patient_id)

    def _mutate_full_stay(self, solution: Solution, patient_id: int) -> None:
        patient = self.data_manager.patients[patient_id]
        candidates = self._candidate_rooms_for(patient.required_specialty)
        solution.set_room_for_stay(patient_id, self._rng.choice(candidates))

    def _mutate_partial(self, solution: Solution, patient_id: int) -> None:
        patient = self.data_manager.patients[patient_id]
        if patient.los < 2:
            self._mutate_full_stay(solution, patient_id)
            return
        split_day = self._rng.choice(list(patient.stay_days)[1:])
        candidates = self._candidate_rooms_for(patient.required_specialty)
        solution.set_room_from_day(patient_id, split_day, self._rng.choice(candidates))

    def _mutate_overlap_swap(self, solution: Solution, patient_id: int) -> None:
        other_id = self._random_overlapping_patient(patient_id)
        if other_id is None:
            self._mutate_full_stay(solution, patient_id)
            return

        patient_a = self.data_manager.patients[patient_id]
        patient_b = self.data_manager.patients[other_id]
        overlap_days = sorted(set(patient_a.stay_days) & set(patient_b.stay_days))

        old_sequence_a = solution.get_room_sequence(patient_id)
        old_sequence_b = solution.get_room_sequence(other_id)
        new_sequence_a = list(old_sequence_a)
        new_sequence_b = list(old_sequence_b)
        for day in overlap_days:
            index_a = day - patient_a.admission_day
            index_b = day - patient_b.admission_day
            new_sequence_a[index_a], new_sequence_b[index_b] = (
                old_sequence_b[index_b],
                old_sequence_a[index_a],
            )

        solution.set_room_sequence(patient_id, new_sequence_a)
        solution.set_room_sequence(other_id, new_sequence_b)

    def _random_overlapping_patient(self, patient_id: int) -> int | None:
        patient = self.data_manager.patients[patient_id]
        candidates: set[int] = set()
        for day in patient.stay_days:
            candidates.update(self.data_manager.patients_by_day.get(day, ()))
        candidates.discard(patient_id)
        if not candidates:
            return None
        return self._rng.choice(sorted(candidates))
