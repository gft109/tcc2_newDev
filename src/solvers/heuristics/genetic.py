"""Genetic Algorithm operators of the memetic solver: initial population, tournament
selection, room-block crossover and mutation."""

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
        feasible_seed_fraction: float = 0.2,
        seed: int | None = None,
    ):
        self.data_manager = data_manager
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.room_inherit_probability = room_inherit_probability
        self.mutation_rate = mutation_rate
        self.feasible_seed_fraction = feasible_seed_fraction
        self._rng = random.Random(seed)

        self._all_room_ids = list(data_manager.rooms.keys())
        self._rooms_by_specialty: dict[str, list[int]] = {}
        for room_id, room in data_manager.rooms.items():
            self._rooms_by_specialty.setdefault(room.specialty, []).append(room_id)

    def _candidate_rooms_for(self, required_specialty: str) -> list[int]:
        return self._rooms_by_specialty.get(required_specialty) or self._all_room_ids

    def create_initial_population(self) -> list[Solution]:
        """A feasible_seed_fraction of the population is built capacity-aware; the rest is random.
        Mixing both constructors gave better results than seeding the whole population greedily."""
        n_seeded = round(self.population_size * self.feasible_seed_fraction)
        return [
            self.create_feasible_individual() if i < n_seeded else self.create_random_individual()
            for i in range(self.population_size)
        ]

    def create_random_individual(self) -> Solution:
        """One random specialty-matching room per patient for the whole stay; ignores capacity."""
        assignment: dict[int, list[int]] = {}
        for patient_id, patient in self.data_manager.patients.items():
            room_id = self._rng.choice(self._candidate_rooms_for(patient.required_specialty))
            assignment[patient_id] = [room_id] * patient.los
        return Solution(self.data_manager, assignment)

    def create_feasible_individual(self) -> Solution:
        """Greedy, capacity-aware: each patient (random order) goes to a specialty room with
        free beds, then any room with free beds, and only as a last resort to a full room."""
        patient_ids = list(self.data_manager.patients.keys())
        self._rng.shuffle(patient_ids)

        occupancy: dict[tuple[int, int], int] = {}
        assignment: dict[int, list[int]] = {}

        for patient_id in patient_ids:
            patient = self.data_manager.patients[patient_id]
            room_id = self._pick_room_with_spare_capacity(patient, occupancy)
            assignment[patient_id] = [room_id] * patient.los
            for day in patient.stay_days:
                occupancy[room_id, day] = occupancy.get((room_id, day), 0) + 1

        return Solution(self.data_manager, assignment)

    def _pick_room_with_spare_capacity(self, patient, occupancy: dict[tuple[int, int], int]) -> int:
        days = list(patient.stay_days)

        def has_spare_capacity(room_id: int) -> bool:
            capacity = self.data_manager.rooms[room_id].capacity
            return all(occupancy.get((room_id, day), 0) < capacity for day in days)

        specialty_pool = self._candidate_rooms_for(patient.required_specialty)
        feasible = [room_id for room_id in specialty_pool if has_spare_capacity(room_id)]
        if not feasible:
            feasible = [room_id for room_id in self._all_room_ids if has_spare_capacity(room_id)]
        if not feasible:
            return self._rng.choice(specialty_pool)
        return self._rng.choice(feasible)

    def tournament_select(self, population: list[Solution]) -> Solution:
        contenders = self._rng.sample(population, min(self.tournament_size, len(population)))
        return min(contenders, key=Solution.sort_key)

    def crossover(self, parent_a: Solution, parent_b: Solution) -> Solution:
        """Room-block crossover: patients sharing a room on their first day in the primary parent
        are inherited together, preserving room-mate combinations (e.g. gender)."""
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

    def mutate(self, solution: Solution, rate: float | None = None) -> None:
        """Blind mutation: each patient is mutated with probability `rate`."""
        self._light_mutation(solution, rate if rate is not None else self.mutation_rate)

    def _light_mutation(self, solution: Solution, rate: float) -> None:
        for patient_id in self.data_manager.patients:
            if self._rng.random() < rate:
                self._mutate_patient(solution, patient_id)

    def mutate_fraction(self, solution: Solution, fraction: float) -> None:
        """Mutates an exact fraction of patients (used by the stagnation-triggered boost)."""
        patient_ids = list(self.data_manager.patients.keys())
        n_to_mutate = max(1, round(len(patient_ids) * fraction))
        for patient_id in self._rng.sample(patient_ids, n_to_mutate):
            self._mutate_patient(solution, patient_id)

    def _mutate_patient(self, solution: Solution, patient_id: int) -> None:
        """Applies one random move mirroring the VND neighborhoods: N1, N3 or N4."""
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
