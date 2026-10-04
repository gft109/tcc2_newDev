"""Local search (VND) of the memetic solver: neighborhoods N1-N4, first improvement,
with don't-look bits and an evaluation cap per call."""

from __future__ import annotations

import collections
import random
import time

from utils.data_manager import DataManager
from utils.solution import Solution


EVALUATIONS_PER_PATIENT = 250  # default evaluation cap per call = this * n_patients


class LocalSearch:
    def __init__(
        self,
        data_manager: DataManager,
        max_candidates: int = 10,
        max_evaluations: int | None | str = "auto",
        seed: int | None = None,
    ):
        """max_evaluations="auto" scales the cap with the number of patients; None disables it."""
        if max_evaluations == "auto":
            max_evaluations = EVALUATIONS_PER_PATIENT * len(data_manager.patients)
        self.data_manager = data_manager
        self.max_candidates = max_candidates
        self.max_evaluations = max_evaluations
        self._rng = random.Random(seed)
        self._all_room_ids = list(data_manager.rooms.keys())
        self.last_run_evaluations = 0

    def run(self, solution: Solution, time_budget: float | None = None) -> Solution:
        """Descends to a local optimum (or until the cap/time budget), modifying `solution` in place."""
        deadline = time.perf_counter() + time_budget if time_budget is not None else None
        self.last_run_evaluations = 0
        patient_ids = list(self.data_manager.patients.keys())

        self._descend(solution, patient_ids, deadline)
        return solution

    def _budget_exhausted(self) -> bool:
        return self.max_evaluations is not None and self.last_run_evaluations >= self.max_evaluations

    def _descend(self, solution, patient_ids, deadline) -> None:
        """Processes a queue of active patients; a patient is re-queued only when a move touches it."""
        order = list(patient_ids)
        self._rng.shuffle(order)
        queue = collections.deque(order)
        queued = set(order)

        while queue:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            if self._budget_exhausted():
                break

            patient_id = queue.popleft()
            queued.discard(patient_id)

            touched = self._try_moves(solution, patient_id)
            if touched is None:
                continue
            for pid in touched:
                if pid not in queued:
                    queue.append(pid)
                    queued.add(pid)

    def _try_moves(self, solution: Solution, patient_id: int) -> set[int] | None:
        """Tries N1-N4 in order; returns the patients touched by the applied move, or None."""
        if self._try_change_room(solution, patient_id):
            return {patient_id}

        swap_partner = self._try_swap(solution, patient_id)
        if swap_partner is not None:
            return {patient_id, swap_partner}

        if self._try_partial_change(solution, patient_id):
            return {patient_id}

        partial_swap_partner = self._try_partial_swap(solution, patient_id)
        if partial_swap_partner is not None:
            return {patient_id, partial_swap_partner}

        return None

    def _candidate_rooms(
        self, exclude_room_id: int, solution: Solution, days: list[int]
    ) -> list[int]:
        """Samples 3x max_candidates rooms and keeps those with the fewest overflowing days."""
        pool = [room_id for room_id in self._all_room_ids if room_id != exclude_room_id]
        if len(pool) <= self.max_candidates:
            return pool

        sample_size = min(len(pool), self.max_candidates * 3)
        sampled = self._rng.sample(pool, sample_size)

        def overflow_days(room_id: int) -> int:
            capacity = self.data_manager.rooms[room_id].capacity
            return sum(1 for day in days if solution.occupancy(room_id, day) >= capacity)

        sampled.sort(key=overflow_days)
        return sampled[: self.max_candidates]

    def _overlapping_patients(self, patient_id: int) -> list[int]:
        patient = self.data_manager.patients[patient_id]
        candidates: set[int] = set()
        for day in patient.stay_days:
            candidates.update(self.data_manager.patients_by_day.get(day, ()))
        candidates.discard(patient_id)
        if len(candidates) <= self.max_candidates:
            return sorted(candidates)
        return self._rng.sample(sorted(candidates), self.max_candidates)

    def _try_change_room(self, solution: Solution, patient_id: int) -> bool:
        old_sequence = solution.get_room_sequence(patient_id)
        current_room = old_sequence[0]
        current_key = solution.sort_key()
        patient_days = list(self.data_manager.patients[patient_id].stay_days)

        for candidate_room in self._candidate_rooms(current_room, solution, patient_days):
            solution.set_room_for_stay(patient_id, candidate_room)
            self.last_run_evaluations += 1
            if solution.sort_key() < current_key:
                return True
            solution.set_room_sequence(patient_id, old_sequence)

        return False

    def _try_swap(self, solution: Solution, patient_id: int) -> int | None:
        old_sequence_a = solution.get_room_sequence(patient_id)
        room_a = old_sequence_a[0]
        current_key = solution.sort_key()

        for other_id in self._overlapping_patients(patient_id):
            old_sequence_b = solution.get_room_sequence(other_id)
            room_b = old_sequence_b[0]
            if room_a == room_b:
                continue

            solution.set_room_for_stay(patient_id, room_b)
            solution.set_room_for_stay(other_id, room_a)
            self.last_run_evaluations += 1
            if solution.sort_key() < current_key:
                return other_id

            solution.set_room_sequence(patient_id, old_sequence_a)
            solution.set_room_sequence(other_id, old_sequence_b)

        return None

    def _try_partial_change(self, solution: Solution, patient_id: int) -> bool:
        patient = self.data_manager.patients[patient_id]
        if patient.los < 2:
            return False  # no interior day to split the stay on

        old_sequence = solution.get_room_sequence(patient_id)
        current_key = solution.sort_key()

        for split_day in list(patient.stay_days)[1:]:
            exclude_room = old_sequence[split_day - patient.admission_day]
            remaining_days = [day for day in patient.stay_days if day >= split_day]
            for candidate_room in self._candidate_rooms(exclude_room, solution, remaining_days):
                solution.set_room_from_day(patient_id, split_day, candidate_room)
                self.last_run_evaluations += 1
                if solution.sort_key() < current_key:
                    return True
                solution.set_room_sequence(patient_id, old_sequence)

        return False

    def _try_partial_swap(self, solution: Solution, patient_id: int) -> int | None:
        """Swaps rooms only on the days both stays overlap, keeping the rest of each sequence."""
        old_sequence_a = solution.get_room_sequence(patient_id)
        patient_a = self.data_manager.patients[patient_id]
        current_key = solution.sort_key()

        for other_id in self._overlapping_patients(patient_id):
            patient_b = self.data_manager.patients[other_id]
            overlap_days = sorted(set(patient_a.stay_days) & set(patient_b.stay_days))
            if not overlap_days:
                continue

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

            if new_sequence_a == old_sequence_a:
                continue  # both patients already share the same room on every overlap day

            solution.set_room_sequence(patient_id, new_sequence_a)
            solution.set_room_sequence(other_id, new_sequence_b)
            self.last_run_evaluations += 1
            if solution.sort_key() < current_key:
                return other_id

            solution.set_room_sequence(patient_id, old_sequence_a)
            solution.set_room_sequence(other_id, old_sequence_b)

        return None
