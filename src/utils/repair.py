"""Capacity repair: turns a solution with capacity violation (V_cap > 0) into a feasible one.

Every excess patient-day is relocated to a room with a free bed on that day. Such a room
always exists, because the instance generator never admits more patients on a day than the
hospital's total number of beds. Moves only go into free beds, so none creates a new
violation, and each one lowers V_cap by at least 1 until it reaches 0."""

from __future__ import annotations

import random

from utils.data_manager import DataManager
from utils.solution import Solution


class CapacityRepair:
    def __init__(self, data_manager: DataManager, max_candidates: int = 10, seed: int | None = None):
        """max_candidates: destination rooms tried per patient (specialty matches first)."""
        self.data_manager = data_manager
        self.max_candidates = max_candidates
        self._rng = random.Random(seed)
        self._all_room_ids = list(data_manager.rooms.keys())
        self.last_run_evaluations = 0

    def repair(self, solution: Solution) -> Solution:
        """Removes every capacity violation in place, choosing the move with the lowest fitness
        increase for each excess patient-day."""
        self.last_run_evaluations = 0
        rooms = self.data_manager.rooms

        while solution.capacity_violation > 0:
            for room_id, day in solution.overflowing_room_days():
                while solution.occupancy(room_id, day) > rooms[room_id].capacity:
                    if not self._relocate_cheapest(solution, room_id, day):
                        raise RuntimeError(
                            f"No free bed on day {day}: the instance has more patients than beds"
                        )
        return solution

    def _relocate_cheapest(self, solution: Solution, room_id: int, day: int) -> bool:
        """Moves one occupant of an overflowing (room, day) to a free bed, trying for each
        candidate room the single day and the longest block of the stay around it."""
        free_rooms = [
            other for other in self._all_room_ids
            if other != room_id and solution.occupancy(other, day) < self.data_manager.rooms[other].capacity
        ]
        if not free_rooms:
            return False

        best_move: tuple[float, int, list[int]] | None = None
        for patient_id in solution.occupants(room_id, day):
            patient = self.data_manager.patients[patient_id]
            sequence = solution.get_room_sequence(patient_id)
            day_index = day - patient.admission_day

            for destination in self._destination_rooms(free_rooms, patient.required_specialty):
                for first, last in self._candidate_blocks(solution, sequence, patient, day_index, destination):
                    new_sequence = sequence[:first] + [destination] * (last - first + 1) + sequence[last + 1:]
                    fitness_before = solution.fitness
                    solution.set_room_sequence(patient_id, new_sequence)
                    delta = solution.fitness - fitness_before
                    solution.set_room_sequence(patient_id, sequence)
                    self.last_run_evaluations += 1
                    if best_move is None or delta < best_move[0]:
                        best_move = (delta, patient_id, new_sequence)

        _, patient_id, new_sequence = best_move
        solution.set_room_sequence(patient_id, new_sequence)
        return True

    def _destination_rooms(self, free_rooms: list[int], required_specialty: str) -> list[int]:
        """Up to max_candidates free rooms, rooms of the patient's specialty first."""
        rooms = self.data_manager.rooms
        matching = [room_id for room_id in free_rooms if rooms[room_id].specialty == required_specialty]
        others = [room_id for room_id in free_rooms if rooms[room_id].specialty != required_specialty]
        candidates = self._rng.sample(matching, min(len(matching), self.max_candidates))
        remaining = self.max_candidates - len(candidates)
        if remaining > 0:
            candidates += self._rng.sample(others, min(len(others), remaining))
        return candidates

    def _candidate_blocks(self, solution, sequence, patient, day_index, destination) -> set[tuple[int, int]]:
        """The single day, and the longest run of days around it spent in the same room with a
        free bed in `destination` (moving more days at once avoids a transfer back)."""
        capacity = self.data_manager.rooms[destination].capacity
        current_room = sequence[day_index]

        def can_extend(index: int) -> bool:
            return (
                0 <= index < len(sequence)
                and sequence[index] == current_room
                and solution.occupancy(destination, patient.admission_day + index) < capacity
            )

        first = day_index
        while can_extend(first - 1):
            first -= 1
        last = day_index
        while can_extend(last + 1):
            last += 1
        return {(day_index, day_index), (first, last)}
