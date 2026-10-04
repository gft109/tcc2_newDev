"""Solution representation and incremental evaluation, shared by both solvers."""

from __future__ import annotations

from dataclasses import dataclass, field

from utils.data_manager import DataManager

WEIGHTS = {
    "W_TRANSF": 50,
    "W_GEN": 75,
    "W_SPEC": 100,
    "W_CAP": 1900,
}

COMPARISON_MODE = "fitness_only"  # or "lexicographic": (capacity_violation, fitness)


class Solution:
    """assignment[patient_id][i] is the patient's room on day admission_day + i (mirrors x_{p,r,d})."""

    def __init__(self, data_manager: DataManager, assignment: dict[int, list[int]]):
        self.data_manager = data_manager
        self.assignment: dict[int, list[int]] = {pid: list(rooms) for pid, rooms in assignment.items()}

        self._room_day_occupants: dict[tuple[int, int], set[int]] = {}
        self._specialty_cost = 0.0
        self._transfer_cost = 0.0
        self._gender_cost = 0.0
        self._capacity_violation = 0

        for patient_id, room_sequence in self.assignment.items():
            patient = self.data_manager.patients[patient_id]
            for offset, room_id in enumerate(room_sequence):
                self._add_occupant(patient_id, room_id, patient.admission_day + offset)
            self._transfer_cost += WEIGHTS["W_TRANSF"] * self._count_transfers(room_sequence)

    def get_room_sequence(self, patient_id: int) -> list[int]:
        return list(self.assignment[patient_id])

    def set_room_sequence(self, patient_id: int, new_sequence: list[int]) -> None:
        """Replaces a patient's room sequence, updating costs incrementally."""
        patient = self.data_manager.patients[patient_id]
        old_sequence = self.assignment[patient_id]

        for offset, (old_room, new_room) in enumerate(zip(old_sequence, new_sequence)):
            if old_room != new_room:
                day = patient.admission_day + offset
                self._remove_occupant(patient_id, old_room, day)
                self._add_occupant(patient_id, new_room, day)

        old_transfers = self._count_transfers(old_sequence)
        new_transfers = self._count_transfers(new_sequence)
        self._transfer_cost += WEIGHTS["W_TRANSF"] * (new_transfers - old_transfers)

        self.assignment[patient_id] = list(new_sequence)

    def set_room_for_stay(self, patient_id: int, room_id: int) -> None:
        patient = self.data_manager.patients[patient_id]
        self.set_room_sequence(patient_id, [room_id] * patient.los)

    def set_room_from_day(self, patient_id: int, day: int, room_id: int) -> None:
        patient = self.data_manager.patients[patient_id]
        start_index = day - patient.admission_day
        old_sequence = self.assignment[patient_id]
        new_sequence = old_sequence[:start_index] + [room_id] * (patient.los - start_index)
        self.set_room_sequence(patient_id, new_sequence)

    def copy(self) -> "Solution":
        return Solution(self.data_manager, self.assignment)

    @staticmethod
    def _count_transfers(room_sequence: list[int]) -> int:
        return sum(1 for a, b in zip(room_sequence, room_sequence[1:]) if a != b)

    def _room_day_genders(self, occupant_ids: set[int]) -> tuple[int, int]:
        patients = self.data_manager.patients
        female = sum(1 for pid in occupant_ids if patients[pid].gender == "F")
        return female, len(occupant_ids) - female

    def _add_occupant(self, patient_id: int, room_id: int, day: int) -> None:
        patient = self.data_manager.patients[patient_id]
        room = self.data_manager.rooms[room_id]
        occupants = self._room_day_occupants.setdefault((room_id, day), set())

        old_count = len(occupants)
        old_female, old_male = self._room_day_genders(occupants)
        old_excess = max(0, old_count - room.capacity)
        old_mixed = old_female > 0 and old_male > 0

        occupants.add(patient_id)

        new_female = old_female + (1 if patient.gender == "F" else 0)
        new_male = old_male + (1 if patient.gender == "M" else 0)
        new_excess = max(0, (old_count + 1) - room.capacity)
        new_mixed = new_female > 0 and new_male > 0

        self._capacity_violation += new_excess - old_excess
        if new_mixed and not old_mixed:
            self._gender_cost += WEIGHTS["W_GEN"]
        if room.specialty != patient.required_specialty:
            self._specialty_cost += WEIGHTS["W_SPEC"]

    def _remove_occupant(self, patient_id: int, room_id: int, day: int) -> None:
        patient = self.data_manager.patients[patient_id]
        room = self.data_manager.rooms[room_id]
        occupants = self._room_day_occupants[(room_id, day)]

        old_count = len(occupants)
        old_female, old_male = self._room_day_genders(occupants)
        old_excess = max(0, old_count - room.capacity)
        old_mixed = old_female > 0 and old_male > 0

        occupants.discard(patient_id)

        new_female = old_female - (1 if patient.gender == "F" else 0)
        new_male = old_male - (1 if patient.gender == "M" else 0)
        new_excess = max(0, (old_count - 1) - room.capacity)
        new_mixed = new_female > 0 and new_male > 0

        self._capacity_violation += new_excess - old_excess
        if old_mixed and not new_mixed:
            self._gender_cost -= WEIGHTS["W_GEN"]
        if room.specialty != patient.required_specialty:
            self._specialty_cost -= WEIGHTS["W_SPEC"]

    @property
    def specialty_cost(self) -> float:
        return self._specialty_cost

    @property
    def transfer_cost(self) -> float:
        return self._transfer_cost

    @property
    def gender_cost(self) -> float:
        return self._gender_cost

    # operational metrics, computed only for reported solutions

    @property
    def specialty_mismatch_count(self) -> int:
        """Patient-days in a room of the wrong specialty."""
        return round(self._specialty_cost / WEIGHTS["W_SPEC"])

    @property
    def transfer_count(self) -> int:
        return round(self._transfer_cost / WEIGHTS["W_TRANSF"])

    @property
    def mixed_room_day_count(self) -> int:
        return round(self._gender_cost / WEIGHTS["W_GEN"])

    def occupancy_rates(self) -> list[float]:
        """occupants/capacity for every (room, day), including empty ones."""
        rooms = self.data_manager.rooms
        horizon = self.data_manager.horizon
        return [
            self.occupancy(room_id, day) / room.capacity
            for room_id, room in rooms.items()
            for day in range(1, horizon + 1)
        ]

    def occupancy_rate_by_specialty(self) -> dict[str, float]:
        rooms = self.data_manager.rooms
        horizon = self.data_manager.horizon
        rates_by_specialty: dict[str, list[float]] = {}
        for room_id, room in rooms.items():
            for day in range(1, horizon + 1):
                rate = self.occupancy(room_id, day) / room.capacity
                rates_by_specialty.setdefault(room.specialty, []).append(rate)
        return {specialty: sum(rates) / len(rates) for specialty, rates in rates_by_specialty.items()}

    def peak_occupancy(self) -> tuple[int, int, int]:
        """(day, patients in the hospital, total beds) for the busiest day."""
        rooms = self.data_manager.rooms
        horizon = self.data_manager.horizon
        totals = {
            day: sum(self.occupancy(room_id, day) for room_id in rooms)
            for day in range(1, horizon + 1)
        }
        peak_day = max(totals, key=totals.get)
        total_capacity = sum(room.capacity for room in rooms.values())
        return peak_day, totals[peak_day], total_capacity

    def occupancy(self, room_id: int, day: int) -> int:
        return len(self._room_day_occupants.get((room_id, day), ()))

    @property
    def capacity_violation(self) -> int:
        """V_cap: patient-days above room capacity, summed over all rooms and days."""
        return self._capacity_violation

    @property
    def capacity_cost(self) -> float:
        return WEIGHTS["W_CAP"] * self.capacity_violation

    @property
    def objective_value(self) -> float:
        """Z (TCC Eq. 3.8), without the capacity penalty."""
        return self.specialty_cost + self.transfer_cost + self.gender_cost

    @property
    def fitness(self) -> float:
        """F = Z + W_CAP * capacity_violation."""
        return self.objective_value + self.capacity_cost

    def sort_key(self) -> tuple[int, float] | float:
        """Ranking key used by the heuristic, see COMPARISON_MODE."""
        if COMPARISON_MODE == "fitness_only":
            return self.fitness
        return (self.capacity_violation, self.fitness)

    def __lt__(self, other: "Solution") -> bool:
        return self.sort_key() < other.sort_key()

    def __repr__(self) -> str:
        return (
            f"Solution(objective_value={self.objective_value:.1f}, "
            f"capacity_violation={self.capacity_violation})"
        )


@dataclass
class SolverResult:
    """Output of one solver run, in the same format for Gurobi and the heuristic."""

    solution: Solution
    runtime: float
    initial_fitness: float | None = None
    evaluation_count: int | None = None  # heuristic: fitness evaluations; Gurobi: simplex iterations
    convergence_history: list[float] = field(default_factory=list)  # best fitness per generation
    upper_bound: float | None = None  # Gurobi only
    lower_bound: float | None = None  # Gurobi only
    gap: float | None = None  # Gurobi only
    is_optimal: bool | None = None  # Gurobi only
    best_feasible_fitness: float | None = None  # heuristic: best capacity-feasible individual seen
    best_feasible_generation: int | None = None
    best_feasible_solution: Solution | None = None

    @property
    def specialty_cost(self) -> float:
        return self.solution.specialty_cost

    @property
    def transfer_cost(self) -> float:
        return self.solution.transfer_cost

    @property
    def gender_cost(self) -> float:
        return self.solution.gender_cost

    @property
    def capacity_cost(self) -> float:
        return self.solution.capacity_cost

    @property
    def total_cost(self) -> float:
        return self.solution.fitness
