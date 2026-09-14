"""Solution representation and evaluation for the Patient-Bed Allocation Problem (PBA).

A single representation shared by the exact (Gurobi, via gurobi_formater.py) and the
heuristic (memetic) solvers, so solution quality can be compared on equal footing using
the formal objective Z from Eq. 3.6 of the TCC.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from utils.data_manager import DataManager

WEIGHTS = {
    "W_TRANSF": 50,
    "W_GEN": 75,
    "W_SPEC": 100,
    "W_CAP": 1200,
}

COMPARISON_MODE = "fitness_only"
"""Global switch for Solution.sort_key() (see there). "fitness_only" (default as of
this A/B test) compares purely on `fitness`, letting capacity_violation compete on equal
footing with the other cost components through W_CAP alone instead of getting an
absolute lexicographic override. Measured across pequeno/medio/grande (3 seeds, 60s
each): fitness_only reached 5-16% lower (better) total fitness than "lexicographic" on
every instance, at the cost of roughly 2x the capacity violation — the lexicographic
override was closing off search paths that traded a little more (still W_CAP=400
-weighted) capacity infeasibility for a bigger win elsewhere. "lexicographic" is kept
available for going back to a hard feasibility-first ranking (CLAUDE.md's original
decision #1) if W_CAP tuning under fitness_only doesn't bring capacity_violation back
down to an acceptable level."""


class Solution:
    """assignment[patient_id][i] is the room occupied by that patient on the i-th day
    of their stay (calendar day = patient.admission_day + i), mirroring x_{p,r,d}.

    Costs are tracked incrementally (updated on every mutation, O(patient.los) per move)
    rather than recomputed from scratch on read. This matters because the VNS local
    search evaluates many candidate moves per patient — with thousands of patients per
    instance, an O(total patient-days) recompute per candidate would make local search
    impractical at the larger instance sizes.
    """

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

    # -- accessors/mutators used by GA crossover and VNS neighborhoods (N1/N2/N3) --

    def get_room(self, patient_id: int, day: int) -> int:
        patient = self.data_manager.patients[patient_id]
        return self.assignment[patient_id][day - patient.admission_day]

    def get_room_sequence(self, patient_id: int) -> list[int]:
        """Returns a snapshot (safe to mutate) of a patient's room sequence — used by the
        VNS to remember the pre-move state so a non-improving trial can be reverted."""
        return list(self.assignment[patient_id])

    def set_room_sequence(self, patient_id: int, new_sequence: list[int]) -> None:
        """Core mutation primitive: replaces a patient's full room sequence, updating all
        incremental cost bookkeeping. set_room_for_stay/set_room_from_day are thin
        wrappers around this; it also serves as the revert operation for VNS trial moves."""
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
        """N1 (Change Room): moves a patient into room_id for their entire stay."""
        patient = self.data_manager.patients[patient_id]
        self.set_room_sequence(patient_id, [room_id] * patient.los)

    def set_room_from_day(self, patient_id: int, day: int, room_id: int) -> None:
        """N3 (Partial Change Room): moves a patient into room_id from `day` onward."""
        patient = self.data_manager.patients[patient_id]
        start_index = day - patient.admission_day
        old_sequence = self.assignment[patient_id]
        new_sequence = old_sequence[:start_index] + [room_id] * (patient.los - start_index)
        self.set_room_sequence(patient_id, new_sequence)

    def copy(self) -> "Solution":
        return Solution(self.data_manager, self.assignment)

    # -- incremental cost bookkeeping --

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

    def recomputed_from_scratch(self) -> "Solution":
        """Rebuilds a Solution from this one's assignment via a fresh __init__ pass,
        bypassing incremental bookkeeping entirely. Used to sanity-check that incremental
        updates haven't drifted from the true cost (see verify_instances-style checks)."""
        return Solution(self.data_manager, self.assignment)

    # -- evaluation --

    @property
    def specialty_cost(self) -> float:
        return self._specialty_cost

    @property
    def transfer_cost(self) -> float:
        return self._transfer_cost

    @property
    def gender_cost(self) -> float:
        return self._gender_cost

    @property
    def capacity_violation(self) -> int:
        """Total beds of capacity exceeded, summed over every (room, day). Always 0 for
        Gurobi solutions (capacity is a hard constraint there); can be > 0 for the
        heuristic, which treats capacity as a soft constraint (see CLAUDE.md)."""
        return self._capacity_violation

    @property
    def capacity_cost(self) -> float:
        """Weighted capacity penalty (W_CAP * capacity_violation). Reported as its own
        cost line in the Gurobi-vs-heuristic comparison output; always 0 for Gurobi."""
        return WEIGHTS["W_CAP"] * self.capacity_violation

    @property
    def objective_value(self) -> float:
        """Z from Eq. 3.6 — the formal PBA objective, comparable between Gurobi and the
        heuristic (does not include the capacity penalty, which is not part of Z)."""
        return self.specialty_cost + self.transfer_cost + self.gender_cost

    @property
    def fitness(self) -> float:
        """Objective plus the capacity penalty. Drives GA/VNS search; equals
        objective_value whenever the solution is capacity-feasible."""
        return self.objective_value + self.capacity_cost

    def sort_key(self) -> tuple[int, float] | float:
        """Ranks/accepts heuristic solutions — see COMPARISON_MODE (module level) for
        the two modes and the A/B result behind the current default. "fitness_only"
        (default) is plain `fitness`: capacity_violation competes through the ordinary
        weighted capacity_cost term, same as any other cost component, with no
        absolute override. "lexicographic" instead ranks by (capacity_violation,
        fitness): no infeasible solution is ever reported as better than a feasible
        one, regardless of how W_CAP compares to the other weights — this was the
        original default (CLAUDE.md design decision #1) until A/B testing found it
        cost 5-16% worse total fitness for roughly half the capacity_violation."""
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
    """Standardized output of a single solver run, produced by both gurobi_formater.py
    and the heuristic solvers so main.py / results_representation.py can compare Gurobi
    against the heuristic uniformly (see the expected terminal output format in
    CLAUDE.md: cost breakdown by category, initial cost, total cost, runtime, and the
    number of cost evaluations performed).
    """

    solution: Solution
    runtime: float
    initial_fitness: float | None = None
    """Fitness of the starting point before optimization (GA generation 0). Not
    applicable to the exact model, which has no notion of an initial candidate — None
    there (reported as NULL, per CLAUDE.md)."""
    evaluation_count: int | None = None
    """How many times a candidate solution's cost was computed/compared. For the
    heuristic, the number of fitness evaluations across GA/VNS; for Gurobi, the simplex
    iteration count (IterCount) as the closest analogous measure of search effort."""
    convergence_history: list[float] = field(default_factory=list)
    """Best fitness found so far, indexed by generation (heuristic only; empty for
    Gurobi). Used by results_representation.py for the convergence plot (Section 3.5)."""

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
