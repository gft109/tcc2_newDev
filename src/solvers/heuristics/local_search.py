"""Variable Neighborhood Search (VNS) for the Patient-Bed Allocation Problem (PBA),
per Section 3.4 of the TCC.

Four neighborhoods are tried in sequence for each patient as the descent (local search)
step — N1 (Change Room), N2 (Swap Patients), N3 (Partial Change Room), N4 (Partial Swap
Patients), first improvement across the four. These mirror the CR, SP, PCR, and PSP
moves of Ceschia & Schaerf (2011) — the paper behind this model's room/bed abstraction —
adapted to this model's day-by-day room sequences rather than their at-most-one-transfer
segment representation. The descent stops when no active patient has an improving move
left in any of them — a local optimum (see the don't-look-bits paragraph below for how
"active" is tracked).

On its own, that descent is really a Variable Neighborhood Descent (VND): once it hits a
local optimum it has nothing left to try and just stops there. `_shake` (perturb a
growing number of patients at random, then descend again — "Basic VNS", Mladenović &
Hansen) is available to push past that, but it's off by default (max_shake_level=0):
each shake+re-descend cycle can consume a full individual's whole local-search time
budget, which in testing meant far fewer GA generations completed overall — and for this
problem, generation-over-generation diversity from crossover mattered more than
exhaustively escaping any one individual's local optimum. Pass max_shake_level > 0 to
re-enable it (it did help when local search was best-improvement rather than
first-improvement, in case that trade-off is revisited later).

`max_evaluations` bounds each call to a fixed amount of work. Without it, a single
"descend to local optimum" call on a large instance could in principle consume an entire
generation's time budget by itself, starving the GA of generations — this is precisely
the failure mode that motivates Ceschia & Schaerf (2011), the paper behind this model's
room/bed abstraction, to use single-move-sampling Simulated Annealing instead of a
full-sweep descent: SA's per-iteration cost is O(1), so it can afford hundreds of
millions of iterations. Keeping VND here (rather than switching to SA) is a deliberate
choice — it fits the memetic GA+local-search structure the TCC targets — but it needs
the same kind of hard cap SA gets from a fixed iteration count. Unlike SA's iteration
count, though, one flat number doesn't fit every instance size here: a cap sized for
`grande` left `muito_grande` cut off mid-descent (see EVALUATIONS_PER_PATIENT below), so
the default scales with `len(data_manager.patients)` instead of being a single constant.

Each candidate move is a first-improvement trial: apply it, check Solution.sort_key(),
keep it if it improved and revert otherwise. This is only cheap because Solution tracks
its costs incrementally (see utils/solution.py) — an O(instance size) recompute per
candidate would make this impractical on the larger instances.

`_descend` uses "don't-look bits" (Johnson & McGeoch's term from TSP/QAP local search)
rather than the naive alternative of repeatedly sweeping every patient through N1, then
every patient through N2/N3, restarting at N1 from scratch whenever ANY patient
improves anywhere. That naive scheme is wasteful for exactly the reason don't-look bits
were invented: after the first pass, most patients are already at a local optimum and
stay that way — re-scanning all of them again after one unrelated patient's move is
pure waste, and it's what made a single descend() call expensive enough to need the
`max_evaluations` cap above in the first place. Instead, each patient carries an
active/inactive status in a work queue: a patient is tried (N1, then N2, then N3) only
while active; if none of its moves improve the solution it goes inactive and is skipped
until a move elsewhere reactivates it (only the patients actually touched by an applied
move — the mover itself, and its partner on a swap — are reactivated, not the whole
instance). The descent still reaches the same kind of local optimum (no active patient
has an improving move left), it just stops paying to re-verify patients nothing has
disturbed.
"""

from __future__ import annotations

import collections
import random
import time

from utils.data_manager import DataManager
from utils.solution import Solution


EVALUATIONS_PER_PATIENT = 250
"""Scaling factor for the "auto" max_evaluations default (see LocalSearch.__init__).
Measured full-convergence cost (uncapped, single random individual, N1-N4) came out to
~150-190 evaluations/patient on grande (1600 patients) and muito_grande (6400 patients);
250 keeps a safety margin above that without needing a fresh calibration run per
instance."""


class LocalSearch:
    def __init__(
        self,
        data_manager: DataManager,
        max_candidates: int = 10,
        max_shake_level: int = 0,
        shake_fraction: float = 0.05,
        max_evaluations: int | None | str = "auto",
        seed: int | None = None,
    ):
        """`max_evaluations="auto"` (default) scales the cap with instance size
        (EVALUATIONS_PER_PATIENT * n_patients) rather than using one flat number for
        every instance — a flat cap tuned for one instance size was found to badly
        undershoot larger ones (300000, enough for `grande`, left `muito_grande`
        individuals cut off mid-descent, at 3-6x worse cost/capacity-violation than a
        cap scaled to its size). Pass an explicit int to override, or None for no cap
        at all (used to measure true convergence cost, e.g. in tuning scripts)."""
        if max_evaluations == "auto":
            max_evaluations = EVALUATIONS_PER_PATIENT * len(data_manager.patients)
        self.data_manager = data_manager
        self.max_candidates = max_candidates
        self.max_shake_level = max_shake_level
        self.shake_fraction = shake_fraction
        self.max_evaluations = max_evaluations
        self._rng = random.Random(seed)
        self._all_room_ids = list(data_manager.rooms.keys())
        self.last_run_evaluations = 0

    def run(self, solution: Solution, time_budget: float | None = None) -> Solution:
        """Basic VNS: descend to a local optimum, then repeatedly shake + re-descend,
        escalating the shake strength on failure and resetting it on any improvement,
        until `max_shake_level` is exceeded, `max_evaluations` moves have been tried, or
        `time_budget` seconds have elapsed. Returns the best solution found (a new object
        once shaking kicks in — callers should use the return value, not assume in-place
        mutation)."""
        deadline = time.perf_counter() + time_budget if time_budget is not None else None
        self.last_run_evaluations = 0
        patient_ids = list(self.data_manager.patients.keys())

        self._descend(solution, patient_ids, deadline)
        best = solution

        shake_level = 1
        while shake_level <= self.max_shake_level:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            if self._budget_exhausted():
                break

            candidate = best.copy()
            self._shake(candidate, shake_level)
            self._descend(candidate, patient_ids, deadline)

            if candidate.sort_key() < best.sort_key():
                best = candidate
                shake_level = 1
            else:
                shake_level += 1

        return best

    def _budget_exhausted(self) -> bool:
        return self.max_evaluations is not None and self.last_run_evaluations >= self.max_evaluations

    def _descend(self, solution, patient_ids, deadline) -> None:
        """Don't-look-bits descent (see module docstring): processes an active-patient
        queue, trying N1 -> N2 -> N3 for each (first improvement across the three).
        Applying a move reactivates only the patients it touched; everyone else stays
        untouched. Runs until the queue drains (a local optimum), the evaluation cap is
        reached, or time runs out."""
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
        """Tries N1, then N2, then N3, then N4 for a single patient (first improvement
        across the four). Returns the set of patient ids to reactivate if a move was
        applied (the patient itself, plus its swap partner for N2/N4), or None if none
        of its candidate moves improved the solution."""
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

    def _shake(self, solution: Solution, level: int) -> None:
        """Randomly reassigns a growing number of patients (proportional to `level`) to
        random rooms, in place. This is a blind perturbation — unlike the descent moves,
        it is not checked for improvement; its only job is to kick the search out of the
        local optimum's basin of attraction before the descent runs again."""
        patient_ids = list(self.data_manager.patients.keys())
        n_to_shake = min(len(patient_ids), max(1, round(level * self.shake_fraction * len(patient_ids))))
        for patient_id in self._rng.sample(patient_ids, n_to_shake):
            solution.set_room_for_stay(patient_id, self._rng.choice(self._all_room_ids))

    # -- candidate pools --

    def _candidate_rooms(
        self, exclude_room_id: int, solution: Solution, days: list[int]
    ) -> list[int]:
        """Candidate rooms for N1/N3 moves. On instances where the room pool is bigger
        than max_candidates, a uniform-random sample rarely lands on the specific room
        that would relieve a capacity overflow (confirmed on "medio": 40 rooms vs.
        max_candidates=10 means ~25% coverage per try, vs. ~100% on "pequeno" where the
        whole pool fits — the heuristic never found a single capacity-feasible
        individual there). To fix that without an O(all rooms) scan on large instances,
        a widened random sample (3x max_candidates, still capped by pool size) is
        ranked by how many of `days` it has spare capacity for and truncated back down
        to max_candidates — biased toward capacity relief, still random beyond that."""
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

    # -- N1: Change Room --

    def _try_change_room(self, solution: Solution, patient_id: int) -> bool:
        """First-improvement: applies the first candidate room found that improves
        sort_key, rather than searching for the single best one. Cheaper per move, which
        in practice matters more than move quality here — it lets far more generations
        run in the same time budget, and generation-over-generation variation (crossover)
        turned out to matter more than exhaustively optimizing any one move."""
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

    # -- N2: Swap Patients --

    def _try_swap(self, solution: Solution, patient_id: int) -> int | None:
        """Returns the partner patient's id if an improving swap was applied (needed by
        _try_moves to reactivate both sides under don't-look bits), or None otherwise."""
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

    # -- N3: Partial Change Room --

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

    # -- N4: Partial Swap Patients --

    def _try_partial_swap(self, solution: Solution, patient_id: int) -> int | None:
        """Ceschia & Schaerf (2011)'s PSP move, adapted to our day-by-day room
        sequences: N2 (_try_swap) always exchanges two patients' ENTIRE stays, even if
        only a few overlapping days are actually in conflict — which also means it
        flattens any pre-existing partial-transfer structure on both sides. N4 instead
        exchanges, day by day, only the days the two patients' stays actually overlap,
        leaving the rest of each patient's sequence untouched. It's a strict
        generalization of N2 (identical to it when both patients' stays fully overlap
        and neither has a prior transfer) that can resolve a conflict confined to part
        of a stay without disturbing the rest."""
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
