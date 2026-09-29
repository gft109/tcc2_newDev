"""Exact Integer Linear Programming model for the Patient-Bed Allocation Problem (PBA),
per Section 3.2 of the TCC (Eqs. 3.6-3.13), solved via Gurobi.

Practical only for small/medium instances: the full model has one binary variable per
(patient, room, day of stay) plus per-room-day gender/mix tracking variables, which grows
too large to solve to optimality for the larger instances (see TCC Section 2.2.1). For
those, use relaxation=True to obtain an LP relaxation lower bound instead (Section 3.6).
"""

from __future__ import annotations

import time

import gurobipy as gp
from gurobipy import GRB

from utils.data_manager import DataManager
from utils.solution import WEIGHTS


class GurobiSolver:
    def __init__(
        self,
        data_manager: DataManager,
        time_limit: float | None = None,
        mip_gap: float | None = None,
        relaxation: bool = False,
        verbose: bool = False,
    ):
        self.data_manager = data_manager
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.relaxation = relaxation
        self.verbose = verbose

        self.model: gp.Model | None = None
        self.x: dict[tuple[int, int, int], gp.Var] = {}
        self.t: dict[tuple[int, int, int], gp.Var] = {}
        self.f: dict[tuple[int, int], gp.Var] = {}
        self.m: dict[tuple[int, int], gp.Var] = {}
        self.b: dict[tuple[int, int], gp.Var] = {}

        self.runtime: float = 0.0
        self.status: int | None = None

    def build_model(self) -> None:
        dm = self.data_manager
        patients = dm.patients
        rooms = dm.rooms
        horizon = dm.horizon

        var_type = GRB.CONTINUOUS if self.relaxation else GRB.BINARY
        var_kwargs = {"lb": 0.0, "ub": 1.0} if self.relaxation else {}

        model = gp.Model("PBA_exact")
        model.Params.OutputFlag = 1 if self.verbose else 0
        if self.time_limit is not None:
            model.Params.TimeLimit = self.time_limit
        if self.mip_gap is not None:
            model.Params.MIPGap = self.mip_gap

        x: dict[tuple[int, int, int], gp.Var] = {}
        t: dict[tuple[int, int, int], gp.Var] = {}
        f: dict[tuple[int, int], gp.Var] = {}
        m: dict[tuple[int, int], gp.Var] = {}
        b: dict[tuple[int, int], gp.Var] = {}

        for pid, patient in patients.items():
            stay_days = list(patient.stay_days)
            for r in rooms:
                for d in stay_days:
                    x[pid, r, d] = model.addVar(vtype=var_type, name=f"x_{pid}_{r}_{d}", **var_kwargs)
                for d in stay_days[:-1]:
                    t[pid, r, d] = model.addVar(vtype=var_type, name=f"t_{pid}_{r}_{d}", **var_kwargs)

        for r in rooms:
            for d in range(1, horizon + 1):
                f[r, d] = model.addVar(vtype=var_type, name=f"f_{r}_{d}", **var_kwargs)
                m[r, d] = model.addVar(vtype=var_type, name=f"m_{r}_{d}", **var_kwargs)
                b[r, d] = model.addVar(vtype=var_type, name=f"b_{r}_{d}", **var_kwargs)

        model.update()

        # Eq. 3.6
        specialty_cost = gp.quicksum(
            WEIGHTS["W_SPEC"] * var
            for (pid, r, _d), var in x.items()
            if rooms[r].specialty != patients[pid].required_specialty
        )
        transfer_cost = WEIGHTS["W_TRANSF"] * gp.quicksum(t.values())
        gender_cost = WEIGHTS["W_GEN"] * gp.quicksum(b.values())
        model.setObjective(specialty_cost + transfer_cost + gender_cost, GRB.MINIMIZE)

        # Eq. 3.7 - single, continuous assignment
        for pid, patient in patients.items():
            for d in patient.stay_days:
                model.addConstr(gp.quicksum(x[pid, r, d] for r in rooms) == 1, name=f"assign_{pid}_{d}")

        # Eq. 3.8 - room capacity
        patients_by_day: dict[int, list[int]] = {}
        for pid, patient in patients.items():
            for d in patient.stay_days:
                patients_by_day.setdefault(d, []).append(pid)

        for r, room in rooms.items():
            for d in range(1, horizon + 1):
                occupants = patients_by_day.get(d, [])
                model.addConstr(
                    gp.quicksum(x[pid, r, d] for pid in occupants) <= room.capacity,
                    name=f"cap_{r}_{d}",
                )

        # Eq. 3.9 - transfer tracking
        for pid, patient in patients.items():
            stay_days = list(patient.stay_days)
            for r in rooms:
                for d in stay_days[:-1]:
                    model.addConstr(
                        t[pid, r, d] >= x[pid, r, d] - x[pid, r, d + 1], name=f"transf_{pid}_{r}_{d}"
                    )

        # Eq. 3.10 / 3.11 - gender identification
        for pid, patient in patients.items():
            gender_flags = f if patient.gender == "F" else m
            for r in rooms:
                for d in patient.stay_days:
                    model.addConstr(gender_flags[r, d] >= x[pid, r, d], name=f"gender_{pid}_{r}_{d}")

        # Eq. 3.12 - mixed room detection
        for r in rooms:
            for d in range(1, horizon + 1):
                model.addConstr(b[r, d] >= m[r, d] + f[r, d] - 1, name=f"mixed_{r}_{d}")

        self.model = model
        self.x, self.t, self.f, self.m, self.b = x, t, f, m, b

    def solve(self) -> None:
        if self.model is None:
            self.build_model()

        start = time.perf_counter()
        self.model.optimize()
        self.runtime = time.perf_counter() - start
        self.status = self.model.Status

    @property
    def objective_value(self) -> float | None:
        if self.model is None or self.model.SolCount == 0:
            return None
        return self.model.ObjVal

    @property
    def best_bound(self) -> float | None:
        """Best known lower bound on Z. Equals the true optimum when status is OPTIMAL;
        used as the LP relaxation lower bound for medium/large instances (Section 3.6)."""
        if self.model is None:
            return None
        return self.model.ObjBound

    @property
    def gap(self) -> float | None:
        """Relative optimality gap |ObjVal - ObjBound| / |ObjVal|. 0.0 once solved to
        proven optimality; None when there is no incumbent to compare against."""
        upper = self.objective_value
        lower = self.best_bound
        if upper is None or lower is None:
            return None
        if upper == 0:
            return 0.0 if lower == 0 else float("inf")
        return abs(upper - lower) / abs(upper)

    @property
    def is_optimal(self) -> bool:
        """True iff Gurobi proved global optimality (status == GRB.OPTIMAL) before the
        time limit, i.e. gap == 0. False when the solve was cut off by the time limit
        (or otherwise ended) without a proof of optimality."""
        return self.status == GRB.OPTIMAL

    def extract_assignment(self) -> dict[int, list[int]]:
        """Reads x_{p,r,d} == 1 back into the assignment format used by Solution
        (patient_id -> ordered list of room_id per day of stay). Requires a binary
        (non-relaxed) solve with an incumbent solution."""
        if self.model is None or self.model.SolCount == 0:
            raise RuntimeError("No incumbent solution available to extract.")

        assignment: dict[int, list[int]] = {}
        for pid, patient in self.data_manager.patients.items():
            room_by_day: dict[int, int] = {}
            for r in self.data_manager.rooms:
                for d in patient.stay_days:
                    var = self.x.get((pid, r, d))
                    if var is not None and var.X > 0.5:
                        room_by_day[d] = r
            assignment[pid] = [room_by_day[d] for d in patient.stay_days]
        return assignment
