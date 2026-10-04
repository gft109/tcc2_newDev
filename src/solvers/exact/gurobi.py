"""Exact ILP model of the PBA (TCC Section 3.2, Eqs. 3.8-3.15), solved with Gurobi."""

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

        # Eq. 3.8
        specialty_cost = gp.quicksum(
            WEIGHTS["W_SPEC"] * var
            for (pid, r, _d), var in x.items()
            if rooms[r].specialty != patients[pid].required_specialty
        )
        transfer_cost = WEIGHTS["W_TRANSF"] * gp.quicksum(t.values())
        gender_cost = WEIGHTS["W_GEN"] * gp.quicksum(b.values())
        model.setObjective(specialty_cost + transfer_cost + gender_cost, GRB.MINIMIZE)

        # Eq. 3.9
        for pid, patient in patients.items():
            for d in patient.stay_days:
                model.addConstr(gp.quicksum(x[pid, r, d] for r in rooms) == 1, name=f"assign_{pid}_{d}")

        # Eq. 3.10
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

        # Eq. 3.11
        for pid, patient in patients.items():
            stay_days = list(patient.stay_days)
            for r in rooms:
                for d in stay_days[:-1]:
                    model.addConstr(
                        t[pid, r, d] >= x[pid, r, d] - x[pid, r, d + 1], name=f"transf_{pid}_{r}_{d}"
                    )

        # Eqs. 3.12-3.13
        for pid, patient in patients.items():
            gender_flags = f if patient.gender == "F" else m
            for r in rooms:
                for d in patient.stay_days:
                    model.addConstr(gender_flags[r, d] >= x[pid, r, d], name=f"gender_{pid}_{r}_{d}")

        # Eq. 3.14
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
        """Best lower bound on Z found by branch-and-bound."""
        if self.model is None:
            return None
        return self.model.ObjBound

    @property
    def gap(self) -> float | None:
        """Relative gap |UB - LB| / |UB|; None without an incumbent."""
        upper = self.objective_value
        lower = self.best_bound
        if upper is None or lower is None:
            return None
        if upper == 0:
            return 0.0 if lower == 0 else float("inf")
        return abs(upper - lower) / abs(upper)

    @property
    def is_optimal(self) -> bool:
        return self.status == GRB.OPTIMAL

    def extract_assignment(self) -> dict[int, list[int]]:
        """Converts x_{p,r,d} into Solution's format: patient_id -> room per day of stay."""
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
