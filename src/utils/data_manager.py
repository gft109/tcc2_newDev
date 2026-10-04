"""Loads an instance set (room.csv, patient.csv) into the structures used by both solvers."""

import os
from dataclasses import dataclass

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_BASE_DIR = os.path.join(PROJECT_ROOT, "data_base")


@dataclass(frozen=True)
class Room:
    room_id: int
    capacity: int
    specialty: str


@dataclass(frozen=True)
class Patient:
    patient_id: int
    gender: str
    admission_day: int
    los: int
    required_specialty: str

    @property
    def discharge_day(self) -> int:
        return self.admission_day + self.los - 1

    @property
    def stay_days(self) -> range:
        return range(self.admission_day, self.discharge_day + 1)


class DataManager:
    def __init__(self, instance_name: str):
        """instance_name is a path under data_base/, e.g. "pequeno/set_01"."""
        self.instance_name = instance_name
        self.rooms: dict[int, Room] = {}
        self.patients: dict[int, Patient] = {}
        self.horizon: int = 0
        self.patients_by_day: dict[int, list[int]] = {}
        self._load()

    def _load(self) -> None:
        instance_dir = os.path.join(DATA_BASE_DIR, self.instance_name)

        rooms_df = pd.read_csv(os.path.join(instance_dir, "room.csv"))
        self.rooms = {
            int(row.room_id): Room(int(row.room_id), int(row.capacity), row.specialty)
            for row in rooms_df.itertuples(index=False)
        }

        patients_df = pd.read_csv(os.path.join(instance_dir, "patient.csv"))
        self.patients = {
            int(row.patient_id): Patient(
                int(row.patient_id),
                row.gender,
                int(row.admission_day),
                int(row.los),
                row.required_specialty,
            )
            for row in patients_df.itertuples(index=False)
        }

        self.horizon = max(patient.discharge_day for patient in self.patients.values())

        self.patients_by_day = {}
        for patient_id, patient in self.patients.items():
            for day in patient.stay_days:
                self.patients_by_day.setdefault(day, []).append(patient_id)
