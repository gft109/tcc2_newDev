"""Generates the synthetic instances: N_SETS_PER_INSTANCE seeded room.csv/patient.csv
sets per instance size, under data_base/<instance>/set_<NN>/."""

import os

import numpy as np
import pandas as pd

SPECIALTIES = ["GENERAL", "SURGERY", "CARDIOLOGY", "ORTHOPEDICS"]
SPECIALTY_WEIGHTS = [0.4, 0.2, 0.2, 0.2]
GENDERS = ["F", "M"]

ROOM_CAPACITY = 3
PATIENTS_PER_ROOM = 10

LOS_MEAN = 4.0
LOS_STD = 1.0
LOS_MIN = 1

TARGET_OCCUPANCY = 0.9
HORIZON_BUFFER_DAYS = 2

INSTANCE_SIZES = {
    "pequeno": {"n_rooms": 10, "seed": 42},
    "medio": {"n_rooms": 40, "seed": 43},
    "grande": {"n_rooms": 160, "seed": 44},
    "muito_grande": {"n_rooms": 640, "seed": 45},
}

N_SETS_PER_INSTANCE = 10


def _set_seed(base_seed: int, set_number: int) -> int:
    """Unique seed per (instance, set): base_seed * 100 + set_number."""
    return base_seed * 100 + set_number

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_BASE_DIR = os.path.join(PROJECT_ROOT, "data_base")


class DataGenerator:
    def __init__(self, seed: int):
        self._rng = np.random.default_rng(seed)

    def _generate_rooms(self, n_rooms: int) -> pd.DataFrame:
        specialties = self._rng.choice(SPECIALTIES, size=n_rooms, p=SPECIALTY_WEIGHTS)
        return pd.DataFrame({
            "room_id": np.arange(1, n_rooms + 1),
            "capacity": ROOM_CAPACITY,
            "specialty": specialties,
        })

    def _sample_los(self, n_patients: int) -> np.ndarray:
        raw = self._rng.normal(LOS_MEAN, LOS_STD, size=n_patients)
        return np.maximum(LOS_MIN, np.round(raw)).astype(int)

    def _schedule_admissions(self, los: np.ndarray, total_capacity: int) -> np.ndarray:
        """First-fit decreasing: longest stays first, each admitted on the earliest day it fits.
        The hospital is never over capacity on any day, so a feasible solution always exists."""
        n_patients = len(los)
        target_daily_census = TARGET_OCCUPANCY * total_capacity
        horizon = int(np.ceil(n_patients * los.mean() / target_daily_census)) + HORIZON_BUFFER_DAYS

        remaining = np.full(horizon, total_capacity, dtype=int)
        admission_days = np.empty(n_patients, dtype=int)

        tie_break = self._rng.permutation(n_patients)
        order = sorted(range(n_patients), key=lambda i: (-los[i], tie_break[i]))

        for idx in order:
            stay = int(los[idx])
            placed = False

            for start in range(1, len(remaining) - stay + 2):
                window = remaining[start - 1: start - 1 + stay]
                if np.all(window > 0):
                    window -= 1
                    admission_days[idx] = start
                    placed = True
                    break

            if placed:
                continue

            extra = stay + HORIZON_BUFFER_DAYS
            remaining = np.concatenate([remaining, np.full(extra, total_capacity, dtype=int)])
            start = len(remaining) - extra + 1
            remaining[start - 1: start - 1 + stay] -= 1
            admission_days[idx] = start

        return admission_days

    def _generate_patients(self, n_patients: int, total_capacity: int) -> pd.DataFrame:
        genders = self._rng.choice(GENDERS, size=n_patients)
        required_specialties = self._rng.choice(SPECIALTIES, size=n_patients, p=SPECIALTY_WEIGHTS)
        los = self._sample_los(n_patients)
        admission_days = self._schedule_admissions(los, total_capacity)

        return pd.DataFrame({
            "patient_id": np.arange(1, n_patients + 1),
            "gender": genders,
            "admission_day": admission_days,
            "los": los,
            "required_specialty": required_specialties,
        })

    def generate_instance(self, name: str, n_rooms: int, set_number: int) -> None:
        n_patients = n_rooms * PATIENTS_PER_ROOM
        total_capacity = n_rooms * ROOM_CAPACITY

        rooms_df = self._generate_rooms(n_rooms)
        patients_df = self._generate_patients(n_patients, total_capacity)

        set_dir = os.path.join(DATA_BASE_DIR, name, f"set_{set_number:02d}")
        os.makedirs(set_dir, exist_ok=True)
        rooms_df.to_csv(os.path.join(set_dir, "room.csv"), index=False)
        patients_df.to_csv(os.path.join(set_dir, "patient.csv"), index=False)

        horizon = int((patients_df["admission_day"] + patients_df["los"] - 1).max())
        print(
            f"{name}/set_{set_number:02d}: {n_rooms} rooms, {n_patients} patients, "
            f"horizon={horizon} days"
        )


def main():
    for name, config in INSTANCE_SIZES.items():
        for set_number in range(1, N_SETS_PER_INSTANCE + 1):
            generator = DataGenerator(seed=_set_seed(config["seed"], set_number))
            generator.generate_instance(name, config["n_rooms"], set_number)


if __name__ == "__main__":
    main()
