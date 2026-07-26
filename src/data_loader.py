from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def load_students() -> pd.DataFrame:
    path = DATA_DIR / "students.csv"

    if not path.exists():
        raise FileNotFoundError(f"Student dataset not found: {path}")

    return pd.read_csv(path)


def load_internships() -> pd.DataFrame:
    path = DATA_DIR / "internships.csv"

    if not path.exists():
        raise FileNotFoundError(f"Internship dataset not found: {path}")

    return pd.read_csv(path)


def load_interactions() -> pd.DataFrame:
    path = DATA_DIR / "interactions.csv"

    if not path.exists():
        raise FileNotFoundError(f"Interaction dataset not found: {path}")

    return pd.read_csv(path)


def load_all_data():
    students = load_students()
    internships = load_internships()
    interactions = load_interactions()

    return students, internships, interactions