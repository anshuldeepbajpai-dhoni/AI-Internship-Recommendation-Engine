import pandas as pd


STUDENT_COLUMNS = {
    "student_id",
    "name",
    "skills",
    "interests",
    "domain",
    "experience_level",
}

INTERNSHIP_COLUMNS = {
    "internship_id",
    "title",
    "company",
    "required_skills",
    "domain",
    "duration",
    "experience_level",
}

INTERACTION_COLUMNS = {
    "student_id",
    "internship_id",
    "rating",
}


def check_required_columns(
    df: pd.DataFrame,
    required_columns: set[str],
    dataset_name: str,
) -> None:
    """Raise an error when required columns are missing."""
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{dataset_name} is missing columns: {sorted(missing)}"
        )


def validate_students(students: pd.DataFrame) -> None:
    check_required_columns(
        students,
        STUDENT_COLUMNS,
        "Students dataset",
    )

    if students.empty:
        raise ValueError("Students dataset is empty.")

    if students["student_id"].isna().any():
        raise ValueError("Student ID cannot be missing.")

    if students["student_id"].duplicated().any():
        raise ValueError("Duplicate student IDs detected.")


def validate_internships(internships: pd.DataFrame) -> None:
    check_required_columns(
        internships,
        INTERNSHIP_COLUMNS,
        "Internships dataset",
    )

    if internships.empty:
        raise ValueError("Internships dataset is empty.")

    if internships["internship_id"].isna().any():
        raise ValueError("Internship ID cannot be missing.")

    if internships["internship_id"].duplicated().any():
        raise ValueError("Duplicate internship IDs detected.")


def validate_interactions(
    interactions: pd.DataFrame,
    students: pd.DataFrame,
    internships: pd.DataFrame,
) -> None:

    check_required_columns(
        interactions,
        INTERACTION_COLUMNS,
        "Interactions dataset",
    )

    if interactions.empty:
        raise ValueError("Interactions dataset is empty.")

    if interactions["rating"].isna().any():
        raise ValueError("Missing ratings detected.")

    if not interactions["rating"].between(1, 5).all():
        raise ValueError("Ratings must be between 1 and 5.")

    valid_students = set(students["student_id"])
    valid_internships = set(internships["internship_id"])

    invalid_students = (
        set(interactions["student_id"]) - valid_students
    )

    invalid_internships = (
        set(interactions["internship_id"]) - valid_internships
    )

    if invalid_students:
        raise ValueError(
            f"Unknown student IDs: {sorted(invalid_students)}"
        )

    if invalid_internships:
        raise ValueError(
            f"Unknown internship IDs: {sorted(invalid_internships)}"
        )