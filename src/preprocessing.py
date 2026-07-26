import pandas as pd


def normalize_text(value: str) -> str:
    """Convert text to lowercase and remove unnecessary spaces."""
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def parse_skills(skills: str) -> list[str]:
    """Convert pipe-separated skills into a cleaned Python list."""
    if pd.isna(skills):
        return []

    return [
        skill.strip().lower()
        for skill in str(skills).split("|")
        if skill.strip()
    ]


def preprocess_students(students: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess student profile data."""
    df = students.copy()

    df = df.drop_duplicates(subset=["student_id"])

    text_columns = [
        "student_id",
        "name",
        "interests",
        "domain",
        "experience_level",
    ]

    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].apply(normalize_text)

    df["skills_list"] = df["skills"].apply(parse_skills)

    return df


def preprocess_internships(internships: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess internship data."""
    df = internships.copy()

    df = df.drop_duplicates(subset=["internship_id"])

    text_columns = [
        "internship_id",
        "title",
        "company",
        "domain",
        "duration",
        "experience_level",
    ]

    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].apply(normalize_text)

    df["required_skills_list"] = df["required_skills"].apply(parse_skills)

    return df


def preprocess_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    """Clean rating/interactions data."""
    df = interactions.copy()

    df["student_id"] = df["student_id"].apply(normalize_text)
    df["internship_id"] = df["internship_id"].apply(normalize_text)

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    df = df.dropna(subset=["student_id", "internship_id", "rating"])

    df = df[df["rating"].between(1, 5)]

    # If the same student rated the same internship more than once,
    # keep the most recent row available in the dataset.
    df = df.drop_duplicates(
        subset=["student_id", "internship_id"],
        keep="last",
    )

    return df