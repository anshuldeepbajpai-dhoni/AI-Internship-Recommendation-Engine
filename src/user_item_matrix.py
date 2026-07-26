import pandas as pd


def create_user_item_matrix(
    interactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create student x internship rating matrix.

    Rows    -> students
    Columns -> internships
    Values  -> ratings
    Missing -> 0
    """

    matrix = interactions.pivot_table(
        index="student_id",
        columns="internship_id",
        values="rating",
        aggfunc="mean",
        fill_value=0,
    )

    return matrix