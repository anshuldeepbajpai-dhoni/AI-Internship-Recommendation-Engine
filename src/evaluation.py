from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


class RecommendationEvaluator:
    """
    Evaluation utilities for internship recommendations.

    Supported metrics:
    - Precision@K
    - Recall@K
    - MAE
    - RMSE
    """

    @staticmethod
    def precision_at_k(
        recommended_ids: Iterable[str],
        relevant_ids: Iterable[str],
        k: int = 5,
    ) -> float:
        """
        Precision@K =
        relevant recommended items / K
        """

        if k <= 0:
            raise ValueError("k must be greater than 0.")

        recommended = list(recommended_ids)[:k]
        relevant = set(relevant_ids)

        if not recommended:
            return 0.0

        hits = len(set(recommended) & relevant)

        # We use the actual returned recommendation count when
        # fewer than K candidates are available.
        denominator = len(recommended)

        return hits / denominator

    @staticmethod
    def recall_at_k(
        recommended_ids: Iterable[str],
        relevant_ids: Iterable[str],
        k: int = 5,
    ) -> float:
        """
        Recall@K =
        relevant recommended items / all relevant items
        """

        if k <= 0:
            raise ValueError("k must be greater than 0.")

        recommended = set(
            list(recommended_ids)[:k]
        )

        relevant = set(relevant_ids)

        if not relevant:
            return 0.0

        hits = len(
            recommended & relevant
        )

        return hits / len(relevant)

    @staticmethod
    def mae(
        actual: Iterable[float],
        predicted: Iterable[float],
    ) -> float:
        """Mean Absolute Error."""

        actual = list(actual)
        predicted = list(predicted)

        RecommendationEvaluator._validate_rating_arrays(
            actual,
            predicted,
        )

        return float(
            mean_absolute_error(
                actual,
                predicted,
            )
        )

    @staticmethod
    def rmse(
        actual: Iterable[float],
        predicted: Iterable[float],
    ) -> float:
        """Root Mean Squared Error."""

        actual = list(actual)
        predicted = list(predicted)

        RecommendationEvaluator._validate_rating_arrays(
            actual,
            predicted,
        )

        return float(
            np.sqrt(
                mean_squared_error(
                    actual,
                    predicted,
                )
            )
        )

    @staticmethod
    def _validate_rating_arrays(
        actual: list[float],
        predicted: list[float],
    ) -> None:

        if not actual or not predicted:
            raise ValueError(
                "Actual and predicted ratings cannot be empty."
            )

        if len(actual) != len(predicted):
            raise ValueError(
                "Actual and predicted ratings must have equal length."
            )


def build_relevant_items(
    interactions: pd.DataFrame,
    student_id: str,
    rating_threshold: float = 4.0,
) -> set[str]:
    """
    Treat internships rated >= threshold as relevant.
    """

    student_id = student_id.strip().lower()

    student_data = interactions[
        interactions["student_id"] == student_id
    ]

    relevant = student_data[
        student_data["rating"] >= rating_threshold
    ]

    return set(
        relevant["internship_id"]
    )