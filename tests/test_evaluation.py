import pytest

from src.evaluation import RecommendationEvaluator


def test_precision_at_k():

    recommended = [
        "i001",
        "i002",
        "i003",
        "i004",
        "i005",
    ]

    relevant = {
        "i001",
        "i003",
    }

    precision = (
        RecommendationEvaluator.precision_at_k(
            recommended_ids=recommended,
            relevant_ids=relevant,
            k=5,
        )
    )

    assert precision == pytest.approx(
        0.4
    )


def test_recall_at_k():

    recommended = [
        "i001",
        "i002",
        "i003",
    ]

    relevant = {
        "i001",
        "i003",
        "i007",
        "i009",
    }

    recall = (
        RecommendationEvaluator.recall_at_k(
            recommended_ids=recommended,
            relevant_ids=relevant,
            k=3,
        )
    )

    assert recall == pytest.approx(
        0.5
    )


def test_mae():

    actual = [
        5,
        4,
        3,
    ]

    predicted = [
        4.5,
        4.0,
        3.5,
    ]

    score = RecommendationEvaluator.mae(
        actual,
        predicted,
    )

    assert score == pytest.approx(
        1 / 3
    )


def test_rmse():

    actual = [
        5,
        4,
        3,
    ]

    predicted = [
        5,
        4,
        3,
    ]

    score = RecommendationEvaluator.rmse(
        actual,
        predicted,
    )

    assert score == pytest.approx(
        0.0
    )


def test_invalid_k():

    with pytest.raises(ValueError):

        RecommendationEvaluator.precision_at_k(
            ["i001"],
            {"i001"},
            k=0,
        )