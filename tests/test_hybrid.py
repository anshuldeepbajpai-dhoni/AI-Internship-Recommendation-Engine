from src.data_loader import load_all_data
from src.preprocessing import (
    preprocess_students,
    preprocess_internships,
    preprocess_interactions,
)
from src.user_item_matrix import create_user_item_matrix
from src.hybrid_recommender import HybridRecommender


def build_recommender():
    students, internships, interactions = load_all_data()

    students = preprocess_students(
        students
    )

    internships = preprocess_internships(
        internships
    )

    interactions = preprocess_interactions(
        interactions
    )

    matrix = create_user_item_matrix(
        interactions
    )

    recommender = HybridRecommender(
        students=students,
        internships=internships,
        user_item_matrix=matrix,
        collaborative_weight=0.6,
        content_weight=0.4,
    )

    return recommender


def test_hybrid_recommendations():
    recommender = build_recommender()

    recommendations = recommender.recommend(
        student_id="s001",
        top_n=5,
    )

    assert not recommendations.empty

    assert len(recommendations) <= 5

    assert (
        "collaborative_score"
        in recommendations.columns
    )

    assert (
        "content_score"
        in recommendations.columns
    )

    assert (
        "hybrid_score"
        in recommendations.columns
    )

    assert recommendations[
        "hybrid_score"
    ].between(0, 1).all()


def test_hybrid_excludes_seen_internships():
    recommender = build_recommender()

    recommendations = recommender.recommend(
        student_id="s001",
        top_n=10,
    )

    already_seen = {
        "i001",
        "i003",
        "i011",
    }

    recommended = set(
        recommendations["internship_id"]
    )

    assert not recommended.intersection(
        already_seen
    )


def test_hybrid_cold_start():
    recommender = build_recommender()

    recommendations = (
        recommender.recommend_new_student(
            skills=[
                "python",
                "pandas",
                "machine learning",
            ],
            interests="ai|data science",
            domain="AI",
            experience_level="intermediate",
            top_n=3,
        )
    )

    assert not recommendations.empty

    assert len(recommendations) <= 3

    assert (
        recommendations[
            "collaborative_score"
        ] == 0
    ).all()

    assert recommendations[
        "hybrid_score"
    ].between(0, 1).all()


def test_weights_validation():
    recommender_created = False

    try:
        students, internships, interactions = load_all_data()

        students = preprocess_students(students)
        internships = preprocess_internships(internships)
        interactions = preprocess_interactions(interactions)

        matrix = create_user_item_matrix(interactions)

        HybridRecommender(
            students=students,
            internships=internships,
            user_item_matrix=matrix,
            collaborative_weight=0.8,
            content_weight=0.8,
        )

        recommender_created = True

    except ValueError:
        pass

    assert recommender_created is False