from src.data_loader import load_all_data
from src.preprocessing import (
    preprocess_students,
    preprocess_internships,
    preprocess_interactions,
)
from src.user_item_matrix import create_user_item_matrix
from src.collaborative_recommender import CollaborativeRecommender


def test_collaborative_recommendations():
    students, internships, interactions = load_all_data()

    students = preprocess_students(students)

    internships = preprocess_internships(
        internships
    )

    interactions = preprocess_interactions(
        interactions
    )

    matrix = create_user_item_matrix(
        interactions
    )

    recommender = CollaborativeRecommender(
        matrix,
        internships,
    )

    recommendations = recommender.recommend(
        "s001",
        top_n=3,
    )

    # Maximum 3 recommendations
    assert len(recommendations) <= 3

    # Output must contain predicted score
    assert (
        "predicted_score"
        in recommendations.columns
    )

    # S001 has already interacted with these internships
    already_seen = {
        "i001",
        "i003",
        "i011",
    }

    recommended_ids = set(
        recommendations["internship_id"]
    )

    assert not recommended_ids.intersection(
        already_seen
    )