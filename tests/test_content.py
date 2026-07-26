from src.data_loader import load_all_data
from src.preprocessing import (
    preprocess_students,
    preprocess_internships,
)
from src.content_recommender import ContentBasedRecommender


def test_content_recommendations():
    students, internships, _ = load_all_data()

    students = preprocess_students(
        students
    )

    internships = preprocess_internships(
        internships
    )

    recommender = ContentBasedRecommender(
        students=students,
        internships=internships,
    )

    recommendations = recommender.recommend(
        student_id="s001",
        top_n=5,
    )

    assert not recommendations.empty

    assert len(recommendations) <= 5

    assert (
        "similarity_score"
        in recommendations.columns
    )

    assert (
        "match_percentage"
        in recommendations.columns
    )

    assert recommendations[
        "similarity_score"
    ].between(0, 1).all()


def test_cold_start_recommendations():
    students, internships, _ = load_all_data()

    students = preprocess_students(
        students
    )

    internships = preprocess_internships(
        internships
    )

    recommender = ContentBasedRecommender(
        students=students,
        internships=internships,
    )

    recommendations = (
        recommender.recommend_new_student(
            skills=[
                "python",
                "machine learning",
                "pandas",
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
        "match_percentage"
        in recommendations.columns
    )