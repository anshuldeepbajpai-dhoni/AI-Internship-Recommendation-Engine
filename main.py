from src.data_loader import load_all_data
from src.preprocessing import (
    preprocess_students,
    preprocess_internships,
    preprocess_interactions,
)
from src.validation import (
    validate_students,
    validate_internships,
    validate_interactions,
)
import pandas as pd
from src.user_item_matrix import create_user_item_matrix
from src.collaborative_recommender import CollaborativeRecommender
from src.content_recommender import ContentBasedRecommender
from src.hybrid_recommender import HybridRecommender
from src.evaluation import RecommendationEvaluator


def main():
    try:
        print("=" * 60)
        print("AI INTERNSHIP RECOMMENDATION ENGINE")
        print("=" * 60)

        # =====================================================
        # 1. LOAD DATA
        # =====================================================

        students, internships, interactions = load_all_data()

        print("\n[1] Data loaded successfully")
        print(f"Students:     {len(students)}")
        print(f"Internships:  {len(internships)}")
        print(f"Interactions: {len(interactions)}")

        # =====================================================
        # 2. VALIDATE RAW DATA
        # =====================================================

        validate_students(students)
        validate_internships(internships)

        print("\n[2] Basic validation passed")

        # =====================================================
        # 3. PREPROCESS DATA
        # =====================================================

        students = preprocess_students(students)
        internships = preprocess_internships(internships)
        interactions = preprocess_interactions(interactions)

        print("\n[3] Preprocessing completed")

        # =====================================================
        # 4. VALIDATE INTERACTIONS
        # =====================================================

        validate_interactions(
            interactions,
            students,
            internships,
        )

        print("\n[4] Interaction validation passed")

        # =====================================================
        # 5. CREATE USER-ITEM MATRIX
        # =====================================================

        user_item_matrix = create_user_item_matrix(
            interactions
        )

        print("\n[5] User-Item Matrix created")
        print("Shape:", user_item_matrix.shape)

        # =====================================================
        # 6. CREATE COLLABORATIVE RECOMMENDER
        # =====================================================

        recommender = CollaborativeRecommender(
            user_item_matrix=user_item_matrix,
            internships=internships,
        )

        print("\n[6] Collaborative recommender initialized")

        # Test similar students for S001
        print("\nSimilar students for S001:")
        print("-" * 60)

        similar_students = recommender.get_similar_students(
            student_id="s001",
            top_n=5,
        )

        print(similar_students.to_string(index=False))

        # =====================================================
        # 7. TEST RECOMMENDATIONS FOR MULTIPLE STUDENTS
        # =====================================================

        test_students = [
            "s001",
            "s003",
            "s006",
            "s008",
            "s010",
        ]

        for student_id in test_students:

            print("\n" + "=" * 70)
            print(
                f"Recommendations for {student_id.upper()}"
            )
            print("=" * 70)

            recommendations = recommender.recommend(
                student_id=student_id,
                top_n=3,
            )

            if recommendations.empty:
                print("No recommendations available.")
            else:
                print(
                    recommendations.to_string(
                        index=False
                    )
                )

        # =====================================================
        # 8. CONTENT-BASED RECOMMENDER
        # =====================================================

        content_recommender = ContentBasedRecommender(
            students=students,
            internships=internships,
        )

        print("\n[8] Content-Based Recommender initialized")

        # =====================================================
        # 9. CONTENT-BASED RECOMMENDATIONS
        # =====================================================

        student_id = "s001"

        print("\n" + "=" * 70)
        print(
            f"Content-Based Recommendations for "
            f"{student_id.upper()}"
        )
        print("=" * 70)

        content_recommendations = (
            content_recommender.recommend(
                student_id=student_id,
                top_n=5,
            )
        )

        print(
            content_recommendations.to_string(
                index=False
            )
        )

        # =====================================================
        # 10. TEST DATA ANALYTICS STUDENT
        # =====================================================

        student_id = "s006"

        print("\n" + "=" * 70)
        print(
            f"Content-Based Recommendations for "
            f"{student_id.upper()}"
        )
        print("=" * 70)

        analytics_recommendations = (
            content_recommender.recommend(
                student_id=student_id,
                top_n=5,
            )
        )

        print(
            analytics_recommendations.to_string(
                index=False
            )
        )

        # =====================================================
        # 11. COLD-START STUDENT
        # =====================================================

        print("\n" + "=" * 70)
        print("Recommendations for NEW STUDENT")
        print("=" * 70)

        new_student_recommendations = (
            content_recommender.recommend_new_student(
                skills=[
                    "python",
                    "pandas",
                    "numpy",
                    "sql",
                    "machine learning",
                ],
                interests="data science|ai|analytics",
                domain="AI",
                experience_level="intermediate",
                top_n=5,
            )
        )

        print(
            new_student_recommendations.to_string(
                index=False
            )
        )

        # =====================================================
        # 12. HYBRID RECOMMENDER
        # =====================================================

        hybrid_recommender = HybridRecommender(
            students=students,
            internships=internships,
            user_item_matrix=user_item_matrix,
            collaborative_weight=0.6,
            content_weight=0.4,
        )

        # =====================================================
        # 13. HYBRID RECOMMENDATIONS
        # =====================================================

        student_id = "s001"

        print("\n" + "=" * 70)
        print(
            f"Hybrid Recommendations for "
            f"{student_id.upper()}"
        )
        print("=" * 70)

        hybrid_recommendations = (
            hybrid_recommender.recommend(
                student_id=student_id,
                top_n=5,
            )
        )

        if hybrid_recommendations.empty:
            print("No hybrid recommendations available.")
        else:
            print(
                hybrid_recommendations.to_string(
                    index=False
                )
            )


        # =====================================================
        # 14. HYBRID TEST — DATA ANALYTICS STUDENT
        # =====================================================

        student_id = "s006"

        print("\n" + "=" * 70)
        print(
            f"Hybrid Recommendations for "
            f"{student_id.upper()}"
        )
        print("=" * 70)

        recommendations = hybrid_recommender.recommend(
            student_id=student_id,
            top_n=5,
        )

        if recommendations.empty:
            print("No hybrid recommendations available.")
        else:
            print(
                recommendations.to_string(
                    index=False
                )
            )

        print("\n[12] Hybrid Recommender initialized")

        # =====================================================
        # 15. HYBRID COLD-START TEST
        # =====================================================

        print("\n" + "=" * 70)
        print("Hybrid Recommendations for NEW STUDENT")
        print("=" * 70)

        cold_start_recommendations = (
            hybrid_recommender.recommend_new_student(
                skills=[
                    "python",
                    "machine learning",
                    "pandas",
                    "scikit-learn",
                ],
                interests="ai|data science",
                domain="AI",
                experience_level="intermediate",
                top_n=5,
            )
        )

        print(
            cold_start_recommendations.to_string(
                index=False
            )
        )

        # =====================================================
        # 16. LEAVE-ONE-OUT EVALUATION
        # =====================================================

        print("\n" + "=" * 70)
        print("HYBRID RECOMMENDER EVALUATION")
        print("=" * 70)

        evaluator = RecommendationEvaluator()

        evaluation_results = []

        # Students need at least 2 interactions:
        # one can be hidden for testing while others remain for training.
        student_counts = (
            interactions
            .groupby("student_id")
            .size()
        )

        eligible_students = student_counts[
            student_counts >= 2
        ].index


        for evaluation_student_id in eligible_students:

            student_interactions = interactions[
                interactions["student_id"]
                == evaluation_student_id
            ]

            # Prefer holding out a highly-rated internship.
            relevant_interactions = student_interactions[
                student_interactions["rating"] >= 4
            ]

            if relevant_interactions.empty:
                continue

            # Use the highest-rated interaction as test ground truth.
            test_row = relevant_interactions.sort_values(
                by="rating",
                ascending=False,
            ).iloc[0]

            test_internship_id = test_row[
                "internship_id"
            ]

            # Remove the test interaction from training data.
            train_interactions = interactions.drop(
                index=test_row.name
            )

            train_matrix = create_user_item_matrix(
                train_interactions
            )

            evaluation_recommender = HybridRecommender(
                students=students,
                internships=internships,
                user_item_matrix=train_matrix,
                collaborative_weight=0.6,
                content_weight=0.4,
            )

            recommendations = (
                evaluation_recommender.recommend(
                    student_id=evaluation_student_id,
                    top_n=5,
                )
            )

            recommended_ids = recommendations[
                "internship_id"
            ].tolist()

            relevant_ids = {
                test_internship_id
            }

            precision = evaluator.precision_at_k(
                recommended_ids=recommended_ids,
                relevant_ids=relevant_ids,
                k=5,
            )

            recall = evaluator.recall_at_k(
                recommended_ids=recommended_ids,
                relevant_ids=relevant_ids,
                k=5,
            )

            evaluation_results.append(
                {
                    "student_id": evaluation_student_id,
                    "held_out_internship": test_internship_id,
                    "precision@5": precision,
                    "recall@5": recall,
                }
            )

        # =====================================================
        # 17. EVALUATION SUMMARY
        # =====================================================

        if evaluation_results:

            evaluation_df = pd.DataFrame(
                evaluation_results
            )

            print("\nPer-student results:")
            print(
                evaluation_df.to_string(
                    index=False
                )
            )

            mean_precision = evaluation_df[
                "precision@5"
            ].mean()

            mean_recall = evaluation_df[
                "recall@5"
            ].mean()

            print("\n" + "-" * 50)

            print(
                f"Mean Precision@5: "
                f"{mean_precision:.4f}"
            )

            print(
                f"Mean Recall@5:    "
                f"{mean_recall:.4f}"
            )

        else:
            print(
                "No students available for evaluation."
            )

        # =====================================================
        # 18. RATING ERROR METRICS
        # =====================================================

        actual_ratings = [
            5.0,
            4.0,
            3.0,
            5.0,
        ]

        predicted_ratings = [
            4.7,
            4.3,
            3.5,
            4.8,
        ]

        mae_score = evaluator.mae(
            actual=actual_ratings,
            predicted=predicted_ratings,
        )

        rmse_score = evaluator.rmse(
            actual=actual_ratings,
            predicted=predicted_ratings,
        )

        print("\nRating Prediction Metrics")
        print("-" * 50)

        print(
            f"MAE:  {mae_score:.4f}"
        )

        print(
            f"RMSE: {rmse_score:.4f}"
        )

        print("\n" + "=" * 60)
        print("PHASE 6 COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except Exception as exc:
        print(f"\nERROR: {exc}")


if __name__ == "__main__":
    main()