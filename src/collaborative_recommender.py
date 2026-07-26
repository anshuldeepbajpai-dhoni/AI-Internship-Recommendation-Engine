import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeRecommender:
    """
    User-based collaborative filtering recommender.

    Recommendations are generated from the preferences of students
    whose internship-rating patterns are similar to the target student.
    """

    def __init__(
        self,
        user_item_matrix: pd.DataFrame,
        internships: pd.DataFrame,
    ):
        if user_item_matrix.empty:
            raise ValueError("User-item matrix cannot be empty.")

        self.user_item_matrix = user_item_matrix
        self.internships = internships

        self.similarity_matrix = self._calculate_similarity()


    def _calculate_similarity(self) -> pd.DataFrame:
        """
        Calculate cosine similarity between all students.
        """

        similarities = cosine_similarity(
            self.user_item_matrix.values
        )

        return pd.DataFrame(
            similarities,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index,
        )


    def get_similar_students(
        self,
        student_id: str,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        Return the most similar students to a target student.
        """

        student_id = student_id.strip().lower()

        if student_id not in self.similarity_matrix.index:
            raise ValueError(
                f"Student '{student_id}' not found."
            )

        similarities = (
            self.similarity_matrix
            .loc[student_id]
            .drop(student_id)
            .sort_values(ascending=False)
        )

        # Ignore students with no meaningful similarity.
        similarities = similarities[
            similarities > 0
        ]

        result = similarities.head(top_n).reset_index()

        result.columns = [
            "student_id",
            "similarity_score",
        ]

        return result


    def recommend(
        self,
        student_id: str,
        top_n: int = 5,
        neighbor_count: int = 5,
    ) -> pd.DataFrame:
        """
        Generate Top-N internship recommendations using
        similarity-weighted user-based collaborative filtering.
        """

        student_id = student_id.strip().lower()

        if student_id not in self.user_item_matrix.index:
            raise ValueError(
                f"Student '{student_id}' not found."
            )

        target_ratings = self.user_item_matrix.loc[student_id]

        similar_students = self.get_similar_students(
            student_id,
            top_n=neighbor_count,
        )

        if similar_students.empty:
            return pd.DataFrame(
                columns=[
                    "internship_id",
                    "title",
                    "company",
                    "domain",
                    "predicted_score",
                ]
            )

        recommendations = {}

        for _, student in similar_students.iterrows():

            similar_student_id = student["student_id"]
            similarity = student["similarity_score"]

            ratings = self.user_item_matrix.loc[
                similar_student_id
            ]

            for internship_id, rating in ratings.items():

                # Recommend only internships that:
                # 1. target student hasn't rated
                # 2. similar student has rated
                if (
                    target_ratings[internship_id] == 0
                    and rating > 0
                ):

                    if internship_id not in recommendations:
                        recommendations[internship_id] = {
                            "weighted_sum": 0.0,
                            "similarity_sum": 0.0,
                        }

                    recommendations[internship_id][
                        "weighted_sum"
                    ] += rating * similarity

                    recommendations[internship_id][
                        "similarity_sum"
                    ] += similarity

        predicted_ratings = []

        for internship_id, values in recommendations.items():

            similarity_sum = values["similarity_sum"]

            if similarity_sum == 0:
                continue

            predicted_score = (
                values["weighted_sum"]
                / similarity_sum
            )

            predicted_ratings.append(
                {
                    "internship_id": internship_id,
                    "predicted_score": predicted_score,
                }
            )

        if not predicted_ratings:
            return pd.DataFrame(
                columns=[
                    "internship_id",
                    "title",
                    "company",
                    "domain",
                    "predicted_score",
                ]
            )

        result = pd.DataFrame(predicted_ratings)

        result = result.sort_values(
            by="predicted_score",
            ascending=False,
        ).head(top_n)

        # Attach internship information
        result = result.merge(
            self.internships[
                [
                    "internship_id",
                    "title",
                    "company",
                    "domain",
                ]
            ],
            on="internship_id",
            how="left",
        )

        result["predicted_score"] = (
            result["predicted_score"].round(2)
        )

        return result[
            [
                "internship_id",
                "title",
                "company",
                "domain",
                "predicted_score",
            ]
        ]