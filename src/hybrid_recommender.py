import pandas as pd

from src.collaborative_recommender import CollaborativeRecommender
from src.content_recommender import ContentBasedRecommender


class HybridRecommender:
    """
    Combine collaborative filtering and content-based filtering.

    Final score:
        collaborative_normalized * collaborative_weight
        +
        content_score * content_weight
    """

    def __init__(
        self,
        students: pd.DataFrame,
        internships: pd.DataFrame,
        user_item_matrix: pd.DataFrame,
        collaborative_weight: float = 0.6,
        content_weight: float = 0.4,
    ):
        if not 0 <= collaborative_weight <= 1:
            raise ValueError(
                "Collaborative weight must be between 0 and 1."
            )

        if not 0 <= content_weight <= 1:
            raise ValueError(
                "Content weight must be between 0 and 1."
            )

        if abs(
            collaborative_weight + content_weight - 1.0
        ) > 1e-9:
            raise ValueError(
                "Collaborative and content weights must sum to 1."
            )

        self.students = students
        self.internships = internships
        self.user_item_matrix = user_item_matrix

        self.collaborative_weight = collaborative_weight
        self.content_weight = content_weight

        self.collaborative_recommender = CollaborativeRecommender(
            user_item_matrix=user_item_matrix,
            internships=internships,
        )

        self.content_recommender = ContentBasedRecommender(
            students=students,
            internships=internships,
        )

    @staticmethod
    def _normalize_collaborative_score(
        score: float,
    ) -> float:
        """
        Collaborative predictions use the 1-5 rating scale.

        Convert:
            1 -> 0.0
            3 -> 0.5
            5 -> 1.0
        """

        normalized = (float(score) - 1.0) / 4.0

        return max(
            0.0,
            min(1.0, normalized),
        )

    def recommend(
        self,
        student_id: str,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        Generate hybrid recommendations for an existing student.
        """

        student_id = student_id.strip().lower()

        if student_id not in self.students["student_id"].values:
            raise ValueError(
                f"Student '{student_id}' not found."
            )

        # Request the full internship catalog from both models
        # so that hybrid ranking is not restricted too early.
        candidate_count = len(self.internships)

        content_recs = self.content_recommender.recommend(
            student_id=student_id,
            top_n=candidate_count,
        )

        # Collaborative filtering only works when the student
        # has interaction history.
        if student_id in self.user_item_matrix.index:
            collaborative_recs = (
                self.collaborative_recommender.recommend(
                    student_id=student_id,
                    top_n=candidate_count,
                    neighbor_count=max(
                        5,
                        len(self.user_item_matrix) - 1,
                    ),
                )
            )
        else:
            collaborative_recs = pd.DataFrame()

        # ---------------------------------------------
        # Collaborative score lookup
        # ---------------------------------------------

        collaborative_scores = {}

        if not collaborative_recs.empty:
            for _, row in collaborative_recs.iterrows():

                collaborative_scores[
                    row["internship_id"]
                ] = self._normalize_collaborative_score(
                    row["predicted_score"]
                )

        # ---------------------------------------------
        # Already-seen internships
        # ---------------------------------------------

        seen_internships = set()

        if student_id in self.user_item_matrix.index:

            ratings = self.user_item_matrix.loc[
                student_id
            ]

            seen_internships = set(
                ratings[ratings > 0].index
            )

        # ---------------------------------------------
        # Combine scores
        # ---------------------------------------------

        results = []

        for _, row in content_recs.iterrows():

            internship_id = row["internship_id"]

            # Do not recommend internships the student
            # has already interacted with.
            if internship_id in seen_internships:
                continue

            content_score = float(
                row["similarity_score"]
            )

            collaborative_score = collaborative_scores.get(
                internship_id,
                0.0,
            )

            final_score = (
                collaborative_score
                * self.collaborative_weight
                +
                content_score
                * self.content_weight
            )

            results.append(
                {
                    "internship_id": internship_id,
                    "title": row["title"],
                    "company": row["company"],
                    "domain": row["domain"],
                    "collaborative_score": collaborative_score,
                    "content_score": content_score,
                    "hybrid_score": final_score,
                }
            )

        if not results:
            return pd.DataFrame(
                columns=[
                    "internship_id",
                    "title",
                    "company",
                    "domain",
                    "collaborative_score",
                    "content_score",
                    "hybrid_score",
                    "match_percentage",
                ]
            )

        result = pd.DataFrame(results)

        result = result.sort_values(
            by="hybrid_score",
            ascending=False,
        ).head(top_n)

        result["collaborative_score"] = (
            result["collaborative_score"].round(4)
        )

        result["content_score"] = (
            result["content_score"].round(4)
        )

        result["hybrid_score"] = (
            result["hybrid_score"].round(4)
        )

        result["match_percentage"] = (
            result["hybrid_score"] * 100
        ).round(2)

        return result.reset_index(drop=True)

    def recommend_new_student(
        self,
        skills: list[str],
        interests: str,
        domain: str,
        experience_level: str,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        Handle cold-start students.

        With no interaction history, use content-based
        recommendations instead of collaborative filtering.
        """

        recommendations = (
            self.content_recommender.recommend_new_student(
                skills=skills,
                interests=interests,
                domain=domain,
                experience_level=experience_level,
                top_n=top_n,
            )
        )

        result = recommendations.copy()

        result["collaborative_score"] = 0.0
        result["content_score"] = result[
            "similarity_score"
        ]

        # For cold start, content is the only available
        # signal, so do not penalize it by multiplying by 0.4.
        result["hybrid_score"] = result[
            "content_score"
        ]

        result["match_percentage"] = (
            result["hybrid_score"] * 100
        ).round(2)

        return result[
            [
                "internship_id",
                "title",
                "company",
                "domain",
                "collaborative_score",
                "content_score",
                "hybrid_score",
                "match_percentage",
            ]
        ].reset_index(drop=True)