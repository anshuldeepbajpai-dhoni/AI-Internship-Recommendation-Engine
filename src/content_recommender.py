import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """
    Recommend internships by comparing a student's profile
    with internship requirements using TF-IDF and cosine similarity.
    """

    def __init__(
        self,
        students: pd.DataFrame,
        internships: pd.DataFrame,
    ):
        if students.empty:
            raise ValueError("Students dataset cannot be empty.")

        if internships.empty:
            raise ValueError("Internships dataset cannot be empty.")

        self.students = students.copy()
        self.internships = internships.copy()

        self.vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        self.internship_profiles = (
            self._create_internship_profiles()
        )

        self.internship_vectors = (
            self.vectorizer.fit_transform(
                self.internship_profiles
            )
        )


    def _create_internship_profiles(self) -> list[str]:
        """
        Combine internship features into one text profile.
        """

        profiles = []

        for _, internship in self.internships.iterrows():

            skills = internship.get(
                "required_skills_list",
                [],
            )

            if not isinstance(skills, list):
                skills = []

            domain = str(
                internship.get("domain", "")
            )

            experience = str(
                internship.get(
                    "experience_level",
                    "",
                )
            )

            title = str(
                internship.get("title", "")
            )

            profile = " ".join(
                skills
                + [
                    domain,
                    experience,
                    title,
                ]
            )

            profiles.append(profile)

        return profiles


    def _create_student_profile(
        self,
        student: pd.Series,
    ) -> str:
        """
        Combine student features into one text profile.
        """

        skills = student.get(
            "skills_list",
            [],
        )

        if not isinstance(skills, list):
            skills = []

        interests = str(
            student.get("interests", "")
        ).replace("|", " ")

        domain = str(
            student.get("domain", "")
        )

        experience = str(
            student.get(
                "experience_level",
                "",
            )
        )

        profile = " ".join(
            skills
            + [
                interests,
                domain,
                experience,
            ]
        )

        return profile


    def recommend(
        self,
        student_id: str,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        Generate Top-N internships for an existing student.
        """

        student_id = student_id.strip().lower()

        student_rows = self.students[
            self.students["student_id"]
            == student_id
        ]

        if student_rows.empty:
            raise ValueError(
                f"Student '{student_id}' not found."
            )

        student = student_rows.iloc[0]

        student_profile = (
            self._create_student_profile(
                student
            )
        )

        student_vector = (
            self.vectorizer.transform(
                [student_profile]
            )
        )

        similarities = cosine_similarity(
            student_vector,
            self.internship_vectors,
        )[0]

        result = self.internships[
            [
                "internship_id",
                "title",
                "company",
                "domain",
                "experience_level",
            ]
        ].copy()

        result["similarity_score"] = similarities

        result = result.sort_values(
            by="similarity_score",
            ascending=False,
        ).head(top_n)

        result["similarity_score"] = (
            result["similarity_score"]
            .round(4)
        )

        result["match_percentage"] = (
            result["similarity_score"] * 100
        ).round(2)

        return result.reset_index(
            drop=True
        )


    def recommend_new_student(
        self,
        skills: list[str],
        interests: str,
        domain: str,
        experience_level: str,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        Generate recommendations for a student who does not
        have previous ratings/interactions.
        """

        clean_skills = [
            str(skill).strip().lower()
            for skill in skills
            if str(skill).strip()
        ]

        profile = " ".join(
            clean_skills
            + [
                str(interests)
                .replace("|", " ")
                .strip()
                .lower(),

                str(domain)
                .strip()
                .lower(),

                str(experience_level)
                .strip()
                .lower(),
            ]
        )

        student_vector = (
            self.vectorizer.transform(
                [profile]
            )
        )

        similarities = cosine_similarity(
            student_vector,
            self.internship_vectors,
        )[0]

        result = self.internships[
            [
                "internship_id",
                "title",
                "company",
                "domain",
                "experience_level",
            ]
        ].copy()

        result["similarity_score"] = similarities

        result = result.sort_values(
            by="similarity_score",
            ascending=False,
        ).head(top_n)

        result["similarity_score"] = (
            result["similarity_score"]
            .round(4)
        )

        result["match_percentage"] = (
            result["similarity_score"] * 100
        ).round(2)

        return result.reset_index(
            drop=True
        )