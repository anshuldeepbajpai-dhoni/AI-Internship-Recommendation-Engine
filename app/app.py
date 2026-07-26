from pathlib import Path
import sys

import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.data_loader import load_all_data
from src.preprocessing import (
    preprocess_students,
    preprocess_internships,
    preprocess_interactions,
)
from src.user_item_matrix import create_user_item_matrix
from src.collaborative_recommender import CollaborativeRecommender
from src.content_recommender import ContentBasedRecommender
from src.hybrid_recommender import HybridRecommender


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Internship Recommendation Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LOAD SYSTEM
# ============================================================

@st.cache_resource
def load_recommendation_system():

    students, internships, interactions = load_all_data()

    students = preprocess_students(students)
    internships = preprocess_internships(internships)
    interactions = preprocess_interactions(interactions)

    matrix = create_user_item_matrix(
        interactions
    )

    collaborative = CollaborativeRecommender(
        user_item_matrix=matrix,
        internships=internships,
    )

    content = ContentBasedRecommender(
        students=students,
        internships=internships,
    )

    hybrid = HybridRecommender(
        students=students,
        internships=internships,
        user_item_matrix=matrix,
        collaborative_weight=0.6,
        content_weight=0.4,
    )

    return (
        students,
        internships,
        interactions,
        matrix,
        collaborative,
        content,
        hybrid,
    )


try:
    (
        students,
        internships,
        interactions,
        user_item_matrix,
        collaborative_recommender,
        content_recommender,
        hybrid_recommender,
    ) = load_recommendation_system()

except Exception as exc:
    st.error(
        f"Failed to initialize recommendation system: {exc}"
    )
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🎯 InternMatch AI")

    st.caption(
        "AI-powered internship recommendation system"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Existing Student",
            "New Student",
            "Analytics",
        ],
    )

    st.divider()

    st.caption(
        "Collaborative + Content-Based + Hybrid Filtering"
    )


# ============================================================
# HOME
# ============================================================

if page == "Home":

    st.title(
        "AI Internship Recommendation Engine"
    )

    st.write(
        """
        Find relevant internships using student skills,
        interests, domain, experience level, and historical
        interaction patterns.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Students",
            len(students),
        )

    with col2:
        st.metric(
            "Internships",
            len(internships),
        )

    with col3:
        st.metric(
            "Interactions",
            len(interactions),
        )

    st.subheader(
        "Recommendation Approaches"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            ### Collaborative

            Uses historical ratings and finds students
            with similar preference patterns.
            """
        )

    with col2:

        st.markdown(
            """
            ### Content-Based

            Matches student skills, interests, domain,
            and experience against internship profiles.
            """
        )

    with col3:

        st.markdown(
            """
            ### Hybrid

            Combines collaborative and content-based
            signals into a final ranked recommendation.
            """
        )

    st.subheader("Available Internships")

    display_internships = internships[
        [
            "internship_id",
            "title",
            "company",
            "domain",
            "duration",
            "experience_level",
        ]
    ].copy()

    st.dataframe(
        display_internships,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# EXISTING STUDENT
# ============================================================

elif page == "Existing Student":

    st.title(
        "Existing Student Recommendations"
    )

    st.write(
        "Select a registered student and recommendation method."
    )

    student_options = {}

    for _, row in students.iterrows():

        label = (
            f"{row['student_id'].upper()} — "
            f"{row['name'].title()}"
        )

        student_options[label] = row[
            "student_id"
        ]

    selected_student_label = st.selectbox(
        "Student",
        list(student_options.keys()),
    )

    student_id = student_options[
        selected_student_label
    ]

    student = students[
        students["student_id"] == student_id
    ].iloc[0]

    st.subheader("Student Profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Domain",
            student["domain"].title(),
        )

    with col2:
        st.metric(
            "Experience",
            student[
                "experience_level"
            ].title(),
        )

    with col3:

        interaction_count = len(
            interactions[
                interactions["student_id"]
                == student_id
            ]
        )

        st.metric(
            "Interactions",
            interaction_count,
        )

    st.write(
        "**Skills:** "
        + ", ".join(
            skill.title()
            for skill in student[
                "skills_list"
            ]
        )
    )

    st.write(
        "**Interests:** "
        + student["interests"]
        .replace("|", ", ")
        .title()
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        algorithm = st.selectbox(
            "Recommendation Method",
            [
                "Hybrid",
                "Content-Based",
                "Collaborative",
            ],
        )

    with col2:

        top_n = st.slider(
            "Number of recommendations",
            min_value=1,
            max_value=10,
            value=5,
        )

    if st.button(
        "Generate Recommendations",
        type="primary",
        use_container_width=True,
    ):

        try:

            if algorithm == "Hybrid":

                recommendations = (
                    hybrid_recommender.recommend(
                        student_id=student_id,
                        top_n=top_n,
                    )
                )

            elif algorithm == "Content-Based":

                recommendations = (
                    content_recommender.recommend(
                        student_id=student_id,
                        top_n=top_n,
                    )
                )

            else:

                recommendations = (
                    collaborative_recommender.recommend(
                        student_id=student_id,
                        top_n=top_n,
                    )
                )

            st.subheader(
                f"{algorithm} Recommendations"
            )

            if recommendations.empty:

                st.warning(
                    "No recommendations are available "
                    "for this student."
                )

            else:

                st.dataframe(
                    recommendations,
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as exc:

            st.error(
                f"Recommendation failed: {exc}"
            )


# ============================================================
# NEW STUDENT / COLD START
# ============================================================

elif page == "New Student":

    st.title(
        "New Student Recommendation"
    )

    st.write(
        """
        Enter a student profile to receive recommendations
        without requiring previous internship ratings.
        """
    )

    with st.form(
        "new_student_form"
    ):

        skills_input = st.text_input(
            "Skills",
            placeholder=(
                "Python, Pandas, SQL, Machine Learning"
            ),
        )

        interests = st.text_input(
            "Interests",
            placeholder=(
                "AI, Data Science, Analytics"
            ),
        )

        domains = sorted(
            internships[
                "domain"
            ].dropna().unique()
        )

        domain = st.selectbox(
            "Preferred Domain",
            domains,
        )

        experience_level = st.selectbox(
            "Experience Level",
            [
                "beginner",
                "intermediate",
                "advanced",
            ],
        )

        top_n = st.slider(
            "Number of recommendations",
            min_value=1,
            max_value=10,
            value=5,
            key="new_student_top_n",
        )

        submitted = st.form_submit_button(
            "Find Internships",
            type="primary",
            use_container_width=True,
        )

    if submitted:

        skills = [
            skill.strip().lower()
            for skill in skills_input.split(",")
            if skill.strip()
        ]

        if not skills:

            st.warning(
                "Enter at least one skill."
            )

        else:

            try:

                recommendations = (
                    hybrid_recommender
                    .recommend_new_student(
                        skills=skills,
                        interests=interests,
                        domain=domain,
                        experience_level=experience_level,
                        top_n=top_n,
                    )
                )

                st.subheader(
                    "Recommended Internships"
                )

                if recommendations.empty:

                    st.warning(
                        "No matching internships found."
                    )

                else:

                    for rank, row in (
                        recommendations.iterrows()
                    ):

                        with st.container(
                            border=True
                        ):

                            col1, col2 = (
                                st.columns(
                                    [4, 1]
                                )
                            )

                            with col1:

                                st.subheader(
                                    f"#{rank + 1} "
                                    f"{row['title'].title()}"
                                )

                                st.write(
                                    f"**Company:** "
                                    f"{row['company'].title()}"
                                )

                                st.write(
                                    f"**Domain:** "
                                    f"{row['domain'].title()}"
                                )

                            with col2:

                                st.metric(
                                    "Match",
                                    (
                                        f"{row['match_percentage']}"
                                        "%"
                                    ),
                                )

            except Exception as exc:

                st.error(
                    f"Recommendation failed: {exc}"
                )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "Analytics":

    st.title(
        "Recommendation Analytics"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Students",
            len(students),
        )

    with col2:
        st.metric(
            "Internships",
            len(internships),
        )

    with col3:
        st.metric(
            "Interactions",
            len(interactions),
        )

    with col4:

        average_rating = (
            interactions[
                "rating"
            ].mean()
        )

        st.metric(
            "Average Rating",
            f"{average_rating:.2f}",
        )

    st.divider()

    st.subheader(
        "Internships by Domain"
    )

    domain_counts = (
        internships[
            "domain"
        ]
        .value_counts()
        .rename_axis("domain")
        .reset_index(name="internships")
    )

    st.bar_chart(
        domain_counts,
        x="domain",
        y="internships",
    )

    st.subheader(
        "Students by Domain"
    )

    student_domain_counts = (
        students[
            "domain"
        ]
        .value_counts()
        .rename_axis("domain")
        .reset_index(name="students")
    )

    st.bar_chart(
        student_domain_counts,
        x="domain",
        y="students",
    )

    st.subheader(
        "Rating Distribution"
    )

    rating_counts = (
        interactions[
            "rating"
        ]
        .value_counts()
        .sort_index()
        .rename_axis("rating")
        .reset_index(name="count")
    )

    st.bar_chart(
        rating_counts,
        x="rating",
        y="count",
    )

    st.subheader(
        "Most Active Students"
    )

    activity = (
        interactions
        .groupby("student_id")
        .size()
        .reset_index(
            name="interactions"
        )
        .sort_values(
            "interactions",
            ascending=False,
        )
    )

    st.dataframe(
        activity,
        use_container_width=True,
        hide_index=True,
    )