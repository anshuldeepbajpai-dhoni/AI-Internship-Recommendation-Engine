# AI Internship Recommendation Engine

An intelligent internship recommendation system that recommends relevant internships to students based on their skills, interests, domain, experience level, and previous internship interactions.

The project implements Collaborative Filtering, Content-Based Filtering, and Hybrid Recommendation using Python and machine learning.

## Task Information

- Task ID: AI-SS-002
- Domain: Student Support & Internship Management Recommendation
- Task Name: AI Internship Recommendation Engine
- Technology: Python, Pandas, NumPy, Scikit-learn
- Interface: Streamlit

## Features

- Student profile management
- Internship dataset management
- Data preprocessing and validation
- User-item rating matrix
- User-based collaborative filtering
- TF-IDF content-based filtering
- Cosine similarity
- Hybrid recommendation engine
- Weighted recommendation scoring
- Cold-start student recommendations
- Top-N internship ranking
- Precision@K and Recall@K
- MAE and RMSE utilities
- Interactive Streamlit dashboard
- Recommendation analytics
- Automated tests

## Recommendation System

### Collaborative Filtering

Collaborative filtering recommends internships based on the preferences of similar students.

The system creates a student-internship rating matrix and calculates similarity between students using cosine similarity.

### Content-Based Filtering

Content-based filtering recommends internships by comparing student profiles with internship requirements.

Student profiles include:

- Skills
- Interests
- Preferred domain
- Experience level

Internship profiles include:

- Required skills
- Domain
- Experience level
- Internship title

TF-IDF is used to convert profiles into numerical vectors, and cosine similarity measures the relevance between students and internships.

### Hybrid Recommendation

The hybrid model combines collaborative and content-based recommendations.

For students with interaction history:

Hybrid Score = 0.6 × Collaborative Score + 0.4 × Content Score

Collaborative rating predictions are normalized before being combined with content similarity.

For new students without interaction history, the system falls back to content-based recommendations.

## Project Structure

AI-Internship-Recommendation-Engine/

    app/
        app.py

    data/
        students.csv
        internships.csv
        interactions.csv

    src/
        __init__.py
        data_loader.py
        preprocessing.py
        validation.py
        user_item_matrix.py
        collaborative_recommender.py
        content_recommender.py
        hybrid_recommender.py
        evaluation.py

    tests/
        test_collaborative.py
        test_content.py
        test_hybrid.py
        test_evaluation.py

    screenshots/

    .streamlit/
        config.toml

    main.py
    requirements.txt
    README.md
    .gitignore

## Installation

Clone the repository:

    git clone YOUR_REPOSITORY_URL
    cd AI-Internship-Recommendation-Engine

Create a virtual environment:

Windows:

    python -m venv .venv
    .venv\Scripts\activate

Install dependencies:

    pip install -r requirements.txt

## Run Command-Line Application

    python main.py

## Run Streamlit Dashboard

    streamlit run app/app.py

## Run Tests

    python -m pytest -v

## Dashboard

The Streamlit dashboard contains four main sections:

### Home

Displays project information, dataset statistics, recommendation approaches, and available internships.

### Existing Student

Generates recommendations for registered students using:

- Hybrid Recommendation
- Content-Based Filtering
- Collaborative Filtering

### New Student

Generates internship recommendations using skills, interests, domain, and experience level without requiring previous interactions.

This addresses the cold-start problem.

### Analytics

Displays:

- Number of students
- Number of internships
- Number of interactions
- Average rating
- Internship domain distribution
- Student domain distribution
- Rating distribution
- Student activity

## Evaluation

The project contains utilities for evaluating recommendation quality using:

- Precision@K
- Recall@K
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

Leave-one-out evaluation is used to test Top-N recommendation performance on held-out interactions.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Pytest
- Matplotlib

## Dataset

The project uses CSV files for:

- Student profiles
- Internship profiles
- Student-internship interactions

No external database is required.

## Future Improvements

- Larger real-world internship dataset
- Skill embeddings using transformer models
- Internship descriptions and location matching
- User feedback integration
- Real-time recommendation updates
- Deep learning recommendation models
- REST API
- A/B testing
- Cloud deployment

## Author

Anshul Deep Bajpai

## Task

AI-SS-002 — AI Internship Recommendation Engine