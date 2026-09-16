# 🤖 AI Internship Recommendation Engine

An intelligent **AI-powered internship recommendation system** that recommends relevant internships to students based on their **skills, interests, preferred domain, experience level, and previous internship interactions**.

The system implements **Collaborative Filtering, Content-Based Filtering, and Hybrid Recommendation** techniques using Python and machine learning, with an interactive **Streamlit dashboard**.

---

## 📌 Task Information

| Attribute      | Details                                 |
| -------------- | --------------------------------------- |
| **Task ID**    | AI-SS-002                               |
| **Task Name**  | AI Internship Recommendation Engine     |
| **Domain**     | Student Support & Internship Management |
| **Category**   | Recommendation System                   |
| **Technology** | Python, Pandas, NumPy, Scikit-learn     |
| **Interface**  | Streamlit                               |
| **Testing**    | Pytest                                  |
| **Author**     | Anshul Deep Bajpai                      |

---

## 🎯 Project Objective

The objective of this project is to build an intelligent recommendation engine that helps students discover internships relevant to their profiles.

Instead of displaying internships using simple keyword matching, the system analyzes:

* 🧑‍🎓 Student skills
* 💡 Interests
* 🏢 Preferred domain
* 📈 Experience level
* ⭐ Previous internship ratings/interactions
* 💼 Internship requirements

The recommendation engine then ranks internships according to their predicted relevance.

---

# ✨ Key Features

### 👤 Student Management

* Student profile management
* Skills and interests management
* Domain preference
* Experience-level information

### 💼 Internship Management

* Internship dataset management
* Internship domain classification
* Required skill processing
* Experience-level matching

### 🧹 Data Processing

* Data loading from CSV files
* Data preprocessing
* Data validation
* Missing-value handling
* Duplicate checking

### 🤝 Collaborative Filtering

* User-item rating matrix
* Student similarity calculation
* User-based collaborative filtering
* Rating prediction
* Similar-student recommendations

### 📝 Content-Based Filtering

* Student profile representation
* Internship profile representation
* TF-IDF vectorization
* Cosine similarity
* Skill/domain/experience matching

### 🔀 Hybrid Recommendation

* Collaborative + Content-based recommendation
* Weighted recommendation scoring
* Normalized collaborative predictions
* Top-N internship ranking

### 🆕 Cold-Start Recommendation

New students can receive recommendations without previous interaction history.

The system automatically falls back to **content-based recommendations** when collaborative interaction data is unavailable.

### 📊 Evaluation

* Precision@K
* Recall@K
* MAE
* RMSE
* Leave-one-out evaluation

### 🖥️ Interactive Dashboard

* Streamlit interface
* Student recommendation interface
* New-student recommendation interface
* Recommendation analytics
* Dataset statistics

### 🧪 Testing

Automated tests are included for:

* Collaborative recommender
* Content recommender
* Hybrid recommender
* Evaluation metrics

---

# 🧠 Recommendation Architecture

The system follows a multi-stage recommendation pipeline:

```text
                    ┌─────────────────────┐
                    │     Student Data    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Preprocessing  │
                    │ & Validation        │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
       ┌───────────────────┐       ┌───────────────────┐
       │ Collaborative     │       │ Content-Based     │
       │ Filtering         │       │ Filtering         │
       └─────────┬─────────┘       └─────────┬─────────┘
                 │                           │
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Hybrid Recommendation│
                    │       Engine        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Weighted Ranking    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Top-N Internships │
                    └─────────────────────┘
```

---

# 🔹 1. Collaborative Filtering

Collaborative filtering recommends internships based on the behavior and preferences of similar students.

The system creates a **student × internship rating matrix**.

Example:

| Student   | Python Internship | Data Analyst | ML Engineer | Web Developer |
| --------- | ----------------: | -----------: | ----------: | ------------: |
| Student A |                 5 |            4 |           5 |             1 |
| Student B |                 5 |            4 |           4 |             1 |
| Student C |                 1 |            2 |           1 |             5 |

Students with similar interaction patterns are considered similar.

### Similarity

Cosine similarity is used to measure student similarity:

$$
Similarity(A,B)=
\frac{A\cdot B}{||A||\,||B||}
$$

A higher similarity value indicates more similar interaction behavior.

The system uses these similarities to generate collaborative recommendation scores.

---

# 🔹 2. Content-Based Filtering

Content-based filtering recommends internships based on the characteristics of the student and internship.

### Student Profile

The student profile contains:

* Skills
* Interests
* Preferred domain
* Experience level

### Internship Profile

The internship profile contains:

* Internship title
* Required skills
* Domain
* Experience level

The system combines these attributes into textual representations.

### TF-IDF

TF-IDF converts the textual profiles into numerical vectors.

$$
TFIDF(t,d)=TF(t,d)\times IDF(t)
$$

The resulting vectors are compared using cosine similarity.

$$
CosineSimilarity(A,B)=
\frac{A\cdot B}{||A||\,||B||}
$$

The resulting similarity represents how relevant an internship is to a student's profile.

---

# 🔹 3. Hybrid Recommendation

The hybrid recommender combines collaborative and content-based signals.

For students with interaction history:

$$
HybridScore =
0.6 \times CollaborativeScore
+
0.4 \times ContentScore
$$

Where:

* **60%** weight → Collaborative Filtering
* **40%** weight → Content-Based Filtering

Collaborative rating predictions are normalized before combining them with content similarity.

### Recommendation Flow

```text
Student Profile
      │
      ├───────────────┐
      │               │
      ▼               ▼
Content Model    Interaction History
      │               │
      ▼               ▼
Content Score   Collaborative Score
      │               │
      └───────┬───────┘
              ▼
       Weighted Hybrid
          Score
              │
              ▼
        Sort / Rank
              │
              ▼
          Top-N Jobs
```

---

# 🆕 Cold-Start Problem

One major challenge in recommendation systems is the **cold-start problem**.

A new student has no previous interactions or ratings, so collaborative filtering cannot reliably generate recommendations.

### Solution

For a new student:

```text
New Student
     │
     ▼
Student Profile
     │
     ├── Skills
     ├── Interests
     ├── Domain
     └── Experience
             │
             ▼
    Content-Based Model
             │
             ▼
     Internship Similarity
             │
             ▼
       Top-N Results
```

Therefore, the system can provide recommendations even when the student has no historical interaction data.

---

# 📊 Recommendation Scoring

The recommendation engine produces a score for every candidate internship.

For existing students:

```text
Final Score
    =
0.6 × Collaborative Score
    +
0.4 × Content Score
```

For new students:

```text
Final Score
    =
Content Similarity Score
```

Internships are then sorted in descending order of their final score.

---

# 📈 Evaluation

The project includes recommendation-quality evaluation utilities.

## Precision@K

Precision@K measures how many of the recommended internships are relevant.

$$
Precision@K =
\frac{\text{Relevant Recommendations}}
{\text{Total Recommendations}}
$$

---

## Recall@K

Recall@K measures how many relevant internships were successfully recommended.

$$
Recall@K =
\frac{\text{Relevant Recommendations}}
{\text{Total Relevant Items}}
$$

---

## MAE

Mean Absolute Error measures the average absolute difference between predicted and actual ratings.

$$
MAE =
\frac{1}{n}
\sum_{i=1}^{n}
|y_i-\hat{y_i}|
$$

---

## RMSE

Root Mean Squared Error penalizes larger prediction errors more heavily.

$$
RMSE =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y_i})^2
}
$$

---

## Leave-One-Out Evaluation

The recommendation system also supports **leave-one-out evaluation**.

A student's interaction is temporarily held out and the system attempts to recommend that internship among the Top-N results.

This provides a way to evaluate ranking performance using historical interactions.

---

# 📂 Project Structure

```text
AI-Internship-Recommendation-Engine/
│
├── app/
│   └── app.py
│
├── data/
│   ├── students.csv
│   ├── internships.csv
│   └── interactions.csv
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── validation.py
│   ├── user_item_matrix.py
│   ├── collaborative_recommender.py
│   ├── content_recommender.py
│   ├── hybrid_recommender.py
│   └── evaluation.py
│
├── tests/
│   ├── test_collaborative.py
│   ├── test_content.py
│   ├── test_hybrid.py
│   └── test_evaluation.py
│
├── screenshots/
│
├── .streamlit/
│   └── config.toml
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🗂️ Dataset

The project uses three CSV datasets.

### `students.csv`

Contains student profile information such as:

* Student ID
* Skills
* Interests
* Preferred domain
* Experience level

### `internships.csv`

Contains internship information such as:

* Internship ID
* Internship title
* Required skills
* Domain
* Experience level

### `interactions.csv`

Contains student-internship interaction information such as:

* Student ID
* Internship ID
* Rating / interaction score

No external database is required.

---

# 🖥️ Streamlit Dashboard

The Streamlit application contains four major sections.

## 🏠 Home

Displays:

* Project overview
* Dataset statistics
* Number of students
* Number of internships
* Number of interactions
* Recommendation approaches
* Available internships

---

## 👤 Existing Student

Registered students can receive recommendations using:

### Hybrid Recommendation

Combines collaborative and content-based scores.

### Content-Based Recommendation

Uses student profile and internship similarity.

### Collaborative Recommendation

Uses interaction history and similar students.

---

## 🆕 New Student

A new student can enter:

* Skills
* Interests
* Preferred domain
* Experience level

No previous interaction history is required.

The system then generates personalized recommendations using the content-based model.

---

## 📊 Analytics

The analytics section provides insights such as:

* Number of students
* Number of internships
* Number of interactions
* Average rating
* Internship domain distribution
* Student domain distribution
* Rating distribution
* Student activity

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
cd AI-Internship-Recommendation-Engine
```

## 2. Create Virtual Environment

### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Command-Line Application

```bash
python main.py
```

---

## Streamlit Dashboard

```bash
streamlit run app/app.py
```

The Streamlit application will open in your browser.

---

# 🧪 Running Tests

Run the complete automated test suite:

```bash
python -m pytest -v
```

The tests cover:

```text
Collaborative Filtering
Content-Based Filtering
Hybrid Recommendation
Evaluation Metrics
```

---

# 🛠️ Technologies Used

| Technology            | Purpose                           |
| --------------------- | --------------------------------- |
| **Python**            | Core programming language         |
| **Pandas**            | Data manipulation                 |
| **NumPy**             | Numerical computation             |
| **Scikit-learn**      | Machine learning and similarity   |
| **TF-IDF**            | Text feature extraction           |
| **Cosine Similarity** | Profile/recommendation similarity |
| **Streamlit**         | Interactive dashboard             |
| **Matplotlib**        | Data visualization                |
| **Pytest**            | Automated testing                 |

---

# 🔄 End-to-End Workflow

```text
CSV Dataset
     │
     ▼
Data Loading
     │
     ▼
Data Validation
     │
     ▼
Preprocessing
     │
     ├───────────────────┐
     │                   │
     ▼                   ▼
Interaction Data     Profile Data
     │                   │
     ▼                   ▼
User-Item Matrix    TF-IDF Features
     │                   │
     ▼                   ▼
Collaborative       Content-Based
Filtering           Filtering
     │                   │
     └─────────┬─────────┘
               ▼
       Hybrid Recommendation
               │
               ▼
       Recommendation Scoring
               │
               ▼
          Top-N Ranking
               │
               ▼
      Streamlit Dashboard
```

---

# 🚀 Future Improvements

Potential improvements include:

* 🌐 Larger real-world internship dataset
* 🧠 Transformer-based skill embeddings
* 📍 Location-based internship matching
* 💼 Internship description matching
* ⭐ Explicit student feedback integration
* 🔄 Real-time recommendation updates
* 🤖 Deep learning recommendation models
* 🔌 REST API
* 🧪 A/B testing framework
* ☁️ Cloud deployment
* 🔍 Semantic search
* 📱 Mobile-friendly interface

---

# 📌 Project Highlights

This project demonstrates practical implementation of:

* Recommendation systems
* Collaborative filtering
* Content-based filtering
* Hybrid recommendation
* TF-IDF
* Cosine similarity
* User-item matrices
* Cold-start handling
* Top-N ranking
* Recommendation evaluation
* Data preprocessing
* Machine learning
* Streamlit application development
* Automated testing

---

# 👨‍💻 Author

**Anshul Deep Bajpai**

B.Tech — Computer Science & Engineering
Artificial Intelligence & Machine Learning

---

# 📜 Task

**AI-SS-002 — AI Internship Recommendation Engine**

A machine-learning-based internship recommendation platform designed to assist students in discovering internships aligned with their profiles and interests.
