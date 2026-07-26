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
from src.user_item_matrix import create_user_item_matrix


def main():
    try:
        print("=" * 60)
        print("AI INTERNSHIP RECOMMENDATION ENGINE")
        print("=" * 60)

        # ---------------------------------
        # 1. Load
        # ---------------------------------

        students, internships, interactions = load_all_data()

        print("\n[1] Data loaded successfully")

        print(f"Students:     {len(students)}")
        print(f"Internships:  {len(internships)}")
        print(f"Interactions: {len(interactions)}")

        # ---------------------------------
        # 2. Validate raw data
        # ---------------------------------

        validate_students(students)
        validate_internships(internships)

        print("\n[2] Basic validation passed")

        # ---------------------------------
        # 3. Preprocess
        # ---------------------------------

        students = preprocess_students(students)
        internships = preprocess_internships(internships)
        interactions = preprocess_interactions(interactions)

        print("\n[3] Preprocessing completed")

        # ---------------------------------
        # 4. Validate relationships
        # ---------------------------------

        validate_interactions(
            interactions,
            students,
            internships,
        )

        print("\n[4] Interaction validation passed")

        # ---------------------------------
        # 5. User-item matrix
        # ---------------------------------

        user_item_matrix = create_user_item_matrix(
            interactions
        )

        print("\n[5] User-Item Rating Matrix")
        print("-" * 60)

        print(user_item_matrix)

        print("\nMatrix shape:", user_item_matrix.shape)

        # ---------------------------------
        # Example processed student
        # ---------------------------------

        print("\n[6] Processed Student Example")
        print("-" * 60)

        student = students.iloc[0]

        print("Name:", student["name"])
        print("Skills:", student["skills_list"])
        print("Interests:", student["interests"])
        print("Domain:", student["domain"])
        print("Experience:", student["experience_level"])

        print("\n" + "=" * 60)
        print("PHASE 2 COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except Exception as exc:
        print(f"\nERROR: {exc}")


if __name__ == "__main__":
    main()