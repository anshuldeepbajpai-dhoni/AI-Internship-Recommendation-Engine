from src.data_loader import load_all_data


def main():
    try:
        students, internships, interactions = load_all_data()

        print("=" * 55)
        print("AI INTERNSHIP RECOMMENDATION ENGINE")
        print("=" * 55)

        print(f"\nStudents:     {len(students)}")
        print(f"Internships:  {len(internships)}")
        print(f"Interactions: {len(interactions)}")

        print("\nStudent sample:")
        print(students.head())

        print("\nInternship sample:")
        print(internships.head())

        print("\nInteraction sample:")
        print(interactions.head())

        print("\nPhase 1 setup successful.")

    except Exception as exc:
        print(f"\nError: {exc}")


if __name__ == "__main__":
    main()