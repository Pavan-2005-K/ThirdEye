import json
from collections import Counter

from app.ai.fs2k_dataset import TRAIN_ANNOTATION, TEST_ANNOTATION


def inspect_file(file_path, name):
    print("\n======================================")
    print(name)
    print("======================================")

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"Number of records: {len(data)}")

    if not data:
        print("Dataset is empty.")
        return

    # Show the first record
    print("\nFirst record:")
    print(data[0])

    # Show available fields
    print("\nFields:")
    print(list(data[0].keys()))

    # Count values for every field
    print("\nField value information:")

    for field in data[0].keys():
        values = [item.get(field) for item in data]

        unique_values = set(
            str(value) for value in values
        )

        print(
            f"{field}: "
            f"{len(unique_values)} unique values"
        )

        # Show first few values
        print(
            "  Examples:",
            list(unique_values)[:10]
        )


def main():

    print("\n======================================")
    print("FS2K ANNOTATION INSPECTION")
    print("======================================")

    inspect_file(
        TRAIN_ANNOTATION,
        "TRAIN ANNOTATION"
    )

    inspect_file(
        TEST_ANNOTATION,
        "TEST ANNOTATION"
    )

    print("\n======================================")
    print("Inspection completed")
    print("======================================")


if __name__ == "__main__":
    main()