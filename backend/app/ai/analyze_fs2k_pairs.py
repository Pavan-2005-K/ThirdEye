import json
from collections import Counter, defaultdict

from app.ai.fs2k_dataset import TRAIN_ANNOTATION, TEST_ANNOTATION


def analyze(file_path, name):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    print("\n======================================")
    print(name)
    print("======================================")

    # Group records by image number
    groups = defaultdict(list)

    for item in data:
        image_name = item["image_name"]

        # Example:
        # photo1/image0110
        #
        # We keep:
        # image0110
        filename = image_name.split("/")[-1]

        groups[filename].append(item)

    counts = Counter(
        len(records)
        for records in groups.values()
    )

    print(f"Total records: {len(data)}")
    print(f"Unique image IDs: {len(groups)}")

    print("\nRecords per image ID:")

    for count, number in sorted(counts.items()):
        print(f"  {count} record(s): {number} image IDs")

    print("\nExamples with multiple records:")

    shown = 0

    for image_id, records in groups.items():

        if len(records) > 1:

            print(f"\n{image_id}")

            for record in records:
                print(
                    f"  {record['image_name']} "
                    f"(style={record['style']})"
                )

            shown += 1

            if shown >= 10:
                break


def main():

    print("\n======================================")
    print("FS2K PAIR STRUCTURE ANALYSIS")
    print("======================================")

    analyze(
        TRAIN_ANNOTATION,
        "TRAIN"
    )

    analyze(
        TEST_ANNOTATION,
        "TEST"
    )

    print("\n======================================")
    print("Analysis completed")
    print("======================================")


if __name__ == "__main__":
    main()