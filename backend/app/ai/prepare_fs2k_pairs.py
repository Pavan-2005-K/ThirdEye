import os
import json
import csv

from app.ai.fs2k_dataset import (
    TRAIN_ANNOTATION,
    get_photo_path,
    get_sketch_path
)


OUTPUT_FILE = "data/online_dataset/FS2K/train_pairs.csv"


def prepare_training_pairs():

    with open(TRAIN_ANNOTATION, "r", encoding="utf-8") as file:
        train_data = json.load(file)

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    valid_pairs = []

    for item in train_data:

        image_name = item["image_name"]

        photo_path = get_photo_path(image_name)
        sketch_path = get_sketch_path(image_name)

        if photo_path is None:
            continue

        if sketch_path is None:
            continue

        valid_pairs.append({
            "image_name": image_name,
            "sketch_path": sketch_path,
            "photo_path": photo_path
        })

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_name",
                "sketch_path",
                "photo_path"
            ]
        )

        writer.writeheader()
        writer.writerows(valid_pairs)

    print("\n======================================")
    print("FS2K TRAINING PAIRS")
    print("======================================")

    print(f"Training records : {len(train_data)}")
    print(f"Valid pairs      : {len(valid_pairs)}")
    print(f"Output file      : {OUTPUT_FILE}")

    print("======================================")


if __name__ == "__main__":
    prepare_training_pairs()