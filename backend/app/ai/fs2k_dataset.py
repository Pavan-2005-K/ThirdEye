import os
import json


FS2K_ROOT = "data/online_dataset/FS2K"

PHOTO_DIR = os.path.join(FS2K_ROOT, "photo")
SKETCH_DIR = os.path.join(FS2K_ROOT, "sketch")

TRAIN_ANNOTATION = os.path.join(FS2K_ROOT, "anno_train.json")
TEST_ANNOTATION = os.path.join(FS2K_ROOT, "anno_test.json")


def get_sketch_path(image_name: str):
    """
    Convert an FS2K photo name into its corresponding sketch path.

    Example:
        photo1/image0110
        ->
        sketch1/sketch0110.jpg
    """

    folder, filename = image_name.split("/")

    sketch_folder = folder.replace("photo", "sketch")
    sketch_filename = filename.replace("image", "sketch")

    sketch_path = os.path.join(
        SKETCH_DIR,
        sketch_folder,
        sketch_filename + ".jpg"
    )

    if os.path.exists(sketch_path):
        return sketch_path

    # Some files may use PNG
    sketch_path_png = os.path.splitext(sketch_path)[0] + ".png"

    if os.path.exists(sketch_path_png):
        return sketch_path_png

    return None


def get_photo_path(image_name: str):
    """
    Convert annotation image_name into the actual photo path.
    """

    photo_path = os.path.join(
        PHOTO_DIR,
        image_name + ".jpg"
    )

    if os.path.exists(photo_path):
        return photo_path

    return None


def load_annotations():
    """
    Load both training and testing annotations.
    """

    with open(TRAIN_ANNOTATION, "r", encoding="utf-8") as file:
        train_data = json.load(file)

    with open(TEST_ANNOTATION, "r", encoding="utf-8") as file:
        test_data = json.load(file)

    return train_data, test_data


def validate_dataset():

    train_data, test_data = load_annotations()

    all_data = train_data + test_data

    valid_pairs = []
    missing_photos = []
    missing_sketches = []

    for item in all_data:

        image_name = item["image_name"]

        photo_path = get_photo_path(image_name)
        sketch_path = get_sketch_path(image_name)

        if photo_path is None:
            missing_photos.append(image_name)
            continue

        if sketch_path is None:
            missing_sketches.append(image_name)
            continue

        valid_pairs.append({
            "image_name": image_name,
            "photo_path": photo_path,
            "sketch_path": sketch_path
        })

    print("\n========== FS2K DATASET VALIDATION ==========")

    print(f"Training records : {len(train_data)}")
    print(f"Testing records  : {len(test_data)}")
    print(f"Total records    : {len(all_data)}")

    print("---------------------------------------------")

    print(f"Valid pairs      : {len(valid_pairs)}")
    print(f"Missing photos   : {len(missing_photos)}")
    print(f"Missing sketches : {len(missing_sketches)}")

    print("---------------------------------------------")

    if missing_photos:
        print("\nMissing photo examples:")
        for item in missing_photos[:10]:
            print(item)

    if missing_sketches:
        print("\nMissing sketch examples:")
        for item in missing_sketches[:10]:
            print(item)

    print("\n=============================================")

    return valid_pairs


if __name__ == "__main__":
    validate_dataset()