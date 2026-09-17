import os
import numpy as np

from app.ai.embedding import generate_embedding
from app.ai.fs2k_dataset import load_annotations, get_photo_path


FS2K_ROOT = "data/online_dataset/FS2K"

EMBEDDING_DIR = os.path.join(
    FS2K_ROOT,
    "embeddings"
)


def build_embeddings():

    os.makedirs(EMBEDDING_DIR, exist_ok=True)

    train_data, test_data = load_annotations()
    all_data = train_data + test_data

    total = len(all_data)
    success = 0
    skipped = 0

    print("\n========== FS2K EMBEDDING GENERATION ==========")
    print(f"Total photos: {total}")
    print("-----------------------------------------------")

    for index, item in enumerate(all_data, start=1):

        image_name = item["image_name"]

        photo_path = get_photo_path(image_name)

        if photo_path is None:
            print(f"[SKIP] Photo not found: {image_name}")
            skipped += 1
            continue

        # Create a safe embedding filename
        embedding_id = image_name.replace("/", "_")

        embedding_path = os.path.join(
            EMBEDDING_DIR,
            embedding_id + ".npy"
        )

        # Don't regenerate existing embeddings
        if os.path.exists(embedding_path):
            success += 1
            print(f"[{index}/{total}] Already exists: {image_name}")
            continue

        try:

            embedding = generate_embedding(photo_path)

            np.save(
                embedding_path,
                embedding
            )

            success += 1

            print(
                f"[{index}/{total}] "
                f"Embedded: {image_name}"
            )

        except Exception as e:

            skipped += 1

            print(
                f"[ERROR] {image_name}: {e}"
            )

    print("\n===============================================")
    print(f"Successfully embedded : {success}")
    print(f"Skipped                : {skipped}")
    print(f"Total                   : {total}")
    print("===============================================")


if __name__ == "__main__":
    build_embeddings()