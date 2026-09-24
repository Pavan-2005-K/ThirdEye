import os
import numpy as np

from app.ai.fs2k_dataset import load_annotations, get_photo_path
from app.ai.siamese_embedding import generate_siamese_embedding


FS2K_ROOT = "data/online_dataset/FS2K"

OUTPUT_DIR = os.path.join(
    FS2K_ROOT,
    "embeddings_siamese_v2"
)


def build_embeddings():

    print("\n======================================")
    print("Building FS2K V2 Embeddings")
    print("======================================")

    train_data, test_data = load_annotations()

    all_data = train_data + test_data

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    successful = 0
    skipped = 0

    for index, item in enumerate(all_data, start=1):

        image_name = item["image_name"]

        photo_path = get_photo_path(
            image_name
        )

        if photo_path is None:
            skipped += 1
            continue

        try:

            embedding = generate_siamese_embedding(
                photo_path
            )

            folder, filename = image_name.split("/")

            output_name = (
                f"{folder}_{filename}.npy"
            )

            output_path = os.path.join(
                OUTPUT_DIR,
                output_name
            )

            np.save(
                output_path,
                embedding
            )

            successful += 1

            if index % 100 == 0:

                print(
                    f"Processed {index}/{len(all_data)}"
                )

        except Exception as e:

            skipped += 1

            print(
                f"Skipped {image_name}: {e}"
            )

    print("\n======================================")
    print("FS2K V2 EMBEDDINGS COMPLETED")
    print("======================================")
    print(
        f"Successfully embedded : {successful}"
    )
    print(
        f"Skipped                : {skipped}"
    )
    print(
        f"Total                   : {len(all_data)}"
    )
    print(
        f"Output directory        : {OUTPUT_DIR}"
    )
    print("======================================")


if __name__ == "__main__":
    build_embeddings()