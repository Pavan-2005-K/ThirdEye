import os
import numpy as np

from app.ai.embedding import generate_embedding
from app.ai.similarity import calculate_similarity


# ============================================================
# FS2K DATASET
# ============================================================

FS2K_ROOT = "data/online_dataset/FS2K"

FS2K_PHOTO_DIR = os.path.join(
    FS2K_ROOT,
    "photo"
)

FS2K_EMBEDDING_DIR = os.path.join(
    FS2K_ROOT,
    "embeddings"
)


# ============================================================
# SEARCH FS2K DATASET
# ============================================================

def search_fs2k_dataset(
    sketch_path: str,
    top_k: int = 5
):
    """
    Compare a user sketch against
    precomputed FS2K photo embeddings.
    """

    if not os.path.exists(sketch_path):
        raise FileNotFoundError(
            f"Sketch not found: {sketch_path}"
        )

    if not os.path.exists(FS2K_EMBEDDING_DIR):
        return []

    # Generate embedding for the uploaded sketch
    sketch_embedding = generate_embedding(
        sketch_path
    )

    results = []

    # Read all FS2K embeddings
    embedding_files = [
        file
        for file in os.listdir(FS2K_EMBEDDING_DIR)
        if file.endswith(".npy")
    ]

    for embedding_file in embedding_files:

        embedding_path = os.path.join(
            FS2K_EMBEDDING_DIR,
            embedding_file
        )

        # Example:
        # photo1_image0110.npy
        embedding_id = os.path.splitext(
            embedding_file
        )[0]

        # Convert:
        # photo1_image0110
        #
        # into:
        # photo1/image0110.jpg

        parts = embedding_id.split("_", 1)

        if len(parts) != 2:
            continue

        photo_folder = parts[0]
        photo_name = parts[1]

        photo_path = os.path.join(
            FS2K_PHOTO_DIR,
            photo_folder,
            photo_name + ".jpg"
        )

        if not os.path.exists(photo_path):
            continue

        try:

            dataset_embedding = np.load(
                embedding_path
            )

            similarity = calculate_similarity(
                sketch_embedding,
                dataset_embedding
            )

            results.append({
                "source": "FS2K",
                "image_id": embedding_id,
                "image_name": f"{photo_folder}/{photo_name}",
                "photo_path": photo_path,
                "image_url": (
                    "/fs2k-images/"
                    + photo_folder
                    + "/"
                    + photo_name
                    + ".jpg"
                ),
                "similarity": round(
                    similarity,
                    4
                ),
                "similarity_percentage": round(
                    similarity * 100,
                    2
                )
            })

        except Exception:
            continue

    # Highest similarity first
    results.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return results[:top_k]


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def search_online_dataset(
    sketch_path: str,
    top_k: int = 5
):
    """
    Existing function used by matching_service.py.

    It now searches the actual FS2K dataset.
    """

    return search_fs2k_dataset(
        sketch_path=sketch_path,
        top_k=top_k
    )