import os
import numpy as np

from app.ai.siamese_embedding import generate_siamese_embedding


OUTPUT_DIR = "data/admin_dataset/embeddings_siamese_v2"


def create_siamese_person_embedding(
    person_id: int,
    photo_path: str
):
    if not os.path.exists(photo_path):
        raise FileNotFoundError(
            f"Photo not found: {photo_path}"
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    embedding = generate_siamese_embedding(
        photo_path
    )

    embedding_path = os.path.join(
        OUTPUT_DIR,
        f"person_{person_id}.npy"
    )

    np.save(
        embedding_path,
        embedding
    )

    return embedding_path