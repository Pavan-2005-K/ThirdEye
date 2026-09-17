import os
import numpy as np

from app.ai.embedding import generate_embedding


EMBEDDING_DIR = "data/admin_dataset/embeddings"


def create_person_embedding(
    person_id: int,
    image_path: str
):
    """
    Generate and save an embedding for a registered person.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    os.makedirs(
        EMBEDDING_DIR,
        exist_ok=True
    )

    embedding = generate_embedding(image_path)

    embedding_path = os.path.join(
        EMBEDDING_DIR,
        f"person_{person_id}.npy"
    )

    np.save(
        embedding_path,
        embedding
    )

    return embedding_path