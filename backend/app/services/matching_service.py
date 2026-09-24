import os
import numpy as np

from sqlalchemy.orm import Session

from app.ai.siamese_embedding import generate_siamese_embedding
from app.ai.similarity import calculate_similarity
from app.models.person import Person


FS2K_ROOT = "data/online_dataset/FS2K"

FS2K_PHOTO_DIR = os.path.join(
    FS2K_ROOT,
    "photo"
)

FS2K_EMBEDDING_DIR = os.path.join(
    FS2K_ROOT,
    "embeddings_siamese_v2"
)

ADMIN_EMBEDDING_DIR = (
    "data/admin_dataset/embeddings_siamese_v2"
)


def search_registered_people(
    db: Session,
    sketch_embedding,
    top_k: int = 5
):

    people = (
        db.query(Person)
        .all()
    )

    results = []

    for person in people:

        embedding_path = os.path.join(
            ADMIN_EMBEDDING_DIR,
            f"person_{person.id}.npy"
        )

        # Skip people without a V2 embedding
        if not os.path.exists(
            embedding_path
        ):
            continue

        try:

            person_embedding = np.load(
                embedding_path
            )

            similarity = calculate_similarity(
                sketch_embedding,
                person_embedding
            )

            results.append({
                "source": "admin_dataset",
                "person_id": person.id,
                "name": person.name,
                "age": person.age,
                "gender": person.gender,
                "address": person.address,
                "phone": person.phone,
                "photo_url": (
                    "/"
                    + person.photo_path.replace(
                        "\\",
                        "/"
                    )
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

    results.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return results[:top_k]


def search_fs2k_dataset(
    sketch_embedding,
    top_k: int = 5
):

    if not os.path.exists(
        FS2K_EMBEDDING_DIR
    ):
        return []

    results = []

    embedding_files = [
        file
        for file in os.listdir(
            FS2K_EMBEDDING_DIR
        )
        if file.endswith(".npy")
    ]

    for embedding_file in embedding_files:

        embedding_id = os.path.splitext(
            embedding_file
        )[0]

        parts = embedding_id.split(
            "_",
            1
        )

        if len(parts) != 2:
            continue

        photo_folder = parts[0]
        photo_name = parts[1]

        photo_path = os.path.join(
            FS2K_PHOTO_DIR,
            photo_folder,
            photo_name + ".jpg"
        )

        if not os.path.exists(
            photo_path
        ):
            continue

        embedding_path = os.path.join(
            FS2K_EMBEDDING_DIR,
            embedding_file
        )

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
                "image_name": (
                    f"{photo_folder}/{photo_name}"
                ),
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

    results.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return results[:top_k]


def search_all_datasets(
    db: Session,
    sketch_path: str,
    top_k: int = 5
):

    if not os.path.exists(
        sketch_path
    ):
        raise FileNotFoundError(
            f"Sketch not found: {sketch_path}"
        )

    # Generate the user's sketch embedding
    # using the trained Siamese V2 model.
    sketch_embedding = (
        generate_siamese_embedding(
            sketch_path
        )
    )

    # Search admin dataset
    admin_results = search_registered_people(
        db=db,
        sketch_embedding=sketch_embedding,
        top_k=top_k
    )

    # Search FS2K dataset
    online_results = search_fs2k_dataset(
        sketch_embedding=sketch_embedding,
        top_k=top_k
    )

    # Combine both datasets
    combined_results = (
        admin_results
        + online_results
    )

    combined_results.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return combined_results[:top_k]