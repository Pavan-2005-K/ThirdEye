import os

import numpy as np
from sqlalchemy.orm import Session

from app.ai.embedding import generate_embedding
from app.ai.similarity import calculate_similarity
from app.ai.online_dataset import search_online_dataset
from app.models.person import Person


def search_registered_people(
    db: Session,
    sketch_embedding,
    top_k: int = 5
):
    """
    Search the Admin Dataset (Registered People).
    """

    people = (
        db.query(Person)
        .filter(Person.embedding_path.isnot(None))
        .all()
    )

    results = []

    for person in people:

        if not person.embedding_path:
            continue

        if not os.path.exists(person.embedding_path):
            continue

        person_embedding = np.load(
            person.embedding_path
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
            "/" + person.photo_path.replace("\\", "/")
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
    """
    Search both Online Dataset and Admin Dataset.
    """

    if not os.path.exists(sketch_path):
        raise FileNotFoundError(
            f"Sketch not found: {sketch_path}"
        )

    # Generate sketch embedding only once
    sketch_embedding = generate_embedding(
        sketch_path
    )

    # Search Admin Dataset
    admin_results = search_registered_people(
        db=db,
        sketch_embedding=sketch_embedding,
        top_k=top_k
    )

    # Search Online Dataset
    online_results = search_online_dataset(
        sketch_path=sketch_path,
        top_k=top_k
    )

    # Combine both datasets
    combined_results = (
        admin_results +
        online_results
    )

    # Sort highest similarity first
    combined_results.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return combined_results[:top_k]