import os

import numpy as np
from sqlalchemy.orm import Session

from app.ai.embedding import generate_embedding
from app.ai.similarity import calculate_similarity
from app.models.person import Person


def search_registered_people(
    db: Session,
    sketch_path: str,
    top_k: int = 5
):
    """
    Compare a sketch against all registered people
    and return the highest similarity matches.
    """

    if not os.path.exists(sketch_path):
        raise FileNotFoundError(
            f"Sketch not found: {sketch_path}"
        )

    # Generate embedding for the uploaded sketch
    sketch_embedding = generate_embedding(sketch_path)

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
            "person_id": person.id,
            "name": person.name,
            "age": person.age,
            "gender": person.gender,
            "address": person.address,
            "phone": person.phone,
            "photo_path": person.photo_path,
            "similarity": round(similarity, 4),
            "similarity_percentage": round(
                similarity * 100,
                2
            )
        })

    # Highest similarity first
    results.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return results[:top_k]