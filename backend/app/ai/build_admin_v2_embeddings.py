import os
import numpy as np

from app.database.database import SessionLocal
from app.models.person import Person
from app.ai.siamese_embedding import generate_siamese_embedding


OUTPUT_DIR = "data/admin_dataset/embeddings_siamese_v2"


def build_admin_embeddings():

    print("\n======================================")
    print("Building Admin V2 Embeddings")
    print("======================================")

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    db = SessionLocal()

    try:
        people = (
            db.query(Person)
            .filter(Person.photo_path.isnot(None))
            .all()
        )

        print(f"Registered people: {len(people)}")

        successful = 0
        skipped = 0

        for person in people:

            photo_path = person.photo_path

            if not os.path.exists(photo_path):
                print(
                    f"Skipping Person {person.id}: "
                    f"photo not found"
                )
                skipped += 1
                continue

            try:

                embedding = generate_siamese_embedding(
                    photo_path
                )

                embedding_path = os.path.join(
                    OUTPUT_DIR,
                    f"person_{person.id}.npy"
                )

                np.save(
                    embedding_path,
                    embedding
                )

                successful += 1

                print(
                    f"Person {person.id}: "
                    f"{person.name} -> V2 embedding created"
                )

            except Exception as e:

                skipped += 1

                print(
                    f"Person {person.id}: "
                    f"failed - {e}"
                )

        print("\n======================================")
        print("ADMIN V2 EMBEDDINGS COMPLETED")
        print("======================================")
        print(
            f"Successfully embedded : {successful}"
        )
        print(
            f"Skipped                : {skipped}"
        )
        print(
            f"Output directory       : {OUTPUT_DIR}"
        )
        print("======================================")

    finally:
        db.close()


if __name__ == "__main__":
    build_admin_embeddings()