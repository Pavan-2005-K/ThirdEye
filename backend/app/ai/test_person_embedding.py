import os
import numpy as np

from app.ai.person_embedding import create_person_embedding


image_folder = "uploads/sketches"

files = os.listdir(image_folder)

if not files:
    print("No images found.")
    exit()


image_path = os.path.join(
    image_folder,
    files[0]
)

print("Testing image:")
print(image_path)


embedding_path = create_person_embedding(
    person_id=1,
    image_path=image_path
)


print("\nPerson embedding created successfully!")
print("Embedding path:")
print(embedding_path)


embedding = np.load(embedding_path)

print("Embedding shape:", embedding.shape)
print("Embedding data type:", embedding.dtype)