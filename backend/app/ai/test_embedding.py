import os

from app.ai.embedding import generate_embedding


image_folder = "uploads/sketches"

files = os.listdir(image_folder)

if not files:
    print("No sketch images found.")
    exit()


image_path = os.path.join(
    image_folder,
    files[0]
)

print("Testing image:")
print(image_path)


embedding = generate_embedding(image_path)


print("Embedding generation successful!")
print("Embedding shape:", embedding.shape)
print("Embedding data type:", embedding.dtype)
print("First 10 values:")
print(embedding[:10])