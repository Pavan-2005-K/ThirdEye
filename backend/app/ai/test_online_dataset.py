import os

from app.ai.online_dataset import (
    create_online_embedding,
    search_online_dataset
)


IMAGE_DIR = "data/online_dataset/images"


files = [
    file
    for file in os.listdir(IMAGE_DIR)
    if file.lower().endswith(".jpg")
]


if not files:
    print("No online dataset images found.")
    print("Add some test JPG images to:")
    print(IMAGE_DIR)
    exit()


print("Creating embeddings...")


for file in files:

    image_path = os.path.join(
        IMAGE_DIR,
        file
    )

    image_id = os.path.splitext(file)[0]

    embedding_path = create_online_embedding(
        image_path=image_path,
        image_id=image_id
    )

    print(
        f"Created embedding: {embedding_path}"
    )


print("\nOnline dataset embeddings created successfully!")