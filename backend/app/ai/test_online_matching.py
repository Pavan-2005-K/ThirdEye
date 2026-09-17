import os

from app.ai.online_dataset import search_online_dataset


SKETCH_DIR = "uploads/sketches"


files = [
    file
    for file in os.listdir(SKETCH_DIR)
    if file.lower().endswith(
        (".jpg", ".jpeg", ".png", ".webp")
    )
]


if not files:
    print("No sketch images found.")
    exit()


sketch_path = os.path.join(
    SKETCH_DIR,
    files[0]
)

print("Testing sketch:")
print(sketch_path)

results = search_online_dataset(
    sketch_path=sketch_path,
    top_k=5
)

print("\nOnline Dataset Matches:")
print("--------------------------------")

for rank, result in enumerate(
    results,
    start=1
):
    print(
        f"{rank}. "
        f"{result['image_id']} - "
        f"{result['similarity_percentage']}%"
    )