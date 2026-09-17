import os

from app.ai.preprocessing import preprocess_image


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

processed = preprocess_image(
    image_path
)

print("Preprocessing successful!")
print("Shape:", processed.shape)
print("Data type:", processed.dtype)
print(
    "Minimum value:",
    processed.min()
)
print(
    "Maximum value:",
    processed.max()
)