import os

from app.ai.embedding import generate_embedding
from app.ai.similarity import calculate_similarity


image_folder = "uploads/sketches"

files = os.listdir(image_folder)

if len(files) < 2:
    print("At least 2 images are required for similarity testing.")
    exit()


image1 = os.path.join(image_folder, files[0])
image2 = os.path.join(image_folder, files[1])


print("Image 1:")
print(image1)

print("\nImage 2:")
print(image2)


embedding1 = generate_embedding(image1)
embedding2 = generate_embedding(image2)


similarity = calculate_similarity(
    embedding1,
    embedding2
)


percentage = similarity * 100


print("\nSimilarity calculation successful!")
print("Cosine similarity:", similarity)
print("Similarity percentage:", round(percentage, 2), "%")