import os
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from app.ai.siamese_model import SiameseNetwork
from app.ai.fs2k_dataset import get_photo_path, get_sketch_path, TEST_ANNOTATION


MODEL_FILE = "models/thirdeye_siamese_v2.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0).to(DEVICE)


def main():
    print("\n======================================")
    print("ThirdEye Siamese Model Evaluation")
    print("======================================")
    print(f"Device: {DEVICE}")

    # Load trained model
    model = SiameseNetwork(embedding_size=256).to(DEVICE)

    state_dict = torch.load(
        MODEL_FILE,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)
    model.eval()

    # Load test annotations
    with open(TEST_ANNOTATION, "r", encoding="utf-8") as file:
        test_data = json.load(file)

    print(f"Test records: {len(test_data)}")

    # Create photo gallery
    gallery = []

    print("\nGenerating photo embeddings...")

    with torch.no_grad():
        for index, item in enumerate(test_data):

            image_name = item["image_name"]

            photo_path = get_photo_path(image_name)

            if photo_path is None:
                continue

            photo = load_image(photo_path)

            photo_embedding = model.forward_once(photo)

            gallery.append({
                "image_name": image_name,
                "embedding": photo_embedding.cpu().numpy()[0]
            })

            if (index + 1) % 100 == 0:
                print(
                    f"Processed {index + 1}/{len(test_data)} photos"
                )

    print(f"\nGallery size: {len(gallery)}")

    rank1_correct = 0
    rank5_correct = 0
    evaluated = 0

    print("\nEvaluating sketches...")

    with torch.no_grad():

        for index, item in enumerate(test_data):

            image_name = item["image_name"]

            sketch_path = get_sketch_path(image_name)

            if sketch_path is None:
                continue

            sketch = load_image(sketch_path)

            sketch_embedding = model.forward_once(sketch)
            sketch_embedding = sketch_embedding.cpu().numpy()[0]

            results = []

            for candidate in gallery:

                similarity = float(
                    np.dot(
                        sketch_embedding,
                        candidate["embedding"]
                    )
                )

                results.append({
                    "image_name": candidate["image_name"],
                    "similarity": similarity
                })

            results.sort(
                key=lambda x: x["similarity"],
                reverse=True
            )

            top_1 = results[:1]
            top_5 = results[:5]

            rank1_names = [
                result["image_name"]
                for result in top_1
            ]

            rank5_names = [
                result["image_name"]
                for result in top_5
            ]

            if image_name in rank1_names:
                rank1_correct += 1

            if image_name in rank5_names:
                rank5_correct += 1

            evaluated += 1

            if (index + 1) % 100 == 0:
                print(
                    f"Evaluated {index + 1}/{len(test_data)} sketches"
                )

    rank1_accuracy = (
        rank1_correct / evaluated * 100
        if evaluated > 0 else 0
    )

    rank5_accuracy = (
        rank5_correct / evaluated * 100
        if evaluated > 0 else 0
    )

    print("\n======================================")
    print("EVALUATION RESULTS")
    print("======================================")
    print(f"Evaluated samples : {evaluated}")
    print(f"Rank-1 correct    : {rank1_correct}")
    print(f"Rank-5 correct    : {rank5_correct}")
    print("--------------------------------------")
    print(f"Rank-1 Accuracy   : {rank1_accuracy:.2f}%")
    print(f"Rank-5 Accuracy   : {rank5_accuracy:.2f}%")
    print("======================================\n")


if __name__ == "__main__":
    main()