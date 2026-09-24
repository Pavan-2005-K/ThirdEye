import csv
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class FS2KSiameseDataset(Dataset):

    def __init__(self, csv_file):

        self.pairs = []

        with open(csv_file, "r", encoding="utf-8") as file:

            reader = csv.DictReader(file)

            for row in reader:

                image_name = row["image_name"]

                # Example:
                # photo1/image0110
                #
                # We use image0110 as the identity/group ID.
                image_id = image_name.split("/")[-1]

                self.pairs.append({
                    "image_name": image_name,
                    "image_id": image_id,
                    "sketch_path": row["sketch_path"],
                    "photo_path": row["photo_path"]
                })

        # Store unique identity IDs
        self.identity_ids = list(
            set(pair["image_id"] for pair in self.pairs)
        )

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print("\n======================================")
        print("FS2K Siamese Dataset")
        print("======================================")
        print(f"Total records       : {len(self.pairs)}")
        print(f"Unique identities   : {len(self.identity_ids)}")
        print("======================================")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):

        pair = self.pairs[index]

        sketch_path = pair["sketch_path"]
        image_id = pair["image_id"]

        # 50% positive pair
        is_positive = random.random() < 0.5

        if is_positive:

            label = 1.0

            selected_photo_path = pair["photo_path"]

        else:

            label = 0.0

            # Select a completely different identity
            negative_identity = random.choice([
                identity
                for identity in self.identity_ids
                if identity != image_id
            ])

            negative_candidates = [
                item
                for item in self.pairs
                if item["image_id"] == negative_identity
            ]

            negative_pair = random.choice(
                negative_candidates
            )

            selected_photo_path = negative_pair["photo_path"]

        sketch = Image.open(
            sketch_path
        ).convert("RGB")

        photo = Image.open(
            selected_photo_path
        ).convert("RGB")

        sketch = self.transform(sketch)
        photo = self.transform(photo)

        label = torch.tensor(
            label,
            dtype=torch.float32
        )

        return sketch, photo, label