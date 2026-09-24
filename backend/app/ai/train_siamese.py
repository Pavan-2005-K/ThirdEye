import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.ai.siamese_model import SiameseNetwork
from app.ai.siamese_dataset import FS2KSiameseDataset


# =========================================================
# SETTINGS
# =========================================================

CSV_FILE = "data/online_dataset/FS2K/train_pairs.csv"

MODEL_DIR = "models"

# Save the improved model separately.
# This keeps the original baseline model safe.
MODEL_FILE = os.path.join(
    MODEL_DIR,
    "thirdeye_siamese_v2.pth"
)

BATCH_SIZE = 16

# Improved training run
EPOCHS = 10

LEARNING_RATE = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# CONTRASTIVE LOSS
# =========================================================

class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.0):

        super().__init__()

        self.margin = margin

    def forward(
        self,
        sketch_embedding,
        photo_embedding,
        label
    ):

        # Calculate distance between sketch and photo
        distance = torch.nn.functional.pairwise_distance(
            sketch_embedding,
            photo_embedding
        )

        # Positive pair:
        # We want the distance to become small.
        positive_loss = (
            label
            * torch.pow(distance, 2)
        )

        # Negative pair:
        # We want the distance to become larger than
        # the specified margin.
        negative_loss = (
            (1 - label)
            * torch.pow(
                torch.clamp(
                    self.margin - distance,
                    min=0.0
                ),
                2
            )
        )

        # Combine positive and negative losses
        loss = torch.mean(
            positive_loss + negative_loss
        )

        return loss


# =========================================================
# TRAINING
# =========================================================

def train():

    print("\n======================================")
    print("ThirdEye Siamese Model Training - V2")
    print("======================================")

    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    print("\nLoading FS2K dataset...")

    dataset = FS2KSiameseDataset(
        CSV_FILE
    )

    print(
        f"Training samples: {len(dataset)}"
    )

    # -----------------------------------------------------
    # DataLoader
    # -----------------------------------------------------

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    print(
        f"Batches per epoch: {len(dataloader)}"
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    print("\nLoading Siamese network...")

    model = SiameseNetwork(
        embedding_size=256
    )

    model = model.to(DEVICE)

    # -----------------------------------------------------
    # Loss function
    # -----------------------------------------------------

    criterion = ContrastiveLoss(
        margin=1.0
    )

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------

    print("\nStarting training...")

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        # -------------------------------------------------
        # Process batches
        # -------------------------------------------------

        for batch_index, (
            sketches,
            photos,
            labels
        ) in enumerate(dataloader):

            # Move data to CPU/GPU
            sketches = sketches.to(DEVICE)
            photos = photos.to(DEVICE)
            labels = labels.to(DEVICE)

            # Clear previous gradients
            optimizer.zero_grad()

            # Generate embeddings
            sketch_embeddings, photo_embeddings = model(
                sketches,
                photos
            )

            # Calculate contrastive loss
            loss = criterion(
                sketch_embeddings,
                photo_embeddings,
                labels
            )

            # Backpropagation
            loss.backward()

            # Update model weights
            optimizer.step()

            # Add current loss
            running_loss += loss.item()

            # Print every 10 batches
            if (batch_index + 1) % 10 == 0:

                print(
                    f"Batch {batch_index + 1}/"
                    f"{len(dataloader)} "
                    f"Loss: {loss.item():.4f}"
                )

        # -------------------------------------------------
        # Average epoch loss
        # -------------------------------------------------

        average_loss = (
            running_loss
            / len(dataloader)
        )

        print(
            f"Epoch {epoch + 1} "
            f"Average Loss: {average_loss:.4f}"
        )

    # -----------------------------------------------------
    # Save trained model
    # -----------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        MODEL_FILE
    )

    # -----------------------------------------------------
    # Training completed
    # -----------------------------------------------------

    print("\n======================================")
    print("Training completed!")
    print(f"Model saved to: {MODEL_FILE}")
    print("======================================")


# =========================================================
# PROGRAM ENTRY POINT
# =========================================================

if __name__ == "__main__":
    train()
    