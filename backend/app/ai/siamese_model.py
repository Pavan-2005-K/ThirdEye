import torch
import torch.nn as nn
import torchvision.models as models


class SiameseNetwork(nn.Module):

    def __init__(self, embedding_size=256):

        super().__init__()

        # Pretrained ResNet18
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # Remove final classification layer
        self.backbone = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        # ResNet18 output = 512
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, embedding_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward_once(self, x):

        features = self.backbone(x)

        embedding = self.embedding(features)

        # Normalize embedding
        embedding = nn.functional.normalize(
            embedding,
            p=2,
            dim=1
        )

        return embedding

    def forward(self, sketch, photo):

        sketch_embedding = self.forward_once(sketch)

        photo_embedding = self.forward_once(photo)

        return sketch_embedding, photo_embedding