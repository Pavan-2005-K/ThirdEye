import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


# Load pretrained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Remove the final classification layer
model = torch.nn.Sequential(*list(model.children())[:-1])

model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def generate_embedding(image_path: str):
    """
    Convert an image into a numerical feature vector.
    """

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        embedding = model(image_tensor)

    embedding = embedding.squeeze()

    return embedding.numpy()