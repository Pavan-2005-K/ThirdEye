import torch
from PIL import Image
import torchvision.transforms as transforms

from app.ai.siamese_model import SiameseNetwork


MODEL_FILE = "models/thirdeye_siamese_v2.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Load trained V2 model
model = SiameseNetwork(
    embedding_size=256
)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
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


def generate_siamese_embedding(
    image_path: str
):

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(
        DEVICE
    )

    with torch.no_grad():

        embedding = model.forward_once(
            image_tensor
        )

    embedding = embedding.squeeze(0)

    return embedding.cpu().numpy()