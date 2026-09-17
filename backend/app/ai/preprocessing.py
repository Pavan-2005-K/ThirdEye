import cv2
import numpy as np


def preprocess_image(image_path: str):
    """
    Load and preprocess a face sketch image.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Unable to read image")

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Resize to standard size
    resized = cv2.resize(
        gray,
        (224, 224)
    )

    # Normalize pixel values
    normalized = resized.astype(
        np.float32
    ) / 255.0

    return normalized