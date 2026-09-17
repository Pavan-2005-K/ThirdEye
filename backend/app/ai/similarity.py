import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(
    embedding1: np.ndarray,
    embedding2: np.ndarray
) -> float:
    """
    Calculate cosine similarity between two embeddings.

    Returns:
        Similarity score between 0 and 1.
    """

    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)

    score = cosine_similarity(
        embedding1,
        embedding2
    )[0][0]

    return float(score)