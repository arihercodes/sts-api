import numpy as np
from sentence_transformers import SentenceTransformer
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#MODEL_NAME = "all-MiniLM-L6-v2"
#MODEL_NAME = "paraphrase-MiniLM-L3-v2"
MODEL_NAME = "sentence-transformers/paraphrase-albert-small-v2"


print(f"Loading model: {MODEL_NAME} ...")
_model = SentenceTransformer(MODEL_NAME, device="cpu")
print("Model loaded.")

def get_model():
    return _model

def encode(texts: list[str]) -> np.ndarray:
    return _model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=4
    )