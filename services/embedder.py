from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"

# Load immediately when the module is imported
print(f"Loading model: {MODEL_NAME} ...")
_model = SentenceTransformer(MODEL_NAME)
print("Model loaded successfully.")

def get_model():
    return _model

def encode(texts: list[str]) -> np.ndarray:
    return _model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)