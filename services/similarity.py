import numpy as np
from .embedder import encode

def compute_similarity(text1: str, text2: str) -> float:
    embeddings = encode([text1, text2])
    # With normalized embeddings, dot product = cosine similarity
    score = float(np.dot(embeddings[0], embeddings[1]))
    return round(max(0.0, min(1.0, score)), 4)

def pairwise_similarity(texts: list[str]) -> list[dict]:
    embeddings = encode(texts)
    results = []
    n = len(texts)
    for i in range(n):
        for j in range(i + 1, n):
            score = float(np.dot(embeddings[i], embeddings[j]))
            results.append({
                "text_a_index": i,
                "text_b_index": j,
                "score": round(max(0.0, min(1.0, score)), 4)
            })
    return results