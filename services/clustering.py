from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from .embedder import encode

def cluster_texts(texts: list[str], n_clusters: int = None) -> dict:
    embeddings = encode(texts)
    n = len(texts)

    if n_clusters is None:
        best_k, best_score = 2, -1
        for k in range(2, min(n, 8)):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(embeddings)
            s = silhouette_score(embeddings, labels)
            if s > best_score:
                best_score, best_k = s, k
        n_clusters = best_k

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings).tolist()

    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(str(label), []).append({
            "index": idx,
            "text": texts[idx]
        })

    return {
        "n_clusters": n_clusters,
        "assignments": labels,
        "clusters": clusters
    }