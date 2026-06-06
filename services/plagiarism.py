import numpy as np
from .embedder import encode

DEFAULT_THRESHOLD = 0.75

def detect_plagiarism(source: str, candidates: list[str], threshold: float = DEFAULT_THRESHOLD) -> dict:
    all_texts = [source] + candidates
    embeddings = encode(all_texts)
    source_emb = embeddings[0]

    results = []
    for i, cand_emb in enumerate(embeddings[1:]):
        score = float(np.dot(source_emb, cand_emb))
        score = round(max(0.0, min(1.0, score)), 4)
        results.append({
            "candidate_index": i,
            "text": candidates[i],
            "similarity_score": score,
            "flagged": score >= threshold
        })

    flagged = [r for r in results if r["flagged"]]
    return {
        "threshold": threshold,
        "total_candidates": len(candidates),
        "flagged_count": len(flagged),
        "results": results
    }