from flask import Flask, request, jsonify
from services.similarity import compute_similarity, pairwise_similarity
from services.clustering import cluster_texts
from services.plagiarism import detect_plagiarism

app = Flask(__name__)

def bad_request(msg):
    return jsonify({"error": msg}), 400

@app.route("/api/similarity", methods=["POST"])
def similarity():
    data = request.get_json()
    if not data or "text1" not in data or "text2" not in data:
        return bad_request("Provide 'text1' and 'text2'")
    score = compute_similarity(data["text1"], data["text2"])
    return jsonify({"score": score, "interpretation": interpret(score)})

@app.route("/api/similarity/pairwise", methods=["POST"])
def pairwise():
    data = request.get_json()
    texts = data.get("texts", [])
    if len(texts) < 2:
        return bad_request("Provide at least 2 texts in 'texts' list")
    return jsonify({"pairs": pairwise_similarity(texts)})

@app.route("/api/cluster", methods=["POST"])
def cluster():
    data = request.get_json()
    texts = data.get("texts", [])
    if len(texts) < 2:
        return bad_request("Provide at least 2 texts")
    n = data.get("n_clusters")  # optional
    return jsonify(cluster_texts(texts, n))

@app.route("/api/plagiarism", methods=["POST"])
def plagiarism():
    data = request.get_json()
    source = data.get("source")
    candidates = data.get("candidates", [])
    threshold = data.get("threshold", 0.85)
    if not source or not candidates:
        return bad_request("Provide 'source' and 'candidates' list")
    return jsonify(detect_plagiarism(source, candidates, threshold))

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

def interpret(score):
    if score >= 0.9: return "near-identical"
    if score >= 0.75: return "highly similar"
    if score >= 0.5: return "moderately similar"
    if score >= 0.25: return "loosely related"
    return "dissimilar"

if __name__ == "__main__":
    from services.embedder import get_model
    get_model()
    app.run(debug=False, port=5000)