import pickle, numpy as np
from sentence_transformers import SentenceTransformer

CENT  = pickle.load(open("data/triage/label_centroids.pkl","rb"))
MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
THRESH = 0.35

def classify_issue(title: str, body: str):
    text = (title or "") + "\n" + (body or "")
    v = MODEL.encode([text], convert_to_numpy=True)[0]
    v = v / (np.linalg.norm(v) + 1e-8)
    best = ("needs-triage", 0.0)
    for name, c in CENT.items():
        sc = float(np.dot(v, c))
        if sc > best[1]: best = (name, sc)
    label, score = best
    if score < THRESH: label = "needs-triage"
    return {"label": label, "score": score}
