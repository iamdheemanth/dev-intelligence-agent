import yaml, pathlib, pickle, numpy as np
from sentence_transformers import SentenceTransformer

YAML = pathlib.Path("data/triage/teams.yaml")
OUT  = pathlib.Path("data/triage/label_centroids.pkl")
MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

if __name__ == "__main__":
    teams = yaml.safe_load(YAML.read_text()).get("teams", {})
    label_vecs = {}
    for name, spec in teams.items():
        text = spec.get("description","") + " " + " ".join(spec.get("keywords", []))
        v = MODEL.encode([text], convert_to_numpy=True)[0]
        v = v / (np.linalg.norm(v) + 1e-8)
        label_vecs[name] = v
    pickle.dump(label_vecs, open(OUT, "wb"))
    print(f"Saved {len(label_vecs)} label centroids -> {OUT}")
