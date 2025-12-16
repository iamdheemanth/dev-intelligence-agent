# app/repo_index/search.py
import pickle, pathlib
import faiss
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_index_and_meta(index_dir: str):
    idxdir = pathlib.Path(index_dir)
    if not (idxdir / "repo_faiss.index").exists() or not (idxdir / "repo_meta.pkl").exists():
        raise FileNotFoundError(f"Index files not found in {index_dir}")
    index = faiss.read_index(str(idxdir / "repo_faiss.index"))
    meta = pickle.load(open(idxdir / "repo_meta.pkl", "rb"))
    model = SentenceTransformer(MODEL)
    return index, meta, model

def topk(query: str, k: int = 5, index_dir: str = "data/indexes"):
    index, meta, model = load_index_and_meta(index_dir)
    v = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(v)
    D, I = index.search(v, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0: 
            continue
        d = meta[idx]
        results.append({"path": d["path"], "text": d["text"], "score": float(score)})
    return results
