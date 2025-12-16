import pickle, pathlib, json
from typing import List, Dict
import faiss
from sentence_transformers import SentenceTransformer

IDX = pathlib.Path("data/indexes")
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class Recommender:
    def __init__(self):
        self.model = SentenceTransformer(MODEL)
        self.index = faiss.read_index(str(IDX / "libs_faiss.index"))
        self.meta  = pickle.load(open(IDX / "libs_meta.pkl", "rb"))

    def recommend(self, query: str, top_k: int = 5) -> List[Dict]:
        import numpy as np
        v = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(v)
        scores, idxs = self.index.search(v, top_k)
        out = []
        for s, i in zip(scores[0], idxs[0]):
            if i < 0: continue
            m = self.meta[i]
            reason = m.get("summary") or (m.get("description","")[:200] + "...")
            links = {}
            try: links = json.loads(m.get("project_urls","{}"))
            except: pass
            out.append({"name": m["name"], "reason": reason, "links": links, "score": float(s)})
        return out
