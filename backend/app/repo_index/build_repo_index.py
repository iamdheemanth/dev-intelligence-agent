# app/repo_index/build_repo_index.py
import pathlib, re, pickle
from sentence_transformers import SentenceTransformer
import faiss

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EXTS = {".py", ".md", ".txt"}
MAX_MB = 1.5
CHUNK = 300
OVERLAP = 60

def files(root: pathlib.Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS and p.stat().st_size <= MAX_MB*1024*1024:
            yield p

def chunk_text(s: str):
    s = re.sub(r"\s+", " ", s)
    for i in range(0, len(s), CHUNK-OVERLAP):
        yield s[i:i+CHUNK]

def build_index_for_root(root: pathlib.Path, out_dir: pathlib.Path, show_progress: bool = False) -> int:
    """
    Index files under `root` (pathlib.Path). Write faiss index + meta to out_dir.
    Returns number of chunks indexed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = []
    for f in files(root):
        try:
            t = f.read_text(errors="ignore")
        except Exception:
            continue
        for c in chunk_text(t):
            docs.append({"path": str(f.relative_to(root)), "text": c})

    if not docs:
        return 0

    model = SentenceTransformer(MODEL)
    X = model.encode([d["text"] for d in docs], convert_to_numpy=True, show_progress_bar=show_progress)
    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    faiss.write_index(index, str(out_dir / "repo_faiss.index"))
    pickle.dump(docs, open(out_dir / "repo_meta.pkl", "wb"))
    return len(docs)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".", help="root folder to index")
    p.add_argument("--out", default="data/indexes", help="output index dir")
    args = p.parse_args()
    n = build_index_for_root(pathlib.Path(args.root), pathlib.Path(args.out), show_progress=True)
    print(f"Repo indexed chunks: {n}")
