import pickle, pathlib, pandas as pd, faiss
from sentence_transformers import SentenceTransformer

IDX = pathlib.Path("data/indexes"); IDX.mkdir(parents=True, exist_ok=True)
CSV = IDX / "packages.csv"
I_FAISS = IDX / "libs_faiss.index"
I_META  = IDX / "libs_meta.pkl"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

if __name__ == "__main__":
    df = pd.read_csv(CSV).fillna("")
    texts = (df["name"] + " \n " + df["summary"] + " \n " + df["description"] + " \n " + df["keywords"]).tolist()
    model = SentenceTransformer(MODEL)
    X = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1]); index.add(X)
    faiss.write_index(index, str(I_FAISS))
    pickle.dump(df.to_dict(orient="records"), open(I_META, "wb"))
    print(f"Index saved: {I_FAISS}, meta: {I_META}, n={len(df)}")
