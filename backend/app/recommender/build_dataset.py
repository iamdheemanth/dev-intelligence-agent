import json, time, pathlib, requests, pandas as pd
PROC = pathlib.Path("data/indexes"); PROC.mkdir(parents=True, exist_ok=True)
OUT = PROC / "packages.csv"
PYPI_URL = "https://pypi.org/pypi/{pkg}/json"

STARTER = [
    "numpy","pandas","matplotlib","scikit-learn","scipy","requests","beautifulsoup4","lxml","selenium",
    "flask","fastapi","pydantic","sqlalchemy","pytest","nltk","spacy","transformers","torch",
    "tensorflow","opencv-python","pillow","tqdm","pyyaml","click","networkx","plotly","seaborn",
    "xgboost","lightgbm","statsmodels","scrapy","jupyter","notebook","pyspark","faiss-cpu","redis"
]

def fetch(pkg):
    r = requests.get(PYPI_URL.format(pkg=pkg), timeout=15)
    if r.status_code != 200:
        return {"name": pkg, "summary": "", "description": "", "keywords": "", "project_urls": "{}"}
    info = r.json().get("info", {})
    return {
        "name": info.get("name", pkg),
        "summary": info.get("summary", ""),
        "description": info.get("description", ""),
        "keywords": info.get("keywords", ""),
        "project_urls": json.dumps(info.get("project_urls", {}))
    }

if __name__ == "__main__":
    rows = []
    for p in STARTER:
        rows.append(fetch(p)); time.sleep(0.1)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Saved -> {OUT}")
