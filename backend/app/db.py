import sqlite3, pathlib, json, datetime

DB_PATH = pathlib.Path("data/db.sqlite")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

SCHEMA = [
    "CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY, ts TEXT, feature TEXT, accepted INTEGER, payload TEXT)",
    "CREATE TABLE IF NOT EXISTS metrics  (id INTEGER PRIMARY KEY, ts TEXT, feature TEXT, detail TEXT)"
]

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    for stmt in SCHEMA:
        conn.execute(stmt)
    return conn

def log_metric(feature: str, detail: dict):
    conn = get_conn()
    conn.execute(
        "INSERT INTO metrics(ts, feature, detail) VALUES (?,?,?)",
        (datetime.datetime.utcnow().isoformat(), feature, json.dumps(detail)),
    )
    conn.commit(); conn.close()
