# app/backend.py
"""
Project-scoped backend for Developer Intelligence Agent.

All functionality operates on uploaded / cloned projects under data/projects/<project_id>.

Endpoints (high level):
- GET  /health
- POST /projects             -> create (git clone) and index (background)
- POST /projects/upload      -> upload zip and index (background)
- GET  /projects             -> list projects & status
- POST /projects/{id}/repo_search   -> search project index
- POST /projects/{id}/summarize     -> summarize project topic (RAG + LLM)
- POST /projects/{id}/recommend     -> recommend libraries (project-scoped)
- POST /projects/{id}/triage        -> triage issue (project-scoped)
- POST /projects/{id}/review        -> review code (by path or pasted)
- POST /projects/{id}/docstrings    -> generate docstring (by path or pasted)
"""
import logging
import os
import json
import uuid
import subprocess
import zipfile
import io
import pathlib
import datetime
from typing import Optional

from fastapi import FastAPI, Depends, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path

# local imports (assumes same structure as your repo)
from app.auth import require_token
from app.recommender.recommender import Recommender
from app.triage.triage import classify_issue
from app.review.review import review_code
from app.docsgen.docstrings import docstring_for_func
from app.docsgen.summarize import summarize_topic
from app.repo_index.search import topk as repo_topk
from app.repo_index.build_repo_index import build_index_for_root

# -------------------------------------------------------------------
# Load environment
# -------------------------------------------------------------------
# Ensure backend/.env is loaded regardless of current working directory
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# ---------- logging ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("dev-intel-agent")

# ---------- FastAPI ----------
app = FastAPI(title="Developer Intelligence Agent (project-scoped)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------- globals ----------
PROJECTS_DIR = pathlib.Path("data/projects")
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

rec = Recommender()  # library recommender (keeps previous logic)

# dependency for endpoints
def auth_dep():
    return Depends(require_token)

# ---------- helpers ----------
def make_project_id() -> str:
    return f"project-{uuid.uuid4().hex[:8]}"

def project_src_dir(project_id: str) -> pathlib.Path:
    return PROJECTS_DIR / project_id / "src"

def project_index_dir(project_id: str) -> pathlib.Path:
    return PROJECTS_DIR / project_id / "indexes"

def project_status_file(project_id: str) -> pathlib.Path:
    return PROJECTS_DIR / project_id / "status.txt"

def ensure_project_exists(project_id: str):
    d = PROJECTS_DIR / project_id
    if not d.exists() or not d.is_dir():
        raise HTTPException(status_code=404, detail=f"project {project_id} not found")

def ensure_index_ready(project_id: str):
    idx = project_index_dir(project_id)
    if not (idx / "repo_faiss.index").exists():
        raise HTTPException(status_code=404, detail="project index not ready")

def index_project_on_disk(src_dir: pathlib.Path, project_id: str) -> int:
    """
    Build FAISS index for src_dir and write to data/projects/<project_id>/indexes
    Returns number of chunks indexed
    """
    out_dir = project_index_dir(project_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = build_index_for_root(src_dir, out_dir)
    meta = {
        "project_id": project_id,
        "chunks_indexed": n,
        "indexed_at": datetime.datetime.utcnow().isoformat()
    }
    (PROJECTS_DIR / project_id / "meta.json").write_text(json.dumps(meta))
    return n

def read_project_file(project_id: str, rel_path: str) -> str:
    """
    Read a file inside the project's src folder. rel_path should be relative (no ..)
    """
    # sanitize rel_path: disallow absolute and .. segments
    if os.path.isabs(rel_path) or ".." in pathlib.Path(rel_path).parts:
        raise HTTPException(status_code=400, detail="invalid path")
    full = project_src_dir(project_id) / rel_path
    if not full.exists() or not full.is_file():
        raise HTTPException(status_code=404, detail=f"file {rel_path} not found in project")
    try:
        return full.read_text(errors="ignore")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"could not read file: {e}")

# -------------------------------------------------------------------
# Optional Supabase integration (non-invasive)
# -------------------------------------------------------------------
# If SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are present, create client.
# This is optional — code will continue to work locally if keys are not set.
try:
    from supabase import create_client
    import hashlib
    import tempfile
    import shutil

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "project-files")

    supabase = None
    if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
        try:
            supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
            logger.info("Supabase client initialized")
        except Exception as e:
            supabase = None
            logger.warning("Failed to initialize Supabase client: %s", e)
    else:
        supabase = None
        logger.warning("Supabase env vars not set – DB/storage disabled")
except Exception:
    # supabase not installed or other import error — keep supabase = None
    supabase = None
    logger.debug("Supabase client library not available; continuing without DB/storage integration.")

def zip_dir_bytes(src_dir: pathlib.Path) -> bytes:
    """Zip a directory into bytes (used for archive upload)."""
    with tempfile.TemporaryDirectory() as td:
        tmp_zip = pathlib.Path(td) / "archive.zip"
        shutil.make_archive(str(tmp_zip.with_suffix("")), 'zip', root_dir=str(src_dir))
        return tmp_zip.read_bytes()

# ---------- request/response models ----------
class ProjectCreateRequest(BaseModel):
    git_url: str
    git_token: Optional[str] = None

class ProjectSearchRequest(BaseModel):
    query: str
    top_k: int = 5

class ProjectSummarizeRequest(BaseModel):
    topic: str

class RecommendRequestProject(BaseModel):
    query: str
    top_k: int = 5

class IssueRequest(BaseModel):
    title: str
    body: Optional[str] = ""

class ReviewRequest(BaseModel):
    # either code (pasted) OR path to a file inside project (relative)
    code: Optional[str] = None
    path: Optional[str] = None
    use_llm: bool = True

class DocstringRequestProject(BaseModel):
    code: Optional[str] = None
    path: Optional[str] = None

# ---------- routes ----------
@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------
# project management
# -----------------------
@app.post("/projects", dependencies=[Depends(require_token)])
def create_project(payload: ProjectCreateRequest, background_tasks: BackgroundTasks):
    git_url = payload.git_url
    git_token = payload.git_token or ""

    if not git_url:
        raise HTTPException(status_code=400, detail="git_url required")

    project_id = make_project_id()
    src_root = project_src_dir(project_id)
    src_root.mkdir(parents=True, exist_ok=True)
    (PROJECTS_DIR / project_id / "status.txt").write_text("indexing")

    def _work():
        try:
            if git_token and git_url.startswith("https://"):
                url = git_url.replace("https://", f"https://{git_token}@")
            else:
                url = git_url
            # shallow clone
            subprocess.check_call(["git", "clone", "--depth", "1", url, str(src_root)])
            index_project_on_disk(src_root, project_id)
            (PROJECTS_DIR / project_id / "status.txt").write_text("ready")
        except Exception as e:
            logger.exception("error indexing git repo")
            (PROJECTS_DIR / project_id / "status.txt").write_text("error: " + str(e))

    background_tasks.add_task(_work)

    # Non-invasive: ensure project row exists in Supabase (if available)
    if supabase:
        try:
            supabase.table("projects").upsert({
                "id": project_id,
                "owner_id": None,
                "name": git_url or project_id,
                "description": None
            }).execute()
            logger.info("ensured projects row exists for %s", project_id)
        except Exception as e:
            logger.warning("Could not upsert project row into supabase: %s", e)

    # After clone completes, archive & upload — do this in background to avoid blocking
    def _archive_and_upload():
        try:
            if src_root.exists():
                try:
                    zip_bytes = zip_dir_bytes(src_root)
                    sha = hashlib.sha256(zip_bytes).hexdigest()
                    archive_key = f"projects/{project_id}/archive_{sha}.zip"
                    if supabase:
                        try:
                            up = supabase.storage.from_(SUPABASE_BUCKET).upload(archive_key, zip_bytes)
                            logger.info("supabase archive upload response: %r", up)
                        except Exception as e:
                            logger.warning("Supabase upload exception (archive): %s", e)
                            up = None

                        # insert archive metadata (guarded)
                        try:
                            res = supabase.table("project_archives").insert({
                                "project_id": project_id,
                                "storage_key": archive_key,
                                "size": len(zip_bytes)
                            }).execute()
                            logger.info("project_archives insert response: %s", res)
                        except Exception as e:
                            logger.warning("Could not insert project_archives row: %s", e)
                except Exception as e:
                    logger.warning("Failed to create/upload archive for project %s: %s", project_id, e)
        except Exception:
            logger.debug("archive/upload background task encountered an issue", exc_info=True)

    background_tasks.add_task(_archive_and_upload)

    return {"project_id": project_id, "status": "indexing_started"}


@app.post("/projects/upload", dependencies=[Depends(require_token)])
def upload_project(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a zip file of a codebase. Returns project_id and indexes it in background.
    """
    project_id = make_project_id()
    dest = PROJECTS_DIR / project_id
    src_root = project_src_dir(project_id)
    src_root.mkdir(parents=True, exist_ok=True)
    (dest / "status.txt").write_text("indexing")

    zip_bytes = file.file.read()

    # Ensure a project row exists in the DB so foreign key constraints succeed
    if supabase:
        try:
            supabase.table("projects").upsert({
                "id": project_id,
                "owner_id": None,
                "name": project_id,
                "description": None
            }).execute()
            logger.info("ensured projects row exists for %s", project_id)
        except Exception as e:
            logger.warning("Could not ensure projects row in supabase: %s", e)

    def _work():
        try:
            z = zipfile.ZipFile(io.BytesIO(zip_bytes))
            z.extractall(src_root)
            index_project_on_disk(src_root, project_id)
            (dest / "status.txt").write_text("ready")
        except Exception as e:
            logger.exception("error indexing uploaded zip")
            (dest / "status.txt").write_text("error: " + str(e))

    background_tasks.add_task(_work)

    # Non-invasive: upload archive to Supabase storage and insert archive metadata
    try:
        if supabase:
            try:
                sha = hashlib.sha256(zip_bytes).hexdigest()
                archive_key = f"projects/{project_id}/archive_{sha}.zip"
                try:
                    up = supabase.storage.from_(SUPABASE_BUCKET).upload(archive_key, zip_bytes)
                    logger.info("supabase archive upload response: %r", up)
                except Exception as e:
                    logger.warning("Supabase upload exception for archive: %s", e)
                    up = None

                # Try to insert metadata even if up is not a dict-like object.
                try:
                    res = supabase.table("project_archives").insert({
                        "project_id": project_id,
                        "storage_key": archive_key,
                        "size": len(zip_bytes)
                    }).execute()
                    logger.info("project_archives insert response: %s", res)
                except Exception as e:
                    logger.warning("Could not insert project_archives row: %s", e)
            except Exception as e:
                logger.warning("Supabase upload error for archive: %s", e)
    except Exception:
        logger.debug("supabase not available for archive upload", exc_info=True)

    # Also schedule per-file upload and metadata insertion after indexing finishes.
    def _upload_files_and_metadata():
        try:
            z = zipfile.ZipFile(io.BytesIO(zip_bytes))
            for info in z.infolist():
                if info.is_dir():
                    continue
                try:
                    raw = z.read(info.filename)
                    file_sha = hashlib.sha256(raw).hexdigest()
                    safe_path = info.filename.replace("/", "__")
                    file_key = f"projects/{project_id}/files/{file_sha}_{safe_path}"
                    if supabase:
                        try:
                            upf = supabase.storage.from_(SUPABASE_BUCKET).upload(file_key, raw)
                            logger.info("supabase file upload response for %s: %r", info.filename, upf)
                        except Exception as e:
                            logger.warning("Failed to upload file %s: %s", info.filename, e)
                            continue
                        # try to extract text content for search (naive)
                        text_content = None
                        try:
                            text_content = raw.decode("utf-8", errors="ignore")
                        except Exception:
                            text_content = None
                        try:
                            resf = supabase.table("project_files").insert({
                                "project_id": project_id,
                                "path": info.filename,
                                "storage_key": file_key,
                                "size": len(raw),
                                "mime": "application/octet-stream",
                                "sha256": file_sha,
                                "content": text_content
                            }).execute()
                            logger.info("project_files insert response for %s: %s", info.filename, resf)
                        except Exception as e:
                            logger.warning("Could not insert project_files row for %s: %s", info.filename, e)
                except Exception as e:
                    logger.debug("skipping file in upload loop: %s", e)
        except Exception as e:
            logger.debug("per-file upload background task failed: %s", e, exc_info=True)

    background_tasks.add_task(_upload_files_and_metadata)

    return {"project_id": project_id, "status": "upload_received"}


@app.get("/projects", dependencies=[Depends(require_token)])
def list_projects():
    out = []
    for d in sorted(PROJECTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        pid = d.name
        status = "unknown"
        if (d / "status.txt").exists():
            status = (d / "status.txt").read_text()
        meta = {}
        if (d / "meta.json").exists():
            try:
                meta = json.loads((d / "meta.json").read_text())
            except Exception:
                meta = {}
        out.append({"project_id": pid, "status": status, "meta": meta})
    return {"projects": out}

# -----------------------
# project-scoped features
# -----------------------
@app.post("/projects/{project_id}/repo_search", dependencies=[Depends(require_token)])
def project_repo_search(project_id: str, req: ProjectSearchRequest):
    ensure_project_exists(project_id)
    ensure_index_ready(project_id)
    results = repo_topk(req.query, k=req.top_k, index_dir=str(project_index_dir(project_id)))
    return {"query": req.query, "results": results}

@app.post("/projects/{project_id}/summarize", dependencies=[Depends(require_token)])
def project_summarize(project_id: str, req: ProjectSummarizeRequest):
    ensure_project_exists(project_id)
    ensure_index_ready(project_id)
    return summarize_topic(req.topic, index_dir=str(project_index_dir(project_id)))

@app.post("/projects/{project_id}/recommend", dependencies=[Depends(require_token)])
def project_recommend(project_id: str, req: RecommendRequestProject):
    """
    Project-scoped recommend: for now we use the same recommender logic but expose it under the project route.
    (You can later extend it to use repo-specific metadata.)
    """
    ensure_project_exists(project_id)
    # optionally you could use project files to augment the query/context
    return {"query": req.query, "results": rec.recommend(req.query, req.top_k)}

@app.post("/projects/{project_id}/triage", dependencies=[Depends(require_token)])
def project_triage(project_id: str, issue: IssueRequest):
    ensure_project_exists(project_id)
    return classify_issue(issue.title, issue.body)

@app.post("/projects/{project_id}/review", dependencies=[Depends(require_token)])
def project_review(project_id: str, req: ReviewRequest):
    ensure_project_exists(project_id)
    # if path provided, read file from project
    code = None
    if req.path:
        code = read_project_file(project_id, req.path)
    elif req.code:
        code = req.code
    else:
        raise HTTPException(status_code=400, detail="provide either 'code' or 'path'")

    return review_code(code, req.use_llm)

@app.post("/projects/{project_id}/docstrings", dependencies=[Depends(require_token)])
def project_docstrings(project_id: str, req: DocstringRequestProject):
    ensure_project_exists(project_id)
    code = None
    if req.path:
        code = read_project_file(project_id, req.path)
    elif req.code:
        code = req.code
    else:
        raise HTTPException(status_code=400, detail="provide either 'code' or 'path'")

    return {"generated": docstring_for_func(code)}