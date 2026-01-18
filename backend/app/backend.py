# app/backend.py
"""
Project-scoped backend for Developer Intelligence Agent.

This backend is intentionally thin: feature logic lives in feature modules:
- app/triage/*         -> triage logic + LLM/heuristic fallbacks
- app/recommender/*    -> code refactor & library recommender
- app/docsgen/*        -> summarizer wrapper (cleans results + LLM expansion)
- app/review/*, app.docsgen.docstrings -> hidden endpoints (include_in_schema=False)
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
import hashlib
import tempfile
import shutil
from typing import Optional, Any

from fastapi import FastAPI, Depends, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path

# -------------------------------------------------------------------
# Feature imports (delegated to feature modules)
# -------------------------------------------------------------------
# Auth dependency (keeps your existing auth)
from app.auth import require_token

# recommender - code refactor recommender
try:
    from app.recommender.code_recommender import suggest_refactors  # type: ignore
except Exception:
    suggest_refactors = None  # type: ignore

# library/tool recommender
try:
    from app.recommender.recommender import Recommender  # type: ignore
    rec = Recommender()
except Exception:
    rec = None  # type: ignore

# triage - classifier + suggestion helper
try:
    from app.triage.triage import classify_issue, suggest_actions_for_issue  # type: ignore
except Exception:
    # define minimal stubs so backend imports won't fail; feature module should replace these
    def classify_issue(title: str, body: Optional[str] = ""):
        return {"label": "needs-triage", "score": 0.0}

    def suggest_actions_for_issue(title: str, body: str, classification: Optional[Any] = None):
        return ["Reproduce and check logs."]

# review / docstrings helpers (kept; hidden from Swagger)
try:
    from app.review.review import review_code  # type: ignore
except Exception:
    def review_code(code: str, use_llm: bool = True):
        return {"review": "review module not installed"}

try:
    from app.docsgen.docstrings import docstring_for_func  # type: ignore
except Exception:
    def docstring_for_func(code: str):
        return "docstring generator not installed"

# summarize wrapper (cleans context and may call LLM)
try:
    from app.docsgen.summarize import summarize_topic  # type: ignore
except Exception:
    def summarize_topic(topic: str, index_dir: Optional[str] = None):
        return {"summary": f"No summarizer installed; topic: {topic}"}

# repo index search & build
try:
    from app.repo_index.search import topk as repo_topk  # type: ignore
except Exception:
    def repo_topk(q: str, k: int = 5, index_dir: Optional[str] = None):
        return []

try:
    from app.repo_index.build_repo_index import build_index_for_root  # type: ignore
except Exception:
    def build_index_for_root(src_dir, out_dir):
        return 0

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- globals ----------
PROJECTS_DIR = pathlib.Path("data/projects")
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

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
        "indexed_at": datetime.datetime.utcnow().isoformat(),
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
try:
    from supabase import create_client
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
        logger.debug("Supabase env vars not set – DB/storage disabled")
except Exception:
    supabase = None
    logger.debug("Supabase client library not available; continuing without DB/storage integration.")

def zip_dir_bytes(src_dir: pathlib.Path) -> bytes:
    """Zip a directory into bytes (used for archive upload)."""
    with tempfile.TemporaryDirectory() as td:
        tmp_zip = pathlib.Path(td) / "archive.zip"
        shutil.make_archive(str(tmp_zip.with_suffix("")), "zip", root_dir=str(src_dir))
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
    code: Optional[str] = None
    path: Optional[str] = None
    use_llm: bool = True

class DocstringRequestProject(BaseModel):
    code: Optional[str] = None
    path: Optional[str] = None

class RefactorRequestProject(BaseModel):
    scope: Optional[str] = None
    limit_files: Optional[int] = 10

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
            subprocess.check_call(["git", "clone", "--depth", "1", url, str(src_root)])
            index_project_on_disk(src_root, project_id)
            (PROJECTS_DIR / project_id / "status.txt").write_text("ready")
        except Exception as e:
            logger.exception("error indexing git repo")
            (PROJECTS_DIR / project_id / "status.txt").write_text("error: " + str(e))

    background_tasks.add_task(_work)

    if supabase:
        try:
            supabase.table("projects").upsert(
                {
                    "id": project_id,
                    "owner_id": None,
                    "name": git_url or project_id,
                    "description": None,
                }
            ).execute()
            logger.info("ensured projects row exists for %s", project_id)
        except Exception as e:
            logger.warning("Could not upsert project row into supabase: %s", e)

    def _archive_and_upload():
        try:
            if src_root.exists():
                try:
                    zip_bytes = zip_dir_bytes(src_root)
                    sha = hashlib.sha256(zip_bytes).hexdigest()
                    archive_key = f"projects/{project_id}/archive_{sha}.zip"
                    if supabase:
                        try:
                            up = supabase.storage.from_(SUPABASE_BUCKET).upload(
                                archive_key, zip_bytes
                            )
                            logger.info("supabase archive upload response: %r", up)
                        except Exception as e:
                            logger.warning("Supabase upload exception (archive): %s", e)
                            up = None
                        try:
                            res = supabase.table("project_archives").insert(
                                {
                                    "project_id": project_id,
                                    "storage_key": archive_key,
                                    "size": len(zip_bytes),
                                }
                            ).execute()
                            logger.info(
                                "project_archives insert response: %s", res
                            )
                        except Exception as e:
                            logger.warning(
                                "Could not insert project_archives row: %s", e
                            )
                except Exception as e:
                    logger.warning(
                        "Failed to create/upload archive for project %s: %s",
                        project_id,
                        e,
                    )
        except Exception:
            logger.debug(
                "archive/upload background task encountered an issue",
                exc_info=True,
            )

    background_tasks.add_task(_archive_and_upload)

    return {"project_id": project_id, "status": "indexing_started"}

@app.post("/projects/upload", dependencies=[Depends(require_token)])
def upload_project(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    project_id = make_project_id()
    dest = PROJECTS_DIR / project_id
    src_root = project_src_dir(project_id)
    src_root.mkdir(parents=True, exist_ok=True)
    (dest / "status.txt").write_text("indexing")

    zip_bytes = file.file.read()

    if supabase:
        try:
            supabase.table("projects").upsert(
                {
                    "id": project_id,
                    "owner_id": None,
                    "name": project_id,
                    "description": None,
                }
            ).execute()
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

    try:
        if supabase:
            try:
                sha = hashlib.sha256(zip_bytes).hexdigest()
                archive_key = f"projects/{project_id}/archive_{sha}.zip"
                try:
                    up = supabase.storage.from_(SUPABASE_BUCKET).upload(
                        archive_key, zip_bytes
                    )
                    logger.info("supabase archive upload response: %r", up)
                except Exception as e:
                    logger.warning(
                        "Supabase upload exception for archive: %s", e
                    )
                    up = None
                try:
                    res = supabase.table("project_archives").insert(
                        {
                            "project_id": project_id,
                            "storage_key": archive_key,
                            "size": len(zip_bytes),
                        }
                    ).execute()
                    logger.info(
                        "project_archives insert response: %s", res
                    )
                except Exception as e:
                    logger.warning(
                        "Could not insert project_archives row: %s", e
                    )
            except Exception as e:
                logger.warning("Supabase upload error for archive: %s", e)
    except Exception:
        logger.debug(
            "supabase not available for archive upload", exc_info=True
        )

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
                            upf = supabase.storage.from_(SUPABASE_BUCKET).upload(
                                file_key, raw
                            )
                            logger.info(
                                "supabase file upload response for %s: %r",
                                info.filename,
                                upf,
                            )
                        except Exception as e:
                            logger.warning(
                                "Failed to upload file %s: %s",
                                info.filename,
                                e,
                            )
                            continue
                        text_content = None
                        try:
                            text_content = raw.decode(
                                "utf-8", errors="ignore"
                            )
                        except Exception:
                            text_content = None
                        try:
                            resf = supabase.table("project_files").insert(
                                {
                                    "project_id": project_id,
                                    "path": info.filename,
                                    "storage_key": file_key,
                                    "size": len(raw),
                                    "mime": "application/octet-stream",
                                    "sha256": file_sha,
                                    "content": text_content,
                                }
                            ).execute()
                            logger.info(
                                "project_files insert response for %s: %s",
                                info.filename,
                                resf,
                            )
                        except Exception as e:
                            logger.warning(
                                "Could not insert project_files row for %s: %s",
                                info.filename,
                                e,
                            )
                except Exception as e:
                    logger.debug("skipping file in upload loop: %s", e)
        except Exception as e:
            logger.debug(
                "per-file upload background task failed: %s", e, exc_info=True
            )

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
    results = repo_topk(
        req.query,
        k=req.top_k,
        index_dir=str(project_index_dir(project_id)),
    )
    return {"query": req.query, "results": results}

@app.post("/projects/{project_id}/summarize", dependencies=[Depends(require_token)])
def project_summarize(project_id: str, req: ProjectSummarizeRequest):
    ensure_project_exists(project_id)
    ensure_index_ready(project_id)
    try:
        # summarize_topic now returns cleaned summary and optional detailed_summary
        resp = summarize_topic(
            req.topic, index_dir=str(project_index_dir(project_id))
        )
        # If the summarizer returned a dict as top-level, ensure it includes project_id for clarity
        if isinstance(resp, dict):
            resp.setdefault("project_id", project_id)
            return resp
        return {"project_id": project_id, "summary": resp}
    except Exception as e:
        logger.exception("Error in project_summarize: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/projects/{project_id}/recommend", dependencies=[Depends(require_token)])
def project_recommend(project_id: str, req: RecommendRequestProject):
    ensure_project_exists(project_id)
    if rec is None:
        raise HTTPException(
            status_code=501,
            detail="Recommender not installed or failed to import. Ensure app.recommender.recommender.Recommender is available.",
        )
    try:
        results = rec.recommend(req.query, req.top_k)
        return {"query": req.query, "results": results}
    except Exception as e:
        logger.exception("Error in project_recommend: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/projects/{project_id}/triage", dependencies=[Depends(require_token)])
def project_triage(project_id: str, issue: IssueRequest):
    ensure_project_exists(project_id)
    classification = classify_issue(issue.title, issue.body)
    # delegate suggestion generation to triage module
    try:
        suggested_actions = suggest_actions_for_issue(
            issue.title, issue.body, classification
        )
    except Exception:
        logger.exception("Error generating triage suggestions")
        suggested_actions = []
    return {"classification": classification, "suggested_actions": suggested_actions}

@app.post(
    "/projects/{project_id}/review",
    dependencies=[Depends(require_token)],
    include_in_schema=False,
)
def project_review(project_id: str, req: ReviewRequest):
    ensure_project_exists(project_id)
    code = None
    if req.path:
        code = read_project_file(project_id, req.path)
    elif req.code:
        code = req.code
    else:
        raise HTTPException(
            status_code=400, detail="provide either 'code' or 'path'"
        )
    try:
        return review_code(code, req.use_llm)
    except Exception as e:
        logger.exception("Error in project_review: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/projects/{project_id}/docstrings",
    dependencies=[Depends(require_token)],
    include_in_schema=False,
)
def project_docstrings(project_id: str, req: DocstringRequestProject):
    ensure_project_exists(project_id)
    code = None
    if req.path:
        code = read_project_file(project_id, req.path)
    elif req.code:
        code = req.code
    else:
        raise HTTPException(
            status_code=400, detail="provide either 'code' or 'path'"
        )
    try:
        return {"generated": docstring_for_func(code)}
    except Exception as e:
        logger.exception("Error in project_docstrings: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/projects/{project_id}/refactor", dependencies=[Depends(require_token)])
def project_refactor(project_id: str, req: RefactorRequestProject):
    ensure_project_exists(project_id)
    if suggest_refactors is None:
        raise HTTPException(
            status_code=501,
            detail=(
                "Refactor recommender not installed. "
                "Add app.recommender.code_recommender.suggest_refactors to enable this feature."
            ),
        )
    repo_root = str(project_src_dir(project_id))
    scope_arg = req.scope if req.scope else None
    limit_files = req.limit_files or 10
    try:
        # delegate to the recommender module (which should return precise diffs or high-level suggestions)
        suggestions = (
            suggest_refactors(
                scope=scope_arg, repo_root=repo_root, limit_files=limit_files
            )
            or []
        )
        return {"project_id": project_id, "suggestions": suggestions}
    except Exception as e:
        logger.exception(
            "Error while running project-scoped refactor suggestions: %s", e
        )
        raise HTTPException(status_code=500, detail=str(e))