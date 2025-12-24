# app/triage/triage.py
"""
Enhanced triage module.

Exports:
- classify_issue(title, body) -> returns classifier output (keeps backward compatibility)
- suggest_actions_for_issue(title, body, classification=None) -> List[str]

Behavior:
- Tries to use an LLM (if configured) to produce a JSON array of actionable triage steps.
- Falls back to deterministic heuristics when LLM is unavailable or returns nothing useful.
- Optionally attempts to surface likely file hints using repo_topk if that helper is importable.
"""

import json
import logging
from typing import Optional, List, Any, Dict

logger = logging.getLogger("triage")

# Try to import an existing classifier implementation (non-destructive).
# If your repo has a classifier under a different name/path, adapt this import.
try:
    # Example: if you have app.triage.legacy_classifier.classify_issue
    from .legacy_classifier import classify_issue as _orig_classify  # type: ignore
except Exception:
    # Fallback stub classifier
    def _orig_classify(title: str, body: str) -> Dict[str, Any]:
        return {"label": "needs-triage", "score": 0.5}


# Optionally import repo_topk to provide file hints (project-aware triage).
try:
    from app.repo_index.search import topk as repo_topk  # type: ignore
except Exception:
    repo_topk = None  # type: ignore


# Try to import a local LLM client; if unavailable, fall back gracefully.
try:
    from app.llm_client import LLMClient, LLMUnavailable  # type: ignore
    _LLM = LLMClient()
except Exception:
    _LLM = None

    class LLMUnavailable(Exception):
        pass


def classify_issue(title: str, body: Optional[str] = "") -> Any:
    """
    Wrapper over the project's classifier (keeps API stable for backend).
    Returns whatever the underlying classifier returns, or a safe default on error.
    """
    try:
        return _orig_classify(title, body)
    except Exception as e:
        logger.exception("internal classifier error: %s", e)
        return {"label": "needs-triage", "score": 0.0, "error": str(e)}


def _llm_call(prompt: str, max_tokens: int = 500) -> str:
    """
    Normalize LLM client responses into a plain text string.
    Raises LLMUnavailable if the client is not configured.
    """
    if _LLM is None:
        raise LLMUnavailable("LLM not configured")
    resp = _LLM.chat_completion(prompt=prompt, model=None, max_tokens=max_tokens)
    # Normalize common response shapes
    if isinstance(resp, dict):
        # OpenAI-like: choices[0].message.content
        if "choices" in resp and resp["choices"]:
            first = resp["choices"][0]
            if isinstance(first, dict):
                # Chat-style
                if "message" in first and isinstance(first["message"], dict):
                    return first["message"].get("content", "") or first.get("text", "") or str(first)
                return first.get("text", "") or str(first)
        # Direct content
        if "text" in resp:
            return resp["text"]
        if "content" in resp:
            return resp["content"]
        return str(resp)
    return str(resp)


def suggest_actions_for_issue(title: str, body: str, classification: Optional[Any] = None) -> List[str]:
    """
    Return a short list of actionable triage steps.

    Strategy:
      1) Try LLM (returns JSON array of short steps)
      2) If LLM unavailable or returns nothing useful, run deterministic heuristics:
         - General reproduce/log/test steps
         - Targeted steps for auth/JWT/token/expiry/signature problems
         - 500/exception specific steps
      3) Optionally prepend likely file hints from repo_topk if available
    """
    actions: List[str] = []

    # 1) LLM attempt
    try:
        if _LLM is not None:
            prompt = (
                "You are a senior engineer. Given an issue title and body, return a JSON array of 3-6 "
                "concrete, ordered triage steps (each one short). Be specific (mention files, logs, tests if applicable).\n\n"
                f"Issue title: {title}\n\nIssue body:\n{body}\n\n"
                "Return JSON only, e.g. [\"step1\",\"step2\"]"
            )
            txt = _llm_call(prompt, max_tokens=600).strip()
            if not txt:
                raise ValueError("empty LLM response")
            try:
                parsed = json.loads(txt)
                if isinstance(parsed, list) and parsed:
                    return [str(x).strip() for x in parsed][:6]
            except Exception:
                # Not JSON: attempt to extract list-like lines
                lines = [ln.strip("-* \t") for ln in txt.splitlines() if ln.strip()]
                if lines:
                    return lines[:6]
    except LLMUnavailable:
        logger.debug("LLM not configured for triage suggestions")
    except Exception:
        logger.exception("LLM triage suggestion error (continuing to heuristics)")

    # 2) Deterministic heuristics fallback
    label = ""
    if isinstance(classification, dict):
        label = (classification.get("label") or classification.get("tag") or "").lower()
    elif isinstance(classification, str):
        label = classification.lower()

    t = f"{title}\n{body}".lower()

    fallback: List[str] = []
    # Always useful initial step
    fallback.append("Reproduce the issue locally and capture full logs and stack traces for the failing request.")

    # Project-aware hint placeholder (we'll try to populate below using repo_topk)
    file_hints: List[str] = []

    # 500 / exception specific
    if "500" in t or "exception" in t or "traceback" in t or "crash" in t:
        fallback.append("Inspect server logs and stack traces to identify the failing function and exact error.")

    # JWT / token / signature / expiry specific checks
    if ("jwt" in t) or ("signature" in t) or ("token" in t) or ("expiry" in t) or ("logged out" in t) or ("timeout" in t) or ("exp" in t):
        fallback.append("Check token expiry and TTL handling: verify token 'exp' claim and any custom expiry logic.")
        fallback.append("Verify signature keys and key rotation: ensure the signing key used for generation matches verification and environment secrets.")
        fallback.append("Check for clock drift/timezone issues between token issuer and verifier (NTP).")
        fallback.append("Inspect token parsing and encoding/decoding (base64/urlsafe, padding issues).")
        fallback.append("Add logging for token creation and validation paths (redact sensitive fields).")

    # broader auth/login/password checks when JWT not explicitly mentioned
    elif ("login" in t) or ("auth" in t) or ("password" in t):
        fallback.append("Inspect authentication code paths: password hashing, token creation, and input validation.")
        fallback.append("Check for hardcoded secrets or weak password hashing (use bcrypt/argon2).")
        fallback.append("Add logging around auth middleware and perform a replay test with valid credentials.")

    # Performance / latency
    if "slow" in t or "performance" in t or "latency" in t:
        fallback.append("Profile the endpoint; inspect DB queries and external calls, consider caching hot paths.")

    # If classifier indicates urgency, add test creation step
    if label in ("needs-triage", "bug", "urgent", "high"):
        fallback.append("Create a minimal reproducible test case and add an automated regression test.")

    # 3) Project-aware hint: include file names using repo_topk if available
    try:
        if repo_topk is not None:
            # choose a short query (title preferred)
            q = title if len(title) > 2 else body[:120]
            hits = []
            # repo_topk signature may vary; attempt safe calls
            try:
                hits = repo_topk(q, k=3, index_dir=None)
            except TypeError:
                try:
                    hits = repo_topk(q, k=3)
                except Exception:
                    hits = []
            except Exception:
                hits = []

            for h in hits:
                if isinstance(h, dict):
                    p = h.get("path") or h.get("file") or h.get("filename")
                    if p:
                        file_hints.append(p)
                else:
                    s = str(h)
                    # naive path-like detection
                    if "/" in s or "\\" in s:
                        file_hints.append(s)

            if file_hints:
                # insert file hint near the top (after repro step)
                hint_step = f"Inspect likely files: {', '.join(file_hints[:3])}"
                # avoid duplication: insert if not already present
                if hint_step not in fallback:
                    fallback.insert(1, hint_step)
    except Exception:
        logger.debug("repo_topk not available or failed for triage file hints", exc_info=True)

    # de-duplicate while preserving order and limit
    seen = set()
    suggested: List[str] = []
    for s in fallback:
        if s and s not in seen:
            suggested.append(s)
            seen.add(s)
        if len(suggested) >= 8:
            break

    # prefer a compact set (3-6)
    return suggested[:6]