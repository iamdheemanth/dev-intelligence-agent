# app/recommender/code_recommender.py
"""
Code refactor recommender core logic.

Exports:
- suggest_refactors(scope=None, repo_root=".", limit_files=10) -> List[dict]
- suggest_refactors_for_file(file_path, repo_root, max_suggestions=3) -> List[dict]
"""

import os
import difflib
import json
import tempfile
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger("code_recommender")

# Try to import existing recommender pieces from repo (non-destructive)
try:
    # if there's a more complete Recommender in your repo, adapt or import that instead
    pass
except Exception:
    pass

# Try LLM client
try:
    from app.llm_client import LLMClient, LLMUnavailable  # type: ignore
    _LLM = LLMClient()
except Exception:
    _LLM = None
    class LLMUnavailable(Exception):
        pass

def _read_file(repo_root: str, rel_path: str) -> Optional[str]:
    full = os.path.join(repo_root, rel_path)
    if os.path.isfile(full):
        try:
            with open(full, "r", encoding="utf-8") as fh:
                return fh.read()
        except Exception:
            return None
    return None

def _llm_call(prompt: str, max_tokens: int = 800) -> str:
    if _LLM is None:
        raise LLMUnavailable("LLM client not configured")
    resp = _LLM.chat_completion(prompt=prompt, model=None, max_tokens=max_tokens)
    if isinstance(resp, dict):
        if "text" in resp:
            return resp["text"]
        if "content" in resp:
            return resp["content"]
        if "choices" in resp and resp["choices"]:
            c = resp["choices"][0]
            if isinstance(c, dict):
                return c.get("text") or c.get("message", {}).get("content") or str(c)
    return str(resp)

def extract_code_snippet(content: str, token: str, ctx: int = 3) -> str:
    lines = content.splitlines()
    for i, line in enumerate(lines):
        if token in line:
            start = max(0, i-ctx)
            end = min(len(lines), i+ctx+1)
            return "\n".join(lines[start:end])
    return ""

def _llm_high_level_refactor_suggestions(file_path: str, code_text: str, n: int = 3) -> List[Dict[str, Any]]:
    if _LLM is None:
        return []
    prompt = (
        "You are a conservative Python refactoring assistant. Given a file path and its code, produce up to "
        f"{n} small, safe, actionable refactor suggestions. Return JSON array of objects with keys: "
        "'explanation' and optionally 'example' (code snippet).\n\n"
        f"Path: {file_path}\n\n----CODE----\n{code_text[:20000]}\n----END----\n\n"
        "Return JSON only."
    )
    try:
        txt = _llm_call(prompt, max_tokens=700).strip()
        parsed = json.loads(txt)
        out = []
        if isinstance(parsed, list):
            for p in parsed[:n]:
                if isinstance(p, dict):
                    out.append({"file": file_path, "explanation": p.get("explanation", "")[:2000], "example": p.get("example", "")})
        return out
    except Exception:
        # fallback to naive parsing
        lines = [l.strip("-* \t") for l in txt.splitlines() if l.strip()] if 'txt' in locals() else []
        return [{"file": file_path, "explanation": l} for l in lines[:n]]

def _heuristic_scan_for_issues(repo_root: str, rel_path: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Deterministic heuristic scanner: search for hardcoded secrets, TODOs, naive token code, weak hashing, etc.
    """
    patterns = [
        ("hardcoded secret", ["demo-secret", "SECRET", "password_hash"], "Hardcoded secret found; move to env/config."),
        ("TODO / FIXME", ["TODO", "FIXME"], "Found TODO/FIXME; convert to tracked work or fix now."),
        ("naive token", ["generate_token(", "validate_token(", "token:"], "Custom token format was found; prefer standard libs (jwt/jose, authlib)."),
        ("weak password hashing", ["hashlib.sha256(", "sha256("], "Detected simple sha256 for passwords; use bcrypt/argon2 with salt."),
        ("hmac usage", ["hmac.new("], "HMAC present; ensure secrets are env-managed and use constant-time comparisons."),
    ]
    out = []
    targets = []
    if rel_path and ("\n" not in rel_path) and (not os.path.isabs(rel_path)):
        p = os.path.join(repo_root, rel_path)
        if os.path.isfile(p):
            targets.append((rel_path, p))
    else:
        # pick up to `limit` python files under repo_root
        count = 0
        for root, _, files in os.walk(repo_root):
            for fn in files:
                if fn.endswith(".py"):
                    rel = os.path.relpath(os.path.join(root, fn), repo_root)
                    targets.append((rel, os.path.join(root, fn)))
                    count += 1
                    if count >= limit:
                        break
            if count >= limit:
                break

    for rel, full in targets:
        try:
            content = open(full, "r", encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        for tag, toks, advice in patterns:
            for t in toks:
                if t in content:
                    snippet = extract_code_snippet(content, t)
                    out.append({"file": rel, "issue": tag, "explanation": advice, "snippet": snippet})
                    break
        if len(out) >= limit:
            break
    return out

# Placeholder original precision recommender function — if you already have one, keep using it.
def suggest_refactors_for_file(file_path: str, repo_root: str = ".", max_suggestions: int = 3) -> List[Dict[str, Any]]:
    """
    Best-effort: try to ask the recommender LLM (or an existing internal recommender) to produce unified diffs.
    This function is deliberately conservative: if no diffs can be produced, it returns [].
    """
    # If you already implement a robust per-file recommender, call it here.
    # For now, return empty to signal primary method did not produce diffs.
    return []

def suggest_refactors(scope: Optional[str] = None, repo_root: str = ".", limit_files: int = 10) -> List[Dict[str, Any]]:
    """
    Top-level: attempt precise suggestions via suggest_refactors_for_file; if none, return high-level suggestions.
    """
    files: List[str] = []
    # resolve scope
    if not scope:
        # pick some python files heuristically
        for root, _, filenames in os.walk(repo_root):
            for fn in filenames:
                if fn.endswith(".py"):
                    rel = os.path.relpath(os.path.join(root, fn), repo_root)
                    files.append(rel)
                    if len(files) >= limit_files:
                        break
            if len(files) >= limit_files:
                break
    else:
        # if scope is path and exists under repo_root
        if ("\n" not in scope) and (not os.path.isabs(scope)):
            cand = os.path.join(repo_root, scope)
            if os.path.exists(cand) and os.path.isfile(cand):
                files = [scope]
            elif os.path.isdir(cand):
                for root, _, filenames in os.walk(cand):
                    for fn in filenames:
                        if fn.endswith(".py"):
                            rel = os.path.relpath(os.path.join(root, fn), repo_root)
                            files.append(rel)
                            if len(files) >= limit_files:
                                break
                    if len(files) >= limit_files:
                        break
            else:
                # treat as snippet
                tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8")
                tmp.write(scope)
                tmp.flush()
                tmp.close()
                files = [os.path.basename(tmp.name)]
                repo_root = os.path.dirname(tmp.name)
        else:
            # treat as snippet
            tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8")
            tmp.write(scope)
            tmp.flush()
            tmp.close()
            files = [os.path.basename(tmp.name)]
            repo_root = os.path.dirname(tmp.name)

    all_suggestions: List[Dict[str, Any]] = []
    high_level: List[Dict[str, Any]] = []

    for f in files[:limit_files]:
        # try precise recommender first
        try:
            precise = suggest_refactors_for_file(f, repo_root, max_suggestions=3) or []
        except Exception:
            precise = []
        if precise:
            all_suggestions.extend(precise)
            continue

        # try LLM high-level suggestions per-file
        try:
            code_text = _read_file(repo_root, f) or ""
            llm_hl = _llm_high_level_refactor_suggestions(f, code_text, n=2)
            if llm_hl:
                high_level.extend(llm_hl)
                continue
        except Exception:
            logger.debug("LLM high-level suggestion failed for %s", f)

        # heuristic deterministic scan
        heur = _heuristic_scan_for_issues(repo_root, rel_path=f, limit=1)
        for h in heur:
            high_level.append({"file": h["file"], "explanation": h["explanation"], "example": h.get("snippet") or ""})

    # prefer precise suggestions if present; otherwise return high-level suggestions
    if all_suggestions:
        return all_suggestions
    return high_level