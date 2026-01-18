# app/recommender/code_recommender.py
"""
Code refactor recommender (LLM-driven).

Exports:
- suggest_refactors(scope: Optional[str], repo_root: str, limit_files: int = 10) -> List[dict]

Behavior:
- If scope is a relative file path inside repo_root: reads that file and asks LLM for refactor suggestions.
- If scope is a code snippet (contains newline): asks LLM using the snippet.
- If scope is None: picks a few .py files from repo_root and asks LLM.

Returns list of dict suggestions:
[
  {"file": "src/auth.py", "explanation": "...", "example": "..."}
]
"""

import os
import json
import logging
import requests
from typing import Optional, List, Dict, Any

logger = logging.getLogger("code_recommender")

OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def _call_openrouter(prompt: str, timeout: int = 40) -> str:
    """
    Call OpenRouter chat completions and return message content as string.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 900,
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    r = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"LLM error: {r.status_code} {r.text}")
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return str(data)


def _read_file(repo_root: str, rel_path: str) -> Optional[str]:
    full = os.path.join(repo_root, rel_path)
    if not os.path.isfile(full):
        return None
    try:
        with open(full, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return None


def _pick_python_files(repo_root: str, limit_files: int) -> List[str]:
    files = []
    for root, _, fns in os.walk(repo_root):
        for fn in fns:
            if fn.endswith(".py"):
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, repo_root)
                files.append(rel)
                if len(files) >= limit_files:
                    return files
    return files


def _suggest_for_target(file_path: str, code: str) -> List[Dict[str, Any]]:
    """
    Ask LLM for refactor suggestions for a single file/snippet.
    Always returns a non-empty list on success.
    """
    prompt = (
        "You are a senior Python engineer. Provide refactor suggestions for the following code.\n\n"
        "Return JSON array with 2-5 items. Each item must be an object with keys:\n"
        "  - explanation: (string) what to change and why (short, actionable)\n"
        "  - example: (string) small code snippet showing the improved approach\n\n"
        f"File: {file_path}\n\n"
        "CODE:\n"
        f"{code}\n\n"
        "Return JSON only."
    )

    txt = _call_openrouter(prompt).strip()
    if not txt:
        return [{
            "file": file_path,
            "explanation": "LLM returned empty response. Check model / key / logs.",
            "example": "",
        }]

    # 1) Try to parse as JSON array
    try:
        parsed = json.loads(txt)
        suggestions: List[Dict[str, Any]] = []
        if isinstance(parsed, list):
            for item in parsed[:5]:
                if isinstance(item, dict) and item.get("explanation"):
                    suggestions.append({
                        "file": file_path,
                        "explanation": item.get("explanation", ""),
                        "example": item.get("example", ""),
                    })
        if suggestions:
            return suggestions
    except Exception:
        logger.warning("Refactor JSON parse failed for %s, using raw text as explanation", file_path)

    # 2) Fallback: treat raw text as one big explanation
    return [{
        "file": file_path,
        "explanation": txt,
        "example": "",
    }]


def suggest_refactors(scope: Optional[str] = None, repo_root: str = ".", limit_files: int = 10) -> List[Dict[str, Any]]:
    """
    Returns a list of refactor suggestions from LLM.
    Never returns empty unless repo_root is invalid AND scope is empty.
    """
    limit_files = max(1, min(int(limit_files or 10), 10))
    suggestions: List[Dict[str, Any]] = []

    targets: List[Dict[str, str]] = []

    # scope is code snippet (contains newline)
    if scope and "\n" in scope:
        targets.append({"file": "snippet.py", "code": scope})

    # scope is file path
    elif scope:
        code = _read_file(repo_root, scope)
        if code is None:
            # treat scope as snippet if file not found
            targets.append({"file": "snippet.py", "code": scope})
        else:
            targets.append({"file": scope, "code": code})

    # no scope: pick some py files
    else:
        for rel in _pick_python_files(repo_root, limit_files):
            code = _read_file(repo_root, rel)
            if code:
                targets.append({"file": rel, "code": code})

    if not targets:
        return [{
            "file": "",
            "explanation": "No files/snippet provided and no Python files found to analyze.",
            "example": "",
        }]

    for t in targets[:limit_files]:
        file_path = t["file"]
        code = t["code"][:20000]  # keep prompt bounded
        try:
            per_target = _suggest_for_target(file_path, code)
            suggestions.extend(per_target)
        except Exception as e:
            logger.warning("LLM refactor failed for %s: %s", file_path, e)
            suggestions.append({
                "file": file_path,
                "explanation": f"LLM error for this file: {e}",
                "example": "",
            })

    return suggestions