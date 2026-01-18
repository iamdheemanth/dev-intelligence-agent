# app/triage/triage.py
"""
Triage utilities (classifier + LLM suggested actions).

Exports:
- classify_issue(title, body) -> returns classifier output (or default)
- suggest_actions_for_issue(title, body, classification=None) -> List[str]

This module calls OpenRouter directly if configured and falls back to heuristics.
"""

import os
import json
import logging
import requests
from typing import Optional, List, Any, Dict

logger = logging.getLogger("triage")

# Optional existing classifier (keep your old logic if it exists)
try:
    from .legacy_classifier import classify_issue as _orig_classify  # type: ignore
except Exception:
    def _orig_classify(title: str, body: str) -> Dict[str, Any]:
        return {"label": "needs-triage", "score": 0.5}

# OpenRouter defaults
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def classify_issue(title: str, body: Optional[str] = "") -> Any:
    try:
        return _orig_classify(title, body or "")
    except Exception as e:
        logger.exception("classifier error: %s", e)
        return {"label": "needs-triage", "score": 0.0, "error": str(e)}


def _call_openrouter_json(prompt: str, timeout: int = 30) -> Dict[str, Any]:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    r = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"LLM error: {r.status_code} {r.text}")
    return r.json()


def _extract_text_from_openrouter(data: Dict[str, Any]) -> str:
    # OpenRouter usually: choices[0].message.content
    try:
        choices = data.get("choices") or []
        if choices:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message")
                if isinstance(msg, dict) and msg.get("content"):
                    return str(msg.get("content"))
                if first.get("text"):
                    return str(first.get("text"))
    except Exception:
        pass
    return ""


def suggest_actions_for_issue(title: str, body: str, classification: Optional[Any] = None) -> List[str]:
    """
    Returns 3-6 actionable triage steps.
    Prefers LLM response; falls back to deterministic heuristics if LLM unavailable.
    """
    # ---- 1) LLM path ----
    try:
        prompt = (
            "You are a senior engineer. Given an issue title and body, output a JSON array of 3-6 "
            "ordered, concrete triage steps. Each step should be short, specific, and actionable.\n\n"
            f"Issue title: {title}\n"
            f"Issue body: {body}\n\n"
            "Return JSON only like:\n"
            "[\"step 1\", \"step 2\", \"step 3\"]"
        )
        data = _call_openrouter_json(prompt)
        txt = _extract_text_from_openrouter(data).strip()
        if txt:
            # Try parse JSON list
            try:
                parsed = json.loads(txt)
                if isinstance(parsed, list) and parsed:
                    return [str(x).strip() for x in parsed][:6]
            except Exception:
                # fallback: parse lines
                lines = [ln.strip("-* \t") for ln in txt.splitlines() if ln.strip()]
                if lines:
                    return lines[:6]
    except Exception as e:
        logger.warning("LLM triage failed, using fallback: %s", e)

    # ---- 2) Heuristic fallback (never empty) ----
    label = ""
    if isinstance(classification, dict):
        label = (classification.get("label") or "").lower()
    t = f"{title}\n{body}".lower()

    actions: List[str] = []
    actions.append("Reproduce the issue and capture logs + stack traces for the failing request.")
    if "500" in t or "exception" in t or "traceback" in t:
        actions.append("Identify the failing function from the stack trace and add targeted logging around it.")
    if "jwt" in t or "token" in t or "signature" in t or "expiry" in t or "logged out" in t:
        actions.append("Check token expiry/TTL logic and verify the 'exp' handling matches expected behavior.")
        actions.append("Verify signing/verification keys (env secrets, key rotation) and check for clock drift.")
    if "login" in t or "auth" in t or "password" in t:
        actions.append("Inspect auth flow: password validation, token generation, and input validation for edge cases.")
    if label in ("needs-triage", "bug", "urgent", "high"):
        actions.append("Create a minimal reproduction + add a regression test that fails before the fix.")
    # cap
    return actions[:6]