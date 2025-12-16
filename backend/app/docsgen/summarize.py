# app/docsgen/summarize.py
"""
Project summarization helper (RAG + LLM).

summarize_topic(topic, index_dir=None, k=5)
 - If index_dir is provided, search that index for context (repo-specific).
 - Calls an LLM (OpenRouter) if configured; otherwise returns a safe fallback summary
   built from the retrieved context.
 - Always returns a dictionary: {"context": [...], "summary": "..."}

This implementation is defensive: network/LLM errors do not raise, they return
a descriptive "LLM unavailable" summary and still provide the relevant context.
"""
import os
import json
import requests
from typing import List, Dict, Any, Optional

# Use the repo search helper to get project-specific context
try:
    from app.repo_index.search import topk as repo_topk
except Exception:
    # If import fails, fall back to a stub that returns empty context
    def repo_topk(q: str, k: int = 5, index_dir: str = "data/indexes"):
        return []

# Environment / OpenRouter defaults
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct:free")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def _sanitize_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("<s>", "").replace("</s>", "").strip()
    return s

def _build_prompt(topic: str, context_chunks: List[Dict[str, Any]]) -> str:
    # Take up to 5 chunks, include path and a truncated excerpt
    pieces = []
    for c in context_chunks[:5]:
        path = c.get("path", "<unknown>")
        text = c.get("text", "")
        excerpt = text.strip().replace("\n", " ")[:1000]
        pieces.append(f"PATH: {path}\n{excerpt}")
    context_text = "\n\n".join(pieces)
    prompt = (
        "You are a senior software engineer. Using ONLY the context below, write a concise 3-line summary "
        f"about the topic: '{topic}'. If the context is not relevant, reply exactly: 'No relevant context found.'\n\n"
        "CONTEXT:\n" + context_text + "\n\nSummary:"
    )
    return prompt

def _fallback_summary_from_context(context_chunks: List[Dict[str,Any]]) -> str:
    # Build a readable short summary from the context (no LLM)
    if not context_chunks:
        return "No relevant context found."
    # Use the first chunk(s) to form a short manual summary
    summaries = []
    for c in context_chunks[:3]:
        path = c.get("path","<unknown>")
        text = c.get("text","").strip().replace("\n"," ")
        summaries.append(f"[{path}] {text[:160]}...")
    return " | ".join(summaries)

def _call_openrouter(prompt: str, timeout: int = 25) -> str:
    """
    Calls OpenRouter (or other configured endpoint) and returns the plain text reply.
    Raises Exception on network/parse errors so caller can handle gracefully.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    payload = {
        "model": OPENROUTER_MODEL,
        # openrouter often expects messages; adapt if your provider expects a different shape
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    r = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"LLM error: {r.status_code} {r.text}")
    data = r.json()
    # Parse common shapes: choices[0].message.content or choices[0].text
    if isinstance(data, dict):
        if "choices" in data and len(data["choices"]) > 0:
            first = data["choices"][0]
            # openrouter style: first["message"]["content"]
            if isinstance(first, dict):
                if "message" in first and isinstance(first["message"], dict):
                    return first["message"].get("content", "") or first.get("text","")
                return first.get("text", "") or json.dumps(first)
    # fallback
    return str(data)

def summarize_topic(topic: str, index_dir: Optional[str] = None, k: int = 5) -> Dict[str,Any]:
    """
    Returns a dict: {"context": [...], "summary": "..."}
    - index_dir: optional path to project-specific index directory
    - k: number of context chunks to pull
    """
    idx = index_dir if index_dir else "data/indexes"
    result: Dict[str,Any] = {"context": [], "summary": ""}

    # 1) fetch context (project-specific if index_dir provided)
    try:
        context = repo_topk(topic, k=k, index_dir=idx)
    except TypeError:
        # older repo_topk signature might not take index_dir; try without it
        try:
            context = repo_topk(topic, k=k)
        except Exception:
            context = []
    except Exception as e:
        # if search fails, return error-ish summary but include empty context
        result["summary"] = f"Error fetching context: {e}"
        return result

    result["context"] = context

    if not context:
        result["summary"] = "No relevant context found."
        return result

    # 2) build prompt and call LLM (if available)
    prompt = _build_prompt(topic, context)
    try:
        text = _call_openrouter(prompt)
        text = _sanitize_text(text)
        if not text:
            # LLM returned empty text -> fallback to manual summary
            result["summary"] = _fallback_summary_from_context(context)
        else:
            result["summary"] = text
    except Exception as e:
        # LLM/network failed — return fallback but keep informative message
        fallback = _fallback_summary_from_context(context)
        result["summary"] = f"LLM unavailable: {e}. Fallback summary: {fallback}"
    return result
