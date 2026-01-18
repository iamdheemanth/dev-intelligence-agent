# app/recommender/recommender.py
"""
Library / tool recommender.

Exports:
- Recommender.recommend(query: str, top_k: int = 5) -> List[dict]

Behavior:
- Tries to call an LLM via OpenRouter to get structured library/tool recommendations.
- On success, returns a list of dictionaries with keys:
    - name
    - why
    - install
    - category
- On failure (no API key, model error, bad JSON, etc.), returns a static fallback list.

Environment variables:
- OPENROUTER_API_KEY  (required for LLM mode)
- OPENROUTER_MODEL    (e.g. "meta-llama/llama-3.3-70b-instruct:free")
- OPENROUTER_URL      (optional; defaults to OpenRouter chat completions endpoint)
"""

import os
import json
import logging
from typing import List, Dict, Any

import requests

logger = logging.getLogger("recommender")

# OpenRouter configuration
OPENROUTER_URL = os.getenv(
    "OPENROUTER_URL",
    "https://openrouter.ai/api/v1/chat/completions",
)
OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "meta-llama/llama-3.3-70b-instruct:free",
)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def _call_openrouter(prompt: str, timeout: int = 30) -> str:
    """
    Calls OpenRouter chat completions API and returns the text content.
    Raises on HTTP errors so caller can fall back to a static list.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set for recommender")

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": 700,
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    resp = requests.post(
        OPENROUTER_URL,
        json=payload,
        headers=headers,
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"LLM error: {resp.status_code} {resp.text}")

    data = resp.json()
    try:
        # OpenRouter: choices[0].message.content
        return data["choices"][0]["message"]["content"]
    except Exception:
        return str(data)


def _fallback_recommendations(top_k: int) -> List[Dict[str, Any]]:
    """
    Static fallback recommendations when LLM is unavailable or errors.
    """
    base = [
        {
            "name": "requests",
            "why": "Simple and widely used HTTP client for calling external APIs.",
            "install": "pip install requests",
            "category": "http",
        },
        {
            "name": "pydantic",
            "why": "Data validation and settings management using Python type hints.",
            "install": "pip install pydantic",
            "category": "validation",
        },
        {
            "name": "pytest",
            "why": "Feature-rich testing framework for Python projects.",
            "install": "pip install pytest",
            "category": "testing",
        },
        {
            "name": "loguru",
            "why": "Convenient, structured logging with a simple API.",
            "install": "pip install loguru",
            "category": "logging",
        },
        {
            "name": "rich",
            "why": "Beautiful terminal rendering for logs and CLI tools.",
            "install": "pip install rich",
            "category": "cli",
        },
    ]
    return base[:top_k]


class Recommender:
    """
    LLM-backed library / tool recommender.
    """

    def recommend(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Recommend libraries/tools for the given query.

        Returns a list of dicts:
        [
          {
            "name": "...",
            "why": "...",
            "install": "pip install ...",
            "category": "..."
          },
          ...
        ]
        """
        try:
            k = int(top_k)
        except Exception:
            k = 5
        k = max(1, min(k, 10))

        # 1) Try LLM via OpenRouter
        try:
            prompt = (
                "You are a senior Python engineer. Based on the user query, recommend Python "
                "libraries/tools that best fit the need.\n\n"
                f"User query: {query}\n\n"
                f"Return a JSON array of exactly {k} items. Each item must be an object with keys:\n"
                "  - name: the library/tool name (string)\n"
                "  - why: a short explanation (1–2 sentences) of why it is a good fit\n"
                "  - install: a one-line install command (e.g. 'pip install ...')\n"
                "  - category: a short label like 'auth', 'api', 'db', 'testing', etc.\n\n"
                "Return JSON only, with no extra commentary."
            )
            raw = _call_openrouter(prompt).strip()
            parsed = json.loads(raw)

            if isinstance(parsed, list) and parsed:
                results: List[Dict[str, Any]] = []
                for item in parsed[:k]:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name") or ""
                    if not name:
                        continue
                    results.append(
                        {
                            "name": name,
                            "why": item.get("why", "").strip(),
                            "install": item.get("install", "").strip(),
                            "category": item.get("category", "general").strip() or "general",
                        }
                    )
                if results:
                    return results

        except Exception as e:
            logger.warning("LLM-based recommend failed, using fallback. Error: %s", e)

        # 2) Fallback: never return an empty list
        return _fallback_recommendations(k)