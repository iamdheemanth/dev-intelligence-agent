# app/llm_client.py
import os, requests
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct:free")
URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")

class LLMUnavailable(Exception):
    pass

def ask_llm(prompt: str, timeout: int = 30) -> str:
    if not API_KEY:
        raise LLMUnavailable("OPENROUTER_API_KEY missing.")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Developer-Intelligence-Agent"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 800,
        "temperature": 0.2
    }
    try:
        r = requests.post(URL, headers=headers, json=payload, timeout=timeout)
    except Exception as e:
        raise LLMUnavailable(f"Network error: {e}")
    if r.status_code != 200:
        raise LLMUnavailable(f"Status {r.status_code}: {r.text}")
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return r.text
