import subprocess, tempfile, textwrap
from app.llm_client import ask_llm, LLMUnavailable

GUIDE = """
You are a senior Python reviewer. Produce 3-6 concise review bullets:
- correctness risks, readability, testability, errors
Include tiny code examples when helpful.
Keep it factual; do not invent APIs.
"""

def run_ruff_on_code(code: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        f.write(code); path = f.name
    out = subprocess.run(["ruff","--quiet","--format","concise", path],
                         capture_output=True, text=True)
    return out.stdout.strip()

def review_code(code: str, use_llm: bool = True):
    lint = run_ruff_on_code(code)
    llm = ""
    if use_llm:
        prompt = textwrap.dedent(f"{GUIDE}\n---\nCODE:\n{code}\n---\nOutput bullets only.")
        try:
            llm = ask_llm(prompt)
        except LLMUnavailable as e:
            llm = f"LLM unavailable: {e}"
    return {"lint": lint, "llm": llm}
