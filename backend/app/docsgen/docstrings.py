import ast
from app.llm_client import ask_llm, LLMUnavailable

PROMPT = (
    "Write a clear Python docstring for the function. "
    "Mention args, returns, side effects, and errors."
)

def docstring_for_func(src: str) -> str:
    try:
        tree = ast.parse(src)
    except Exception:
        return "Could not parse code."
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if not fns:
        return "No function definitions found."
    out = []
    for fn in fns:
        start, end = fn.lineno-1, fn.end_lineno
        code = "\n".join(src.splitlines()[start:end])
        try:
            ds = ask_llm(PROMPT + "\n\n" + code)
        except LLMUnavailable as e:
            ds = f"LLM unavailable: {e}"
        out.append(f'def {fn.name}(...):\n"""' + ds + '"""')
    return "\n\n".join(out)
