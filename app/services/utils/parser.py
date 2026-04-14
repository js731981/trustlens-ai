import ast
import json
import re
from typing import Any

_FALLBACK = {"ranked_products": []}

_LOG_PREFIX = "parse_llm_json"


def _log(step: str, detail: str = "") -> None:
    if detail:
        print(f"{_LOG_PREFIX}: [{step}] {detail}")
    else:
        print(f"{_LOG_PREFIX}: [{step}]")


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json / ``` fences often wrapped around LLM JSON."""
    s = re.sub(r"```\s*json\s*", "", text, flags=re.IGNORECASE)
    s = s.replace("```", "")
    return s.strip()


def _clean_llm_text(text: str) -> str:
    """Strip markdown fences and remove newline characters."""
    after_fences = _strip_markdown_fences(text)
    if "```" in text:
        _log("clean", "stripped ```json / ``` markdown fences")
    s = after_fences
    before_nl = s
    s = s.replace("\n", "").replace("\r", "")
    if before_nl != s:
        _log("clean", "removed newline characters")
    else:
        _log("clean", "no newlines present after fence strip")
    return s


def _apply_string_key_aliases(s: str) -> str:
    """Fix common mis-keyed fields via regex (key position only)."""
    replacements: list[tuple[re.Pattern[str], str, str]] = [
        (re.compile(r'"suggestions"\s*:'), '"ranked_products":', "suggestions → ranked_products"),
        (re.compile(r'"service_name"\s*:'), '"name":', "service_name → name"),
        (re.compile(r'"reasoning"\s*:'), '"reason":', "reasoning → reason"),
    ]
    out = s
    for pattern, repl, label in replacements:
        new_out, n = pattern.subn(repl, out)
        if n:
            _log("key_alias", f"{label} ({n} occurrence(s))")
            out = new_out
    return out


def _strip_trailing_commas(s: str) -> str:
    """Remove invalid trailing commas before } or ] (repeat until stable)."""
    out = s
    prev = None
    rounds = 0
    while prev != out:
        prev = out
        out = re.sub(r",\s*}", "}", out)
        out = re.sub(r",\s*]", "]", out)
        if out != prev:
            rounds += 1
    if rounds:
        _log("trailing_comma", f"removed ',}}' / ',]' patterns ({rounds} pass(es))")
    else:
        _log("trailing_comma", "no trailing commas before }} or ]")
    return out


def _extract_first_brace_slice(text: str) -> str | None:
    """First '{' through last '}' inclusive."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        _log("extract", "no usable {{ ... }} span (missing braces)")
        return None
    span = text[start : end + 1]
    _log("extract", f"slice from first '{{' to last '}}' ({len(span)} chars)")
    return span


def _first_balanced_json(text: str) -> str | None:
    """
    From the first '{' or '[', return the shortest balanced JSON span.
    Respects string literals so brackets inside strings are ignored.
    """
    start = -1
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start == -1:
        return None

    brace_depth = 0
    bracket_depth = 0
    if text[start] == "{":
        brace_depth = 1
    else:
        bracket_depth = 1

    in_string = False
    string_delim = ""
    escape = False

    for i in range(start + 1, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == string_delim:
                in_string = False
            continue

        if ch in ('"', "'"):
            in_string = True
            string_delim = ch
            continue

        if ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth -= 1

        if brace_depth < 0 or bracket_depth < 0:
            return None
        if brace_depth == 0 and bracket_depth == 0:
            return text[start : i + 1]

    return None


def _first_balanced_object(text: str) -> str | None:
    """
    Locate the first '{', then take the shortest balanced {...} slice.
    Respects string literals so braces inside strings do not affect depth.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    string_delim = ""
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == string_delim:
                in_string = False
            continue

        if ch in ('"', "'"):
            in_string = True
            string_delim = ch
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


_KEY_ALIASES = {
    "suggestions": "ranked_products",
    "service_name": "name",
    "reasoning": "reason",
    "product_name": "name",
}


def _normalize_llm_keys(obj: Any) -> Any:
    """Normalize common LLM key aliases recursively."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, val in obj.items():
            new_key = _KEY_ALIASES.get(key, key)
            out[new_key] = _normalize_llm_keys(val)
        return out
    if isinstance(obj, list):
        return [_normalize_llm_keys(item) for item in obj]
    return obj


def _coerce_to_dict(parsed: Any) -> dict:
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        return {"ranked_products": parsed}
    _log("coerce", "top-level JSON is not an object or array; using fallback shape")
    return dict(_FALLBACK)


def _finalize(parsed: Any) -> dict:
    return _normalize_llm_keys(_coerce_to_dict(parsed))


def _try_ast_literal_eval(text: str) -> Any | None:
    """
    Safe structured literal parse (no code execution).
    Only used on bounded text after json.loads fails.
    """
    if not text or len(text) > 2_000_000:
        _log("literal_eval", "skipped (empty or too large)")
        return None
    _log("literal_eval", "attempting ast.literal_eval (controlled, no builtins)")
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError) as e:
        _log("literal_eval", f"failed: {e}")
        return None


def _repair_pipeline(raw: str) -> str:
    """Ordered string repairs: clean → key aliases → trailing commas."""
    s = _clean_llm_text(raw)
    s = _apply_string_key_aliases(s)
    s = _strip_trailing_commas(s)
    return s


def parse_llm_json(response: str) -> dict:
    """
    Parse JSON from LLM text. Never raises: returns ``{"ranked_products": []}``
    on failure.

    Cleans fences/newlines, fixes common key names and trailing commas, extracts
    ``{...}`` by first/last brace, then tries ``json.loads`` and ``ast.literal_eval``.
    """
    if response is None:
        _log("fallback", "response is None")
        return dict(_FALLBACK)

    if not isinstance(response, str):
        _log("fallback", f"expected str, got {type(response).__name__}")
        return dict(_FALLBACK)

    repaired = _repair_pipeline(response)
    if not repaired.strip():
        _log("fallback", "empty after cleaning")
        return dict(_FALLBACK)

    candidates: list[str] = []
    seen: set[str] = set()

    def add_candidate(label: str, s: str | None) -> None:
        if not s:
            return
        t = s.strip()
        if not t or t in seen:
            return
        seen.add(t)
        _log("candidate", f"{label} ({len(t)} chars)")
        candidates.append(t)

    slice_span = _extract_first_brace_slice(repaired)
    if slice_span:
        add_candidate("brace_slice", slice_span)
    add_candidate("full_repaired", repaired)

    balanced = _first_balanced_json(repaired)
    add_candidate("first_balanced_json", balanced)

    balanced_obj = _first_balanced_object(repaired)
    add_candidate("first_balanced_object", balanced_obj)
    if balanced_obj and balanced_obj != repaired:
        nested = _first_balanced_json(balanced_obj)
        add_candidate("balanced_object_inner", nested)

    for i, candidate in enumerate(candidates):
        _log("json.loads", f"try candidate #{i + 1}")
        try:
            parsed = json.loads(candidate)
            _log("json.loads", "success")
            return _finalize(parsed)
        except Exception as e:  # noqa: BLE001 - LLM output may trigger any json.loads failure
            _log("json.loads", f"failed: {e}")

    _log("recovery", "json.loads failed for all candidates; trying ast.literal_eval on primary spans")

    for label, blob in (
        ("brace_slice", slice_span),
        ("full_repaired", repaired),
        ("first_balanced_object", balanced_obj),
    ):
        if not blob:
            continue
        lit = _try_ast_literal_eval(blob.strip())
        if lit is not None:
            _log("literal_eval", f"success on {label}")
            return _finalize(lit)

    _log("fallback", "all parsing attempts failed")
    return dict(_FALLBACK)
