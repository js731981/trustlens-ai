from pathlib import Path
from string import Template

_TEMPLATE_ROOT = Path(__file__).resolve().parent / "templates"


class UnknownPromptTemplateError(LookupError):
    """Raised when ``template_id`` does not match a templates/ subfolder."""


def _validate_template_id(template_id: str) -> None:
    if not template_id or template_id in (".", ".."):
        raise UnknownPromptTemplateError("template_id must be a non-empty folder name.")
    if Path(template_id).name != template_id:
        raise UnknownPromptTemplateError("template_id must be a single path segment.")


def list_template_ids() -> list[str]:
    if not _TEMPLATE_ROOT.is_dir():
        return []
    ids: list[str] = []
    for path in sorted(_TEMPLATE_ROOT.iterdir()):
        if path.is_dir() and (path / "system.txt").is_file() and (path / "user.txt").is_file():
            ids.append(path.name)
    return ids


def render_prompt(template_id: str, variables: dict[str, str]) -> tuple[str, str]:
    _validate_template_id(template_id)
    available_ids = list_template_ids()
    if template_id not in available_ids:
        available = ", ".join(available_ids) or "(none)"
        msg = f"Unknown template_id={template_id!r}. Available: {available}"
        raise UnknownPromptTemplateError(msg)
    base = _TEMPLATE_ROOT / template_id
    system = (base / "system.txt").read_text(encoding="utf-8")
    user_template = Template((base / "user.txt").read_text(encoding="utf-8"))
    user = user_template.substitute(variables)
    return system.strip(), user.strip()
