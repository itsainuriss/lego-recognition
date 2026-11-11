"""Load the YAML config (falls back to a tiny parser if PyYAML is missing)."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _mini_yaml(text: str) -> dict:
    """Enough YAML for our flat two-level config file."""
    out: dict = {}
    section = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        key, _, val = line.strip().partition(":")
        val = val.strip()
        if not raw.startswith((" ", "\t")):
            section = out.setdefault(key, {})
            continue
        if val in ("null", "~", ""):
            v = None
        elif val in ("true", "false"):
            v = val == "true"
        else:
            try:
                v = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                v = val
        section[key] = v
    return out


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        cfg = yaml.safe_load(text)
    except ImportError:
        cfg = _mini_yaml(text)
    return cfg


def resolve(p: str | Path) -> Path:
    """Paths in the config are relative to the project root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
