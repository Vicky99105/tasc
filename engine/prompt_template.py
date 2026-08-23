"""Shared parser for the Markdown prompt files under prompts/. Each file has a
System / User / Schema section; this reads them into plain strings plus the parsed
schema dict, and leaves placeholder substitution to the caller since fields differ
per prompt."""
from __future__ import annotations

import json
import re
from pathlib import Path


def load_prompt_sections(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"^## (System|User|Schema)\s*$", text, flags=re.MULTILINE)
    sections: dict[str, str] = {}
    for i in range(1, len(parts), 2):
        sections[parts[i].strip().lower()] = parts[i + 1].strip()
    return sections


def extract_fenced_json(text: str) -> dict:
    m = re.search(r"```json\n(.*?)\n```", text, re.S)
    if not m:
        raise ValueError("no fenced json block found")
    return json.loads(m.group(1))
