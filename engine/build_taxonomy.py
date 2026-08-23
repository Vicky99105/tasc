"""Generates a draft vocabulary per pillar via one model call each, from
prompts/01_vocabulary.md. The draft is not the artifact — a human reads every
row and patches it (engine/taxonomy_patches.py) before it becomes taxonomy.json.

A term's `narrower` list in the raw model output is a list of STRINGS (candidate
phrasings that are full-credit specialisations, e.g. SQL.narrower = ["PostgreSQL",
"MySQL"]) — those become extra exact-match surfaces on the term, not term-to-term
edges. A real term-to-term narrower edge (e.g. POSTGRESQL, which is itself a
requirement string and so its own term, narrower than SQL) only exists if a human
adds it during review — the model cannot know two separately-created terms are
related without being told.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from engine.llm import LLMClient
from engine.prompt_template import load_prompt_sections as _load_prompt_sections
from engine.taxonomy import Edge, Term

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "01_vocabulary.md"

PILLAR_CONFIG = {
    "skill": {
        "pillar_noun": "skill",
        "concept": "a skill, tool, language, domain or knowledge area",
        "out_of_scope": "Job titles are handled in a separate pillar.",
    },
    "occupation": {
        "pillar_noun": "occupation",
        "concept": "a job title / occupation",
        "out_of_scope": (
            "Skills, tools and certifications are handled in a separate pillar; "
            "a job title must never imply a skill."
        ),
    },
}


def load_prompt_sections(path: Path) -> dict[str, str]:
    return _load_prompt_sections(path)


def render_prompt(
    pillar: str, requirement_strings: list[str], candidate_strings: list[str]
) -> tuple[str, str, dict]:
    if pillar not in PILLAR_CONFIG:
        raise ValueError(f"unknown pillar: {pillar!r}")
    sections = load_prompt_sections(PROMPT_PATH)
    cfg = PILLAR_CONFIG[pillar]

    def sub(text: str) -> str:
        text = text.replace("{PILLAR_NOUN}", cfg["pillar_noun"])
        text = text.replace("{CONCEPT}", cfg["concept"])
        text = text.replace("{OUT_OF_SCOPE}", cfg["out_of_scope"])
        text = text.replace("{n_requirements}", str(len(requirement_strings)))
        text = text.replace("{requirement_strings}", "\n".join(f"- {s}" for s in requirement_strings))
        text = text.replace("{candidate_strings}", "\n".join(f"- {s}" for s in candidate_strings))
        return text

    system = sub(sections["system"])
    user = sub(sections["user"])
    m = re.search(r"```json\n(.*?)\n```", sections["schema"], re.S)
    if not m:
        raise ValueError("no fenced json schema found in prompt template")
    schema = json.loads(m.group(1))
    return system, user, schema


def raw_to_terms(pillar: str, raw: dict) -> tuple[list[Term], list[Edge]]:
    terms: list[Term] = []
    edges: list[Edge] = []
    known_ids = {t["id"] for t in raw["terms"]}
    for t in raw["terms"]:
        surfaces = tuple(dict.fromkeys([t["canonical"], *t["aliases"], *t["narrower"]]))
        related = tuple((r["phrase"], r["weight"]) for r in t["related"])
        terms.append(
            Term(
                id=t["id"],
                pillar=pillar,
                label=t["canonical"],
                surfaces=surfaces,
                related=related,
                from_requirements=tuple(t["from_requirements"]),
            )
        )
        for adj in t["adjacent"]:
            if adj["term"] not in known_ids:
                continue  # dangling reference in the draft — surfaced by validate_taxonomy, not silently kept
            pair = tuple(sorted((t["id"], adj["term"])))
            if not any((e.src, e.dst) == pair for e in edges):
                edges.append(Edge(pillar=pillar, src=pair[0], dst=pair[1], kind="adjacent", weight=adj["weight"]))
    return terms, edges


def build_pillar(
    llm: LLMClient, pillar: str, requirement_strings: list[str], candidate_strings: list[str]
) -> dict:
    system, user, schema = render_prompt(pillar, requirement_strings, candidate_strings)
    return llm.call(system, user, schema)
