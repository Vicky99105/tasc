"""Narrates a compiled rubric in English and interprets steering guidance — the
one model call inside the agent's compile_rubric tool. Term resolution already
happened deterministically in engine/rubric.py; this only ever proposes: the four
top-level weights, the shortlist threshold, and moving a named requirement to a
different tier. Nothing else exists to change.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from engine.ingest import Role
from engine.llm import LLMClient
from engine.prompt_template import extract_fenced_json, load_prompt_sections
from engine.rubric import Requirement, Rubric

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "03_role_link.md"


@dataclass(frozen=True)
class Unsupported:
    guidance: str
    reason: str


@dataclass(frozen=True)
class SteeringResult:
    rubric: Rubric
    diff_en: str
    unsupported: tuple[Unsupported, ...]


def render_steering_prompt(role: Role, rubric: Rubric, guidance_text: str) -> tuple[str, str, dict]:
    sections = load_prompt_sections(PROMPT_PATH)
    skill_reqs = [r for r in rubric.requirements if r.kind == "skill"]
    requirements_block = "\n".join(f"- {r.source} — {r.tier}" for r in skill_reqs)

    user = sections["user"]
    user = user.replace("{role_id}", role.role_id)
    user = user.replace("{role_title}", role.title)
    user = user.replace("{current_weights}", str(rubric.weights))
    user = user.replace("{current_threshold}", str(rubric.shortlist_threshold))
    user = user.replace("{requirements_block}", requirements_block)
    user = user.replace("{guidance_text}", guidance_text or "(none — first compile)")

    schema = extract_fenced_json(sections["schema"])
    schema["properties"]["retier"]["items"]["properties"]["requirement_source"]["enum"] = [
        r.source for r in skill_reqs
    ]
    return sections["system"], user, schema


def apply_steering(rubric: Rubric, raw: dict) -> Rubric:
    weights = {k: float(v) for k, v in raw["weights"].items()}
    total = sum(weights.values())
    if abs(total - 100) > 1e-6:
        raise ValueError(f"proposed weights must sum to 100, got {total}")

    retier_by_source = {r["requirement_source"]: r["new_tier"] for r in raw["retier"]}
    new_requirements = []
    for req in rubric.requirements:
        new_tier = retier_by_source.get(req.source, req.tier)
        new_requirements.append(replace(req, tier=new_tier) if new_tier != req.tier else req)

    from engine.rubric import _hash_requirements

    return Rubric(
        role_id=rubric.role_id,
        requirements=tuple(new_requirements),
        weights=weights,
        shortlist_threshold=float(raw["threshold"]),
        max_return=rubric.max_return,
        assessment_key=_hash_requirements(tuple(new_requirements)),
    )


def steer(llm: LLMClient, role: Role, rubric: Rubric, guidance_text: str) -> SteeringResult:
    system, user, schema = render_steering_prompt(role, rubric, guidance_text)
    raw = llm.call(system, user, schema)
    new_rubric = apply_steering(rubric, raw)
    unsupported = tuple(Unsupported(u["guidance"], u["reason"]) for u in raw["unsupported"])
    return SteeringResult(rubric=new_rubric, diff_en=raw["diff_en"], unsupported=unsupported)
