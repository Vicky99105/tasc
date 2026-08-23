"""Role row -> requirements. Term resolution is a deterministic taxonomy lookup,
never a model call — P0's rubric compiler matched a requirement string against the
list that created the term with exact equality ("AWS/Azure" resolved, "AWS/Azure "
with a trailing space did not). This resolves through normalize() instead, exactly
as the linker does, so whitespace or casing in a CSV cell cannot silently blank a
requirement.

Weights are {"required": 80, "preferred": 10, "availability": 5, "location": 5} —
named for the CSV columns they come from, not a "skills" bucket. Experience is not
a separate weight; it is folded into the required-tier mean as one more item, on
equal footing with each required skill.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from engine.ingest import Role
from engine.taxonomy import Taxonomy, normalize

DEFAULT_WEIGHTS: dict[str, float] = {"required": 80, "preferred": 10, "availability": 5, "location": 5}
DEFAULT_SHORTLIST_THRESHOLD = 56.0
DEFAULT_MAX_RETURN = 50


@dataclass(frozen=True)
class Requirement:
    source: str
    kind: str  # "skill" | "experience"
    term_ids: tuple[str, ...]
    tier: str  # "required" | "preferred"
    substitutable: bool
    verifiable: bool
    blocked_terms: tuple[str, ...] = ()
    experience_min: int | None = None
    experience_max: int | None = None


@dataclass(frozen=True)
class Rubric:
    role_id: str
    requirements: tuple[Requirement, ...]
    weights: dict[str, float]
    shortlist_threshold: float
    max_return: int
    assessment_key: str

    def by_tier(self, tier: str) -> tuple[Requirement, ...]:
        return tuple(r for r in self.requirements if r.tier == tier)


def resolve_requirement_terms(taxonomy: Taxonomy, requirement_string: str, pillar: str = "skill") -> tuple[str, ...]:
    key = normalize(requirement_string)
    return tuple(
        sorted(
            t.id
            for t in taxonomy.by_pillar(pillar)
            if any(normalize(fr) == key for fr in t.from_requirements)
        )
    )


def _hash_requirements(requirements: tuple[Requirement, ...]) -> str:
    canonical = [
        {"source": r.source, "kind": r.kind, "term_ids": list(r.term_ids), "tier": r.tier}
        for r in requirements
    ]
    blob = json.dumps(canonical, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def compile_rubric(
    role: Role,
    taxonomy: Taxonomy,
    weights: dict[str, float] | None = None,
    shortlist_threshold: float = DEFAULT_SHORTLIST_THRESHOLD,
    max_return: int = DEFAULT_MAX_RETURN,
) -> Rubric:
    weights = dict(weights) if weights else dict(DEFAULT_WEIGHTS)
    total_weight = sum(weights.values())
    if abs(total_weight - 100) > 1e-6:
        raise ValueError(f"weights must sum to 100, got {total_weight}")

    groups: list[tuple[str, tuple[str, ...], str]] = []
    for s in role.required_skills:
        groups.append((s, resolve_requirement_terms(taxonomy, s), "required"))
    for s in role.nice_to_have_skills:
        groups.append((s, resolve_requirement_terms(taxonomy, s), "preferred"))

    requirements: list[Requirement] = []
    for i, (source, terms, tier) in enumerate(groups):
        blocked: set[str] = set()
        for j, (_, other_terms, _) in enumerate(groups):
            if j != i:
                blocked.update(other_terms)
        blocked -= set(terms)
        requirements.append(
            Requirement(
                source=source, kind="skill", term_ids=terms, tier=tier,
                substitutable=True, verifiable=len(terms) > 0, blocked_terms=tuple(sorted(blocked)),
            )
        )

    requirements.append(
        Requirement(
            source=f"{role.experience_min}-{role.experience_max} years of relevant {role.title} experience",
            kind="experience", term_ids=(), tier="required", substitutable=False, verifiable=True,
            experience_min=role.experience_min, experience_max=role.experience_max,
        )
    )

    return Rubric(
        role_id=role.role_id,
        requirements=tuple(requirements),
        weights=weights,
        shortlist_threshold=shortlist_threshold,
        max_return=max_return,
        assessment_key=_hash_requirements(tuple(requirements)),
    )
