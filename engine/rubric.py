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


# ---------------------------------------------------------------------------
# Implicit-baseline skill guard
# ---------------------------------------------------------------------------
# These are generalised / table-stakes skills that any practitioner who holds
# the relevant *experience band* would implicitly demonstrate.  They are not
# checkable from a resume in isolation — a 6-year Customer Support specialist
# has "written communication" by definition; a 7-year Product Marketer has
# "content creation" by definition.  Requiring them verbatim in the skills
# cell creates systematic false negatives without adding signal.
#
# Rule: if a skill string (after normalisation) is in this set AND its tier
# in the CSV is "required", compile_rubric silently demotes it to "preferred".
# This does NOT remove the signal — it just stops it from halving a
# candidate's required-skill score for omitting an obvious professional norm.
#
# What stays Required (never in this set):
#   - Concrete tools:       Zendesk, SQL, Docker, AWS, IFRS, ATS tools
#   - Domain methods:       Go-to-market strategy, Full-cycle recruiting,
#                           Contract drafting, Financial modeling, CI/CD pipelines
#   - Specialised certs:    UAE commercial law, IFRS, Kubernetes
#   - Language / culture:   Arabic fluency (not obvious, genuinely differentiates)
#
# IMPLICIT_BASELINE_SKILLS (auto-demoted to preferred):
IMPLICIT_BASELINE_SKILLS: frozenset[str] = frozenset({
    # Communication
    "written communication",
    "communication skills",
    "verbal communication",
    "interpersonal skills",
    "presentation skills",
    # Execution / general craft
    "content creation",       # R006 — implicit for any Product Marketer
    "data visualization",     # R004 — implicit for any practicing analyst
    "troubleshooting",        # R007 — implicit for any support specialist
    "problem solving",
    "problem-solving",
    "attention to detail",
    "time management",
    "organizational skills",
    "analytical skills",
    "critical thinking",
    "teamwork",
    "collaboration",
    "adaptability",
    "multitasking",
    "customer service",       # implied by Customer Support Specialist title
    "reporting",              # generic, not a tool
})


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
    # True when compile_rubric demoted this from required -> preferred
    baseline_demoted: bool = False


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


def _is_implicit_baseline(source: str) -> bool:
    """Return True if this skill string is a known table-stakes / soft skill
    that should be demoted from required to preferred."""
    return normalize(source) in {normalize(s) for s in IMPLICIT_BASELINE_SKILLS}


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

    groups: list[tuple[str, tuple[str, ...], str, bool]] = []  # (source, terms, tier, demoted)
    for s in role.required_skills:
        demoted = _is_implicit_baseline(s)
        tier = "preferred" if demoted else "required"
        groups.append((s, resolve_requirement_terms(taxonomy, s), tier, demoted))
    for s in role.nice_to_have_skills:
        groups.append((s, resolve_requirement_terms(taxonomy, s), "preferred", False))

    requirements: list[Requirement] = []
    for i, (source, terms, tier, demoted) in enumerate(groups):
        blocked: set[str] = set()
        for j, (_, other_terms, _, _) in enumerate(groups):
            if j != i:
                blocked.update(other_terms)
        blocked -= set(terms)
        requirements.append(
            Requirement(
                source=source, kind="skill", term_ids=terms, tier=tier,
                substitutable=True, verifiable=len(terms) > 0,
                blocked_terms=tuple(sorted(blocked)),
                baseline_demoted=demoted,
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
