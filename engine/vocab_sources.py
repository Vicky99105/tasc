"""Pulls the raw strings a taxonomy build reads. Pure functions, no model calls —
what goes INTO the vocabulary-generation prompt, never what comes out of it.

Roles create structure: role requirement strings and role titles are the input
that can create a new term. Candidate strings only attach to what already exists.
"""
from __future__ import annotations

from engine.ingest import Candidate, Role


def role_requirement_strings(roles: list[Role]) -> list[str]:
    seen: list[str] = []
    for r in roles:
        for s in list(r.required_skills) + list(r.nice_to_have_skills):
            if s not in seen:
                seen.append(s)
    return seen


def candidate_skill_strings(candidates: list[Candidate]) -> list[str]:
    seen: list[str] = []
    for c in candidates:
        raw = c.skills_raw
        if not raw or not raw.strip() or raw.strip() == "-":
            continue
        for s in raw.split(","):
            s = s.strip()
            if s and s not in seen:
                seen.append(s)
    return seen


def role_titles(roles: list[Role]) -> list[str]:
    return [r.title for r in roles]


def observed_job_titles(candidates: list[Candidate]) -> list[str]:
    seen: list[str] = []
    for c in candidates:
        for pr in c.past_roles:
            if pr.title not in seen:
                seen.append(pr.title)
    return seen
