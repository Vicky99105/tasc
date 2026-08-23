"""Role x candidate -> score. Zero model calls — this is the phase that makes the
architecture worth building. Every number here is a lookup or a mean.

score = relation x field_strength, where relation is met (1.0), narrower (full
credit, same as met), or adjacent (the taxonomy edge weight, capped by the term's
own registered evidence weight when the phrase matches one of its `related`
entries — a phrase the taxonomy already knows to be partial cannot be upgraded to
full credit just because extraction called it "met").

Experience = band_fit(sum(duration x own_weight x relevance)), relevance applied
per past role via occupation adjacency, not once per candidate — legal years earn
zero toward a DevOps band even if the person also held a DevOps title elsewhere.

Availability and location are zero-based: the worst case scores 0, not a floor
that hands every candidate free points.
"""
from __future__ import annotations

from dataclasses import dataclass

from engine.extract import CandidateRole, LinkResult, resolve_candidate_roles
from engine.ingest import Candidate, Role
from engine.rubric import Requirement, Rubric
from engine.taxonomy import FIELD_STRENGTH, Taxonomy, adjacent_weight, narrower_closure, normalize, resolve_surface

_CITY_TO_COUNTRY = {
    "dubai": "uae", "abu-dhabi": "uae", "sharjah": "uae",
    "riyadh": "saudi-arabia", "jeddah": "saudi-arabia",
    "cairo": "egypt", "alexandria": "egypt",
    "amman": "jordan", "beirut": "lebanon", "doha": "qatar",
}


@dataclass(frozen=True)
class RequirementScore:
    source: str
    kind: str
    tier: str
    term_ids: tuple[str, ...]
    score: float
    via: str  # "direct" | "narrower" | "adjacent" | "absent" | "band"
    field: str
    phrase: str
    stated_years: float | None = None
    implied_years: float | None = None
    conflict: bool | None = None


@dataclass(frozen=True)
class MatchScore:
    role_id: str
    candidate_id: str
    total: float
    required_component: float
    preferred_component: float
    availability_component: float
    location_component: float
    requirement_scores: tuple[RequirementScore, ...]
    availability_score: float
    location_score: float


def have_terms(link_result: LinkResult, taxonomy: Taxonomy) -> dict[str, tuple[float, str, str]]:
    """candidate's own confirmed skill terms -> (value, field, phrase), best mention wins."""
    best: dict[str, tuple[float, str, str]] = {}
    for m in link_result.mentions:
        v = 1.0 if m.relation == "met" else 0.6
        term = taxonomy.term(m.term_id)
        if term is not None:
            for phrase, cap in term.related:
                if normalize(phrase) == normalize(m.phrase):
                    v = min(v, cap)
                    break
        v = round(v * FIELD_STRENGTH.get(m.field, 1.0), 3)
        if v > best.get(m.term_id, (0.0, "", ""))[0]:
            best[m.term_id] = (v, m.field, m.phrase)
    return best


def single_term_lookup(
    term_id: str, have: dict[str, tuple[float, str, str]], blocked: tuple[str, ...], taxonomy: Taxonomy
) -> tuple[float, str, str, str]:
    if term_id in have:
        v, field, phrase = have[term_id]
        return v, "direct", field, phrase
    best: tuple[float, str, str, str] = (0.0, "absent", "", "")
    for ht, (v, field, phrase) in have.items():
        if ht in blocked:
            continue
        if term_id in narrower_closure(taxonomy, ht):
            if v > best[0]:
                best = (v, "narrower", field, phrase)
        w = adjacent_weight(taxonomy, "skill", ht, term_id)
        if w and round(v * w, 3) > best[0]:
            best = (round(v * w, 3), "adjacent", field, phrase)
    return best


def score_skill_requirement(req: Requirement, have: dict[str, tuple[float, str, str]], taxonomy: Taxonomy) -> RequirementScore:
    if not req.term_ids:
        return RequirementScore(source=req.source, kind=req.kind, tier=req.tier, term_ids=(),
                                 score=0.0, via="absent", field="", phrase="")
    candidates = [single_term_lookup(t, have, req.blocked_terms, taxonomy) for t in req.term_ids]
    v, via, field, phrase = max(candidates, key=lambda x: x[0])
    return RequirementScore(source=req.source, kind=req.kind, tier=req.tier, term_ids=req.term_ids,
                             score=v, via=via, field=field, phrase=phrase)


def role_occupation(role: Role, taxonomy: Taxonomy) -> str | None:
    t = resolve_surface(taxonomy, "occupation", role.title)
    return t.id if t else None


def relevant_years(candidate_roles: tuple[CandidateRole, ...], target_occupation: str | None, taxonomy: Taxonomy) -> float:
    if target_occupation is None:
        return 0.0
    total = 0.0
    for cr in candidate_roles:
        if cr.occupation_id is None:
            continue
        if cr.occupation_id == target_occupation:
            relevance = 1.0
        else:
            relevance = adjacent_weight(taxonomy, "occupation", cr.occupation_id, target_occupation) or 0.0
        total += cr.duration_years * cr.own_weight * relevance
    return round(total, 2)


def _band_fit(rel_years: float, lo: int, hi: int) -> float:
    if rel_years <= 0:
        return 0.0
    if lo <= rel_years <= hi:
        return 1.0
    if rel_years > hi:
        return 0.85
    return round(min(1.0, rel_years / lo), 3)


def score_experience_requirement(
    req: Requirement, role: Role, candidate: Candidate, taxonomy: Taxonomy
) -> RequirementScore:
    candidate_roles = resolve_candidate_roles(candidate, taxonomy)
    target = role_occupation(role, taxonomy)
    rel_years = relevant_years(candidate_roles, target, taxonomy)
    band = _band_fit(rel_years, req.experience_min, req.experience_max)
    stated = candidate.experience_years_stated_numeric
    conflict = stated is None or abs(stated - rel_years) > 2
    trace = "; ".join(
        f"{cr.title} {cr.duration_years:g}y x{cr.own_weight:.2f}" for cr in candidate_roles if cr.occupation_id
    ) or "no relevant role"
    return RequirementScore(
        source=req.source, kind="experience", tier=req.tier, term_ids=(),
        score=band, via="band", field="past_roles", phrase=trace,
        stated_years=stated, implied_years=rel_years, conflict=conflict,
    )


def avail_score(days: int | None) -> float:
    """Zero-based: unknown or 60+ days scores 0, not a floor that hands out free points."""
    if days is None:
        return 0.0
    if days <= 0:
        return 1.0
    if days <= 30:
        return 0.75
    if days <= 45:
        return 0.5
    if days <= 60:
        return 0.25
    return 0.0


def loc_score(role_location: str, candidate_city: str, candidate_country: str) -> float:
    """Zero-based: no city/country on file scores 0, not a partial default."""
    if not candidate_city:
        return 0.0
    if normalize(candidate_city) == normalize(role_location):
        return 1.0
    role_country = _CITY_TO_COUNTRY.get(normalize(role_location))
    if role_country and candidate_country and normalize(candidate_country) == role_country:
        return 0.5
    return 0.0


def score_pair(role: Role, rubric: Rubric, candidate: Candidate, link_result: LinkResult, taxonomy: Taxonomy) -> MatchScore:
    have = have_terms(link_result, taxonomy)
    scores: list[RequirementScore] = []
    for req in rubric.requirements:
        if req.kind == "experience":
            scores.append(score_experience_requirement(req, role, candidate, taxonomy))
        else:
            scores.append(score_skill_requirement(req, have, taxonomy))

    required = [s for s in scores if s.tier == "required"]
    preferred = [s for s in scores if s.tier == "preferred"]
    mr = sum(s.score for s in required) / len(required) if required else 0.0
    mp = sum(s.score for s in preferred) / len(preferred) if preferred else 0.0
    av = avail_score(candidate.notice_period_days)
    lc = loc_score(role.location, candidate.city, candidate.country)

    w = rubric.weights
    required_component = round(w["required"] * mr, 3)
    preferred_component = round(w["preferred"] * mp, 3)
    availability_component = round(w["availability"] * av, 3)
    location_component = round(w["location"] * lc, 3)
    total = round(required_component + preferred_component + availability_component + location_component, 1)

    return MatchScore(
        role_id=role.role_id, candidate_id=candidate.candidate_id, total=total,
        required_component=required_component, preferred_component=preferred_component,
        availability_component=availability_component, location_component=location_component,
        requirement_scores=tuple(scores), availability_score=av, location_score=lc,
    )


def score_all(
    roles: list[Role], rubrics: dict[str, Rubric], candidates: list[Candidate],
    links_by_id: dict[str, LinkResult], taxonomy: Taxonomy,
) -> list[MatchScore]:
    out = []
    for role in roles:
        rubric = rubrics[role.role_id]
        for candidate in candidates:
            link = links_by_id[candidate.candidate_id]
            out.append(score_pair(role, rubric, candidate, link, taxonomy))
    return out
