"""Mention -> term -> relation -> evidence field. Two model calls per candidate at
most: one for the explicit list fields (headline, skills), one for prose (projects,
extra_curriculars, education, certifications) — never both in one call, which was
tested and measured worse (P0: inconsistency on the projects field went from 60% to
21% once separated). A call is skipped when every field in its group is blank.

Occupation resolution is separate and lives at the bottom of this file: it is a
deterministic surface lookup against the taxonomy, zero model calls, because job
titles in this corpus are lexically clean enough that the taxonomy's own alias and
related lists already cover them (P2's review made sure of that).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from engine.ingest import Candidate
from engine.llm import LLMClient
from engine.prompt_template import extract_fenced_json, load_prompt_sections
from engine.taxonomy import Taxonomy, normalize

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "02_entity_link.md"

FIELD_GROUPS: dict[str, tuple[str, ...]] = {
    "list": ("headline", "skills"),
    "prose": ("projects", "extra_curriculars", "education", "certifications"),
}

_FIELD_TO_ATTR = {
    "headline": "headline",
    "skills": "skills_raw",
    "projects": "projects",
    "extra_curriculars": "extra_curriculars",
    "education": "education",
    "certifications": "certifications",
}


def field_text(candidate: Candidate, field: str) -> str:
    return getattr(candidate, _FIELD_TO_ATTR[field]) or ""


def _is_blank(v: str) -> bool:
    v = (v or "").strip()
    return not v or v == "-"


@dataclass(frozen=True)
class Mention:
    candidate_id: str
    term_id: str
    field: str
    phrase: str
    relation: str  # "met" | "adjacent"


@dataclass(frozen=True)
class RejectedMention:
    candidate_id: str
    group: str
    raw: dict
    reason: str


@dataclass(frozen=True)
class LinkResult:
    candidate_id: str
    mentions: tuple[Mention, ...]
    failed_groups: tuple[str, ...]
    rejected: tuple[RejectedMention, ...] = ()

    @property
    def partial(self) -> bool:
        return len(self.failed_groups) > 0


def render_vocabulary_block(taxonomy: Taxonomy, pillar: str = "skill") -> str:
    lines = []
    for t in sorted(taxonomy.by_pillar(pillar), key=lambda t: t.id):
        parts = [t.id]
        if t.surfaces:
            parts.append("surfaces: " + ", ".join(t.surfaces))
        if t.related:
            parts.append("related: " + ", ".join(f"{p}({w})" for p, w in t.related))
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def _profile_lines(candidate: Candidate, fields: tuple[str, ...]) -> list[str]:
    lines = []
    for f in fields:
        v = field_text(candidate, f)
        if not _is_blank(v):
            lines.append(f"{f}: {v.strip()}")
    return lines


def render_extraction_prompt(
    candidate: Candidate, group: str, taxonomy: Taxonomy
) -> tuple[str, str, dict] | None:
    if group not in FIELD_GROUPS:
        raise ValueError(f"unknown field group: {group!r}")
    fields = FIELD_GROUPS[group]
    lines = _profile_lines(candidate, fields)
    if not lines:
        return None

    sections = load_prompt_sections(PROMPT_PATH)
    term_ids = [t.id for t in taxonomy.by_pillar("skill")]
    vocab_block = render_vocabulary_block(taxonomy, "skill")

    user = sections["user"]
    user = user.replace("{candidate_id}", candidate.candidate_id)
    user = user.replace("{group_label}", group)
    user = user.replace("{profile_lines}", "\n".join(lines))
    user = user.replace("{vocabulary_block}", vocab_block)

    schema = extract_fenced_json(sections["schema"])
    props = schema["properties"]["mentions"]["items"]["properties"]
    props["term"]["enum"] = term_ids
    props["field"]["enum"] = list(fields)

    return sections["system"], user, schema


def _mention_rejection_reason(raw: dict, candidate: Candidate, fields: tuple[str, ...]) -> str | None:
    """A phrase must be a real, minimal substring of the field it claims to cite.
    Groundedness is checked here, not trusted — this is what P0's 537/537 measures.
    Returns None if the mention is valid, otherwise the reason it was dropped."""
    field = raw.get("field")
    if field not in fields:
        return f"field {field!r} not in this call's group {fields!r}"
    phrase = raw.get("phrase") or ""
    if not phrase:
        return "empty phrase"
    text = field_text(candidate, field)
    if phrase.lower() not in text.lower():
        return f"phrase not found verbatim in {field!r}"
    return None


def extract_group(
    llm: LLMClient, taxonomy: Taxonomy, candidate: Candidate, group: str
) -> tuple[list[Mention], list[RejectedMention]] | None:
    """Returns None if the call itself failed (network/parse) — caller marks partial.
    Returns ([], []) if the group was skipped (all fields blank) or the model found nothing."""
    rendered = render_extraction_prompt(candidate, group, taxonomy)
    if rendered is None:
        return [], []
    system, user, schema = rendered
    fields = FIELD_GROUPS[group]
    try:
        out = llm.call(system, user, schema)
    except Exception:
        return None
    mentions: list[Mention] = []
    rejected: list[RejectedMention] = []
    for raw in out["mentions"]:
        reason = _mention_rejection_reason(raw, candidate, fields)
        if reason is None:
            mentions.append(
                Mention(
                    candidate_id=candidate.candidate_id,
                    term_id=raw["term"],
                    field=raw["field"],
                    phrase=raw["phrase"],
                    relation=raw["relation"],
                )
            )
        else:
            rejected.append(RejectedMention(candidate_id=candidate.candidate_id, group=group, raw=raw, reason=reason))
    return mentions, rejected


def extract_candidate(llm: LLMClient, taxonomy: Taxonomy, candidate: Candidate) -> LinkResult:
    mentions: list[Mention] = []
    rejected: list[RejectedMention] = []
    failed: list[str] = []
    for group in FIELD_GROUPS:
        result = extract_group(llm, taxonomy, candidate, group)
        if result is None:
            failed.append(group)
        else:
            group_mentions, group_rejected = result
            mentions.extend(group_mentions)
            rejected.extend(group_rejected)
    return LinkResult(
        candidate_id=candidate.candidate_id,
        mentions=tuple(mentions),
        failed_groups=tuple(failed),
        rejected=tuple(rejected),
    )


def link_all(
    llm: LLMClient, taxonomy: Taxonomy, candidates: list[Candidate], concurrency: int = 16
) -> list[LinkResult]:
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        return list(ex.map(lambda c: extract_candidate(llm, taxonomy, c), candidates))


def links_to_dicts(results: list[LinkResult]) -> list[dict]:
    return [
        {
            "candidate_id": r.candidate_id,
            "partial": r.partial,
            "failed_groups": list(r.failed_groups),
            "mentions": [
                {"term": m.term_id, "field": m.field, "phrase": m.phrase, "relation": m.relation}
                for m in r.mentions
            ],
            "rejected": [
                {"group": rj.group, "reason": rj.reason, "raw": rj.raw} for rj in r.rejected
            ],
        }
        for r in results
    ]


def save_links(results: list[LinkResult], path: str) -> None:
    import json

    with open(path, "w", encoding="utf-8") as f:
        json.dump(links_to_dicts(results), f, indent=1, ensure_ascii=False)
        f.write("\n")


def load_links(path: str) -> list[LinkResult]:
    import json

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for d in data:
        mentions = tuple(
            Mention(candidate_id=d["candidate_id"], term_id=m["term"], field=m["field"],
                    phrase=m["phrase"], relation=m["relation"])
            for m in d["mentions"]
        )
        rejected = tuple(
            RejectedMention(candidate_id=d["candidate_id"], group=rj["group"], raw=rj["raw"], reason=rj["reason"])
            for rj in d.get("rejected", [])
        )
        out.append(
            LinkResult(
                candidate_id=d["candidate_id"], mentions=mentions,
                failed_groups=tuple(d["failed_groups"]), rejected=rejected,
            )
        )
    return out


# --- occupation resolution: deterministic, zero model calls -----------------------

def resolve_occupation(title: str, taxonomy: Taxonomy) -> tuple[str | None, float]:
    """(occupation_id, own_weight). own_weight is 1.0 for a full-credit surface
    match, the taxonomy's own related weight for a partial-credit title, or
    (None, 0.0) if the title is not in the taxonomy at all."""
    key = normalize(title)
    for t in taxonomy.by_pillar("occupation"):
        if key in (normalize(s) for s in t.surfaces):
            return t.id, 1.0
    for t in taxonomy.by_pillar("occupation"):
        for phrase, weight in t.related:
            if normalize(phrase) == key:
                return t.id, weight
    return None, 0.0


@dataclass(frozen=True)
class CandidateRole:
    title: str
    occupation_id: str | None
    own_weight: float
    duration_years: float
    is_current: bool


def resolve_candidate_roles(candidate: Candidate, taxonomy: Taxonomy) -> tuple[CandidateRole, ...]:
    out = []
    for pr in candidate.past_roles:
        occ_id, weight = resolve_occupation(pr.title, taxonomy)
        out.append(
            CandidateRole(
                title=pr.title,
                occupation_id=occ_id,
                own_weight=weight,
                duration_years=pr.duration_years,
                is_current=pr.is_current,
            )
        )
    return tuple(out)
