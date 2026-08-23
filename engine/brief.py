"""Finalist selection, gap/strength detection, and the one model call per role
that turns pre-selected gaps into natural questions. Gap SELECTION is deterministic
code — the model only phrases what it's given, in the same order, one question per
gap. It cannot invent a gap that wasn't provided.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from engine.ingest import Candidate, Role
from engine.llm import LLMClient
from engine.match import MatchScore
from engine.prompt_template import extract_fenced_json, load_prompt_sections
from engine.rubric import Rubric

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "04_brief.md"

_PROSE_FIELDS = ("projects", "extra_curriculars", "education", "certifications")
_GAP_PRIORITY = {
    "absent_required": 0, "absent_preferred": 1, "adjacent_substitution": 2,
    "unverifiable": 3, "empty_field": 4, "logistics": 5,
}


@dataclass(frozen=True)
class StrengthItem:
    source: str
    via: str
    field: str
    phrase: str
    score: float


@dataclass(frozen=True)
class Gap:
    kind: str
    detail: str


@dataclass(frozen=True)
class CandidateBriefInput:
    candidate_id: str
    headline: str
    total: float
    strengths: tuple[StrengthItem, ...]
    gaps: tuple[Gap, ...]
    experience_stated: float | None
    experience_implied: float | None
    experience_conflict: bool


@dataclass(frozen=True)
class CandidateBrief:
    candidate_id: str
    summary: str
    differentiator: str
    questions: tuple[str, ...]
    gaps_asked: tuple[Gap, ...]


def detect_strengths(score: MatchScore) -> tuple[StrengthItem, ...]:
    items = [
        StrengthItem(source=rs.source, via=rs.via, field=rs.field, phrase=rs.phrase, score=rs.score)
        for rs in score.requirement_scores
        if rs.kind == "skill" and rs.score > 0
    ]
    return tuple(sorted(items, key=lambda s: -s.score))


def detect_gaps(candidate: Candidate, rubric: Rubric, score: MatchScore) -> tuple[Gap, ...]:
    gaps: list[Gap] = []
    for req, rs in zip(rubric.requirements, score.requirement_scores):
        if req.kind != "skill":
            continue
        if not req.verifiable:
            gaps.append(Gap("unverifiable", f'"{req.source}" ({req.tier}) — no vocabulary term exists for this yet'))
        elif rs.score == 0:
            gaps.append(Gap("absent_required" if req.tier == "required" else "absent_preferred",
                             f'"{req.source}" ({req.tier}) — no evidence in the profile'))
        elif rs.via == "adjacent":
            gaps.append(Gap("adjacent_substitution",
                             f'"{req.source}" ({req.tier}) — credited via a related skill ("{rs.phrase}"), not named directly'))

    empty = [f for f in _PROSE_FIELDS if not (getattr(candidate, f) or "").strip() or getattr(candidate, f).strip() == "-"]
    if empty:
        gaps.append(Gap("empty_field", f"no {', '.join(empty)} listed on the profile"))

    if score.availability_score < 1.0:
        gaps.append(Gap("logistics", f'notice period: "{candidate.notice_period_raw}" — not immediately available'))
    if score.location_score < 1.0:
        gaps.append(Gap("logistics", f"location: {candidate.city or 'unknown'}, {candidate.country or 'unknown'} — not the role's location"))

    return tuple(sorted(gaps, key=lambda g: _GAP_PRIORITY.get(g.kind, 9)))


def select_top_gaps(gaps: tuple[Gap, ...], n: int = 3) -> tuple[Gap, ...]:
    return gaps[:n]


def build_brief_input(candidate: Candidate, rubric: Rubric, score: MatchScore) -> CandidateBriefInput:
    exp_rs = next(rs for rs in score.requirement_scores if rs.kind == "experience")
    return CandidateBriefInput(
        candidate_id=candidate.candidate_id,
        headline=candidate.headline,
        total=score.total,
        strengths=detect_strengths(score),
        gaps=detect_gaps(candidate, rubric, score),
        experience_stated=exp_rs.stated_years,
        experience_implied=exp_rs.implied_years,
        experience_conflict=bool(exp_rs.conflict),
    )


def select_finalists(scores: list[MatchScore], rubric: Rubric, fallback_n: int = 5) -> tuple[list[MatchScore], bool]:
    cleared = sorted((s for s in scores if s.total >= rubric.shortlist_threshold), key=lambda s: -s.total)
    if cleared:
        return cleared[: rubric.max_return], False
    all_sorted = sorted(scores, key=lambda s: -s.total)
    return all_sorted[: min(fallback_n, len(all_sorted))], True


def _format_strengths(strengths: tuple[StrengthItem, ...]) -> str:
    return "; ".join(f'{s.source} — {s.via}, from {s.field} ("{s.phrase}")' for s in strengths) or "none"


def _format_gaps(gaps: tuple[Gap, ...]) -> str:
    return "\n".join(f"    {i}. [{g.kind}] {g.detail}" for i, g in enumerate(gaps))


def render_brief_prompt(role: Role, rubric: Rubric, inputs: list[tuple[CandidateBriefInput, tuple[Gap, ...]]]) -> tuple[str, str, dict]:
    sections = load_prompt_sections(PROMPT_PATH)
    rubric_summary = ", ".join(r.source for r in rubric.requirements if r.kind == "skill")

    blocks = []
    for bi, top_gaps in inputs:
        conflict_note = ""
        if bi.experience_conflict:
            conflict_note = f" (stated {bi.experience_stated:g}y, dates imply {bi.experience_implied:g}y)"
        blocks.append(
            f"- {bi.candidate_id} — {bi.headline} — total {bi.total}\n"
            f"  strengths: {_format_strengths(bi.strengths)}\n"
            f"  experience: implied {bi.experience_implied:g}y{conflict_note}\n"
            f"  gaps to ask about:\n{_format_gaps(top_gaps)}"
        )

    user = sections["user"]
    user = user.replace("{role_id}", role.role_id)
    user = user.replace("{role_title}", role.title)
    user = user.replace("{rubric_summary}", rubric_summary)
    user = user.replace("{finalists_block}", "\n".join(blocks))

    schema = extract_fenced_json(sections["schema"])
    schema["properties"]["briefs"]["items"]["properties"]["candidate_id"]["enum"] = [bi.candidate_id for bi, _ in inputs]

    return sections["system"], user, schema


def generate_briefs(
    llm: LLMClient, role: Role, rubric: Rubric, brief_inputs: list[CandidateBriefInput], top_n_gaps: int = 3
) -> list[CandidateBrief]:
    inputs = [(bi, select_top_gaps(bi.gaps, top_n_gaps)) for bi in brief_inputs]
    if not inputs:
        return []
    system, user, schema = render_brief_prompt(role, rubric, inputs)
    out = llm.call(system, user, schema)

    top_gaps_by_id = {bi.candidate_id: gaps for bi, gaps in inputs}
    briefs = []
    for raw in out["briefs"]:
        cid = raw["candidate_id"]
        gaps = top_gaps_by_id.get(cid, ())
        questions = tuple(raw["questions"])[: len(gaps)]
        while len(questions) < len(gaps):
            questions = questions + (gaps[len(questions)].detail,)  # never pad with invention: fall back to the gap itself
        briefs.append(
            CandidateBrief(
                candidate_id=cid, summary=raw["summary"], differentiator=raw["differentiator"],
                questions=questions, gaps_asked=gaps,
            )
        )
    return briefs
