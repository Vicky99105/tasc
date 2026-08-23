"""Structured brief data -> Markdown. Pure formatting, no model call, no
randomness — the same input renders to byte-identical output every time.
"""
from __future__ import annotations

from engine.brief import CandidateBrief, CandidateBriefInput
from engine.ingest import Role
from engine.rubric import Rubric


def render_role_briefs(
    role: Role,
    rubric: Rubric,
    pairs: list[tuple[CandidateBriefInput, CandidateBrief]],
    fallback: bool,
) -> str:
    lines: list[str] = []
    lines.append(f"# {role.role_id} — {role.title}")
    lines.append("")
    if fallback:
        lines.append(
            f"_No candidate cleared the shortlist threshold of {rubric.shortlist_threshold:g}. "
            f"Showing the highest-scoring {len(pairs)} instead — consider lowering the threshold._"
        )
        lines.append("")

    for bi, brief in pairs:
        lines.append(f"## {bi.candidate_id} — {bi.total:g}")
        lines.append("")
        lines.append(f"_{bi.headline}_")
        lines.append("")
        lines.append(brief.summary)
        lines.append("")
        lines.append(f"**What sets them apart:** {brief.differentiator}")
        lines.append("")

        if bi.strengths:
            lines.append("**Strengths**")
            for s in bi.strengths:
                lines.append(f"- {s.source} — {s.via}, from {s.field} (\"{s.phrase}\")")
            lines.append("")

        conflict = ""
        if bi.experience_conflict:
            conflict = f" — profile states {bi.experience_stated:g}y, dates imply {bi.experience_implied:g}y"
        lines.append(f"**Relevant experience:** {bi.experience_implied:g}y{conflict}")
        lines.append("")

        if brief.questions:
            lines.append("**Questions**")
            for i, q in enumerate(brief.questions, 1):
                lines.append(f"{i}. {q}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
