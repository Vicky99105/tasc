"""One TypedDict holds everything. Every node reads and returns a partial update;
LangGraph merges them — no reducers, last write wins per key, which is what we
want (a re-score replaces results[], it doesn't accumulate onto the old list).
"""
from __future__ import annotations

from typing import Optional, TypedDict

from engine.brief import CandidateBrief, CandidateBriefInput
from engine.ingest import Role
from engine.match import MatchScore
from engine.rubric import Rubric


class AgentState(TypedDict, total=False):
    role_id: Optional[str]
    role: Optional[Role]

    rubric: Optional[Rubric]
    diff_en: Optional[str]
    est_cost: Optional[float]
    guidance: str  # accumulated free text, appended to across amends, never replaced

    decision: Optional[str]
    question: Optional[str]
    return_to: Optional[str]
    edit_note: Optional[str]

    results: list[MatchScore]
    excluded: list[MatchScore]
    fallback: bool

    brief_inputs: list[CandidateBriefInput]
    briefs: list[CandidateBrief]
    markdown: Optional[str]

    answer: Optional[str]
    slack_ts: Optional[str]

    call_count: int


def reset_for_new_role(state: AgentState, role_id: str, role: Role) -> AgentState:
    """Switching role resets rubric, guidance, results and markdown — stale state
    across roles is the easiest way to show a recruiter the wrong list."""
    return {
        "role_id": role_id,
        "role": role,
        "rubric": None,
        "diff_en": None,
        "est_cost": None,
        "guidance": "",
        "decision": None,
        "question": None,
        "return_to": None,
        "edit_note": None,
        "results": [],
        "excluded": [],
        "fallback": False,
        "brief_inputs": [],
        "briefs": [],
        "markdown": None,
        "answer": None,
        "slack_ts": None,
        "call_count": state.get("call_count", 0),
    }
