"""Turns a free-text chat message into a structured action. Two jobs, both
enum-constrained so the model can only pick among things that actually exist:

1. Before a role is selected, classify which of the 10 roles the recruiter means
   (or "unclear", which gatekeeps everything else — no rubric, no approval, no
   question-answering can happen without a role first).
2. At an interrupt gate, classify the message into one of that gate's valid
   decisions (approve / amend / ask / reject, or the summary gate's five), plus
   the free text that belongs with it (guidance, a question, an edit note).

This is intent classification only. It never decides what an amend DOES to the
rubric (engine/steering.py) or what an answer says (answer_question_tool) — it
only routes the raw message to the right place.
"""
from __future__ import annotations

from engine.ingest import Role
from engine.llm import LLMClient

_ROLE_SYSTEM = """You match a recruiter's chat message to one of a fixed list of open roles, by
id. If the message clearly names or describes exactly one role (by title, department, or role_id),
return its id. If it's ambiguous, names none of them, or isn't about picking a role at all, return
"unclear" — never guess."""

_DECISION_SYSTEM = """You classify a recruiter's chat message into exactly one of a fixed set of
actions for the gate they're currently at, and extract the free text that belongs with it.
- "approve" / "reject": detail is empty.
- "amend" / "change_rubric": detail is the steering guidance, verbatim or lightly cleaned up —
  never summarised away, the next step needs the actual wording.
- "ask": detail is the question, verbatim.
- "edit": detail is the wording note, verbatim.
If the message doesn't clearly match any of the allowed actions for this gate, pick "ask" and put
the whole message in detail — that routes to a grounded answer rather than a wrong guess."""


def classify_role(llm: LLMClient, message: str, roles: list[Role]) -> str | None:
    schema = {
        "type": "object", "additionalProperties": False, "required": ["role_id"],
        "properties": {"role_id": {"type": "string", "enum": [r.role_id for r in roles] + ["unclear"]}},
    }
    roles_block = "\n".join(f"- {r.role_id}: {r.title} ({r.department}, {r.location})" for r in roles)
    user = f"ROLES\n{roles_block}\n\nMESSAGE\n{message}"
    out = llm.call(_ROLE_SYSTEM, user, schema)
    role_id = out["role_id"]
    return None if role_id == "unclear" else role_id


def classify_decision(llm: LLMClient, message: str, allowed_decisions: list[str]) -> dict:
    schema = {
        "type": "object", "additionalProperties": False, "required": ["decision", "detail"],
        "properties": {
            "decision": {"type": "string", "enum": allowed_decisions},
            "detail": {"type": "string"},
        },
    }
    user = f"ALLOWED ACTIONS AT THIS GATE: {', '.join(allowed_decisions)}\n\nMESSAGE\n{message}"
    return llm.call(_DECISION_SYSTEM, user, schema)


def resume_payload(decision: dict) -> dict:
    """Maps the classifier's generic {decision, detail} to the exact resume shape
    each gate expects (agent/graph.py's node_approve_rubric / node_approve_summary)."""
    d, detail = decision["decision"], decision["detail"]
    if d in ("approve", "reject"):
        return {"decision": d}
    if d in ("amend", "change_rubric"):
        return {"decision": d, "guidance": detail}
    if d == "ask":
        return {"decision": d, "question": detail}
    if d == "edit":
        return {"decision": d, "note": detail}
    raise ValueError(f"unknown decision: {d!r}")
