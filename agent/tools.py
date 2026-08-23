"""Six tools. Only three of them cost anything — list_roles, run_match and
get_assessments are free, which is what lets steering and interrogating happen
without a cost conversation.
"""
from __future__ import annotations

from dataclasses import dataclass

from agent.slack import SlackClient
from engine.brief import CandidateBriefInput, build_brief_input, generate_briefs, select_finalists
from engine.ingest import Candidate, Role, load_roles
from engine.llm import LLMClient
from engine.match import MatchScore, score_pair
from engine.render import render_role_briefs
from engine.rubric import Rubric, compile_rubric
from engine.steering import Unsupported, steer
from engine.taxonomy import Taxonomy

_ANSWER_SYSTEM = """You answer a recruiter's question about an already-computed shortlist. You may
only use the results given to you — every requirement verdict, its score, and the field its evidence
came from. Never invent a number, a skill, or a comparison that isn't in the data you were given.
Every claim must cite the evidence field it came from: "C076 has CI/CD from skills, 'Jenkins'" is a
checkable answer; "C076 looks strong on CI/CD" is not.
If the question can only be answered by re-scoring something ("what if I drop Kubernetes?"), say so
plainly and suggest the recruiter ask for that change instead of guessing the outcome."""

_ANSWER_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["answer"],
    "properties": {"answer": {"type": "string"}},
}


class CallBudgetExceeded(Exception):
    pass


def list_roles_tool(roles_path: str = "data/open_roles.csv") -> list[Role]:
    return load_roles(roles_path)


def compile_rubric_tool(
    llm: LLMClient, role: Role, taxonomy: Taxonomy, guidance_text: str, base_rubric: Rubric | None = None
) -> tuple[Rubric, str, list[Unsupported]]:
    base = base_rubric if base_rubric is not None else compile_rubric(role, taxonomy)
    result = steer(llm, role, base, guidance_text)
    return result.rubric, result.diff_en, list(result.unsupported)


def run_match_tool(
    role: Role, rubric: Rubric, candidates: list[Candidate], links_by_id: dict, taxonomy: Taxonomy
) -> tuple[list[MatchScore], list[MatchScore], bool]:
    scores = [score_pair(role, rubric, c, links_by_id[c.candidate_id], taxonomy) for c in candidates]
    finalists, fallback = select_finalists(scores, rubric)
    finalist_ids = {s.candidate_id for s in finalists}
    excluded = [s for s in scores if s.candidate_id not in finalist_ids]
    return finalists, excluded, fallback


def get_assessments_tool(results: list[MatchScore], candidate_ids: list[str] | None = None) -> list[MatchScore]:
    if candidate_ids is None:
        return list(results)
    ids = set(candidate_ids)
    return [r for r in results if r.candidate_id in ids]


def render_brief_tool(
    llm: LLMClient, role: Role, rubric: Rubric, results: list[MatchScore], cand_by_id: dict, fallback: bool
):
    brief_inputs = [build_brief_input(cand_by_id[s.candidate_id], rubric, s) for s in results]
    briefs = generate_briefs(llm, role, rubric, brief_inputs)
    briefs_by_id = {b.candidate_id: b for b in briefs}
    pairs = [(bi, briefs_by_id[bi.candidate_id]) for bi in brief_inputs if bi.candidate_id in briefs_by_id]
    markdown = render_role_briefs(role, rubric, pairs, fallback)
    return markdown, brief_inputs, briefs


def _format_results_for_answer(results: list[MatchScore]) -> str:
    if not results:
        return "(none yet — the rubric has not been scored against candidates)"
    lines = []
    for r in results:
        parts = [f"{r.candidate_id} total={r.total}"]
        for rs in r.requirement_scores:
            if rs.kind == "experience":
                parts.append(f"  experience: implied {rs.implied_years}y, band score {rs.score}")
            else:
                parts.append(f"  {rs.source} ({rs.tier}): score={rs.score} via={rs.via} field={rs.field} phrase={rs.phrase!r}")
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


def _format_rubric_for_answer(rubric: Rubric | None) -> str:
    if rubric is None:
        return "(no rubric compiled yet)"
    lines = [f"weights: {rubric.weights}", f"shortlist_threshold: {rubric.shortlist_threshold}"]
    for r in rubric.requirements:
        lines.append(f"  {r.source} ({r.tier}, {r.kind})")
    return "\n".join(lines)


def answer_question_tool(llm: LLMClient, question: str, results: list[MatchScore], rubric: Rubric | None = None) -> str:
    """Reads state and nothing else. No tool it calls recomputes anything. Works at
    either gate: results[] may be empty if asked before run_match — the rubric
    alone is still groundable context for a rubric-stage question."""
    user = (
        f"QUESTION\n{question}\n\n"
        f"RUBRIC\n{_format_rubric_for_answer(rubric)}\n\n"
        f"RESULTS\n{_format_results_for_answer(results)}"
    )
    out = llm.call(_ANSWER_SYSTEM, user, _ANSWER_SCHEMA)
    return out["answer"]


def send_to_slack_tool(slack: SlackClient, markdown: str) -> str:
    return slack.post(markdown)


def check_call_budget(call_count: int, budget: int) -> None:
    if call_count >= budget:
        raise CallBudgetExceeded(f"session call budget of {budget} reached ({call_count} calls made)")
