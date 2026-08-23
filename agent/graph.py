"""Nine nodes (eight work nodes plus the implicit START), three conditional
edges, three loops. Every arrow leaving an interrupt is a branch of one route_*
function reading state["decision"] and nothing else.

approve_rubric returns:
  {"decision": "approve"}
  {"decision": "amend", "guidance": "<free text>"}   -> appended to state["guidance"], never replacing it
  {"decision": "ask", "question": "<free text>"}     -> answer_question, return_to="approve_rubric"
  {"decision": "reject"}                             -> END, nothing scored, nothing sent

approve_summary returns:
  {"decision": "approve"}                            -> send_to_slack
  {"decision": "edit", "note": "<free text>"}         -> render_brief only, same candidates/scores
  {"decision": "change_rubric", "guidance": "<text>"} -> compile_rubric, whole shortlist rebuilt
  {"decision": "ask", "question": "<free text>"}      -> answer_question, return_to="approve_summary"
  {"decision": "reject"}                              -> END

answer_question always returns to state["return_to"]. It never advances the graph.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agent.slack import SlackClient
from agent.state import AgentState
from agent.tools import (
    answer_question_tool,
    check_call_budget,
    compile_rubric_tool,
    list_roles_tool,
    render_brief_tool,
    run_match_tool,
    send_to_slack_tool,
)
from engine.ingest import Candidate, Role
from engine.llm import LLMClient
from engine.rubric import Rubric
from engine.taxonomy import Taxonomy

DEFAULT_CALL_BUDGET = 40


def _rubric_summary(rubric: Rubric) -> dict:
    return {
        "weights": rubric.weights,
        "shortlist_threshold": rubric.shortlist_threshold,
        "requirements": [
            {"source": r.source, "tier": r.tier, "kind": r.kind} for r in rubric.requirements
        ],
    }


def build_graph(
    llm: LLMClient,
    taxonomy: Taxonomy,
    candidates: list[Candidate],
    links_by_id: dict,
    slack: SlackClient,
    checkpointer,
    call_budget: int = DEFAULT_CALL_BUDGET,
):
    cand_by_id = {c.candidate_id: c for c in candidates}

    def node_list_roles(state: AgentState) -> dict:
        role = state.get("role")
        if role is None:
            role = next(r for r in list_roles_tool() if r.role_id == state["role_id"])
        return {"role": role}

    def node_compile_rubric(state: AgentState) -> dict:
        check_call_budget(state.get("call_count", 0), call_budget)
        rubric, diff_en, unsupported = compile_rubric_tool(llm, state["role"], taxonomy, state.get("guidance", ""))
        est_cost = round(llm.usage.cost_usd(0.375, 1.875) + 0.01, 4)
        return {
            "rubric": rubric, "diff_en": diff_en, "est_cost": est_cost,
            "call_count": state.get("call_count", 0) + 1,
        }

    def node_approve_rubric(state: AgentState) -> dict:
        resume = interrupt({
            "gate": "approve_rubric",
            "role_id": state["role"].role_id,
            "rubric": _rubric_summary(state["rubric"]),
            "diff_en": state["diff_en"],
            "est_cost": state["est_cost"],
        })
        update: dict = {"decision": resume["decision"]}
        if resume["decision"] == "amend":
            update["guidance"] = (state.get("guidance", "") + "\n" + resume["guidance"]).strip()
        elif resume["decision"] == "ask":
            update["question"] = resume["question"]
            update["return_to"] = "approve_rubric"
        return update

    def route_after_approve_rubric(state: AgentState) -> str:
        d = state["decision"]
        return {"approve": "run_match", "amend": "compile_rubric", "ask": "answer_question"}.get(d, END)

    def node_run_match(state: AgentState) -> dict:
        finalists, excluded, fallback = run_match_tool(state["role"], state["rubric"], candidates, links_by_id, taxonomy)
        return {"results": finalists, "excluded": excluded, "fallback": fallback}

    def node_render_brief(state: AgentState) -> dict:
        check_call_budget(state.get("call_count", 0), call_budget)
        markdown, brief_inputs, briefs = render_brief_tool(
            llm, state["role"], state["rubric"], state["results"], cand_by_id, state.get("fallback", False)
        )
        return {
            "markdown": markdown, "brief_inputs": brief_inputs, "briefs": briefs,
            "call_count": state.get("call_count", 0) + 1, "edit_note": None,
        }

    def node_approve_summary(state: AgentState) -> dict:
        resume = interrupt({"gate": "approve_summary", "role_id": state["role"].role_id, "markdown": state["markdown"]})
        update: dict = {"decision": resume["decision"]}
        if resume["decision"] == "edit":
            update["edit_note"] = resume["note"]
        elif resume["decision"] == "change_rubric":
            update["guidance"] = (state.get("guidance", "") + "\n" + resume["guidance"]).strip()
        elif resume["decision"] == "ask":
            update["question"] = resume["question"]
            update["return_to"] = "approve_summary"
        return update

    def route_after_approve_summary(state: AgentState) -> str:
        d = state["decision"]
        return {
            "approve": "send_to_slack", "edit": "render_brief",
            "change_rubric": "compile_rubric", "ask": "answer_question",
        }.get(d, END)

    def node_answer_question(state: AgentState) -> dict:
        check_call_budget(state.get("call_count", 0), call_budget)
        answer = answer_question_tool(llm, state["question"], state.get("results", []), state.get("rubric"))
        return {"answer": answer, "call_count": state.get("call_count", 0) + 1}

    def route_after_answer_question(state: AgentState) -> str:
        return state["return_to"]

    def node_send_to_slack(state: AgentState) -> dict:
        ts = send_to_slack_tool(slack, state["markdown"])
        return {"slack_ts": ts}

    graph = StateGraph(AgentState)
    graph.add_node("list_roles", node_list_roles)
    graph.add_node("compile_rubric", node_compile_rubric)
    graph.add_node("approve_rubric", node_approve_rubric)
    graph.add_node("run_match", node_run_match)
    graph.add_node("render_brief", node_render_brief)
    graph.add_node("approve_summary", node_approve_summary)
    graph.add_node("answer_question", node_answer_question)
    graph.add_node("send_to_slack", node_send_to_slack)

    graph.add_edge(START, "list_roles")
    graph.add_edge("list_roles", "compile_rubric")
    graph.add_edge("compile_rubric", "approve_rubric")
    graph.add_conditional_edges("approve_rubric", route_after_approve_rubric,
                                 {"run_match": "run_match", "compile_rubric": "compile_rubric",
                                  "answer_question": "answer_question", END: END})
    graph.add_edge("run_match", "render_brief")
    graph.add_edge("render_brief", "approve_summary")
    graph.add_conditional_edges("approve_summary", route_after_approve_summary,
                                 {"send_to_slack": "send_to_slack", "render_brief": "render_brief",
                                  "compile_rubric": "compile_rubric", "answer_question": "answer_question", END: END})
    graph.add_conditional_edges("answer_question", route_after_answer_question,
                                 {"approve_rubric": "approve_rubric", "approve_summary": "approve_summary"})
    graph.add_edge("send_to_slack", END)

    return graph.compile(checkpointer=checkpointer)
