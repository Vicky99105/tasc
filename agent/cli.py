"""Minimal terminal driver for the graph. P8 builds the fuller interface; this
exists to prove P7's graph runs end to end against real data, real interrupts,
and a real checkpointer on disk.

    > list roles
    > pick R008
    > approve
    > send
    > approve
"""
from __future__ import annotations

import sqlite3
import sys

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from agent.graph import build_graph
from agent.observability import build_run_config
from agent.slack import DefaultSlackClient
from engine.config import load_config
from engine.extract import load_links
from engine.ingest import load_candidates, load_roles
from engine.llm import DefaultLLMClient
from engine.taxonomy import load_taxonomy

DB_PATH = "data/app.db"


def _print_rubric(payload: dict) -> None:
    print(f"\nRubric for {payload['role_id']} — {payload['diff_en']}")
    print(f"weights: {payload['rubric']['weights']}  threshold: {payload['rubric']['shortlist_threshold']}")
    for r in payload["rubric"]["requirements"]:
        print(f"  [{r['tier']:9s}] {r['source']}")
    print(f"est. cost so far: ${payload['est_cost']:.4f}")
    print("⏸  approve / amend <text> / ask <text> / reject")


def _print_summary(payload: dict) -> None:
    print(f"\n{payload['markdown']}")
    print("⏸  approve / edit <note> / change_rubric <text> / ask <text> / reject")


def _resume_from_input(prompt: str, gate: str) -> dict:
    raw = input(prompt).strip()
    if raw in ("approve", "send"):
        return {"decision": "approve"}
    if raw == "reject":
        return {"decision": "reject"}
    if raw.startswith("amend:") or raw.startswith("amend "):
        return {"decision": "amend", "guidance": raw.split(":", 1)[-1].strip() if ":" in raw else raw[6:].strip()}
    if raw.startswith("edit:") or raw.startswith("edit "):
        return {"decision": "edit", "note": raw.split(":", 1)[-1].strip() if ":" in raw else raw[5:].strip()}
    if raw.startswith("change_rubric:") or raw.startswith("change_rubric "):
        text = raw.split(":", 1)[-1].strip() if ":" in raw else raw[len("change_rubric"):].strip()
        return {"decision": "change_rubric", "guidance": text}
    if raw.startswith("ask:") or raw.startswith("ask "):
        return {"decision": "ask", "question": raw.split(":", 1)[-1].strip() if ":" in raw else raw[4:].strip()}
    print(f"unrecognised input {raw!r} at gate {gate}, treating as ask")
    return {"decision": "ask", "question": raw}


def main() -> None:
    cfg = load_config(".env")
    result = load_candidates("data/candidate_profiles.csv", cfg.reference_date)
    roles = load_roles("data/open_roles.csv")
    taxonomy = load_taxonomy("data/taxonomy.json")
    links = {r.candidate_id: r for r in load_links("data/links.json")}

    llm = DefaultLLMClient(cfg.openrouter_api_key, cfg.openrouter_base_url, cfg.model_link)
    if not cfg.slack_webhook_url:
        print("SLACK_WEBHOOK_URL not set — approving the summary will fail at send_to_slack.")
    slack = DefaultSlackClient(cfg.slack_webhook_url or "")

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = build_graph(llm, taxonomy, result.candidates, links, slack, checkpointer)

    print("Roles:")
    for r in roles:
        print(f"  {r.role_id}  {r.title}  ({r.department}, {r.location})")

    role_id = input("\npick a role id: ").strip()
    thread_id = f"cli-{role_id}"
    cfg_run = {
        "configurable": {"thread_id": thread_id},
        **build_run_config(cfg, thread_id, role_id=role_id, taxonomy_version=taxonomy.version),
    }

    state = graph.invoke({"role_id": role_id}, config=cfg_run)
    while "__interrupt__" in state:
        payload = state["__interrupt__"][0].value
        gate = payload["gate"]
        if gate == "approve_rubric":
            _print_rubric(payload)
        elif gate == "approve_summary":
            _print_summary(payload)
        resume = _resume_from_input(f"[{gate}] > ", gate)
        state = graph.invoke(Command(resume=resume), config=cfg_run)
        if state.get("answer"):
            print(f"\n{state['answer']}")

    if state.get("slack_ts"):
        print(f"\nPosted to Slack. (webhook response: {state['slack_ts']!r})")
    else:
        print("\nSession ended without sending.")

    conn.close()


if __name__ == "__main__":
    main()
