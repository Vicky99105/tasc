import pytest

from agent.intent import classify_decision, classify_role, resume_payload
from engine.ingest import Role
from engine.llm import FakeLLMClient


def _roles() -> list[Role]:
    return [
        Role("R008", "DevOps Engineer", "Engineering", ("Docker",), (), 4, 7, "Senior", "Riyadh"),
        Role("R002", "Sales Development Representative", "Sales", ("B2B outreach",), (), 1, 3, "Junior", "Riyadh"),
    ]


class TestClassifyRole:
    def test_explicit_id_resolves(self):
        fake = FakeLLMClient([{"role_id": "R008"}])
        assert classify_role(fake, "match for R008", _roles()) == "R008"

    def test_ambiguous_message_returns_none(self):
        fake = FakeLLMClient([{"role_id": "unclear"}])
        assert classify_role(fake, "hello", _roles()) is None

    def test_role_list_only_contains_given_roles(self):
        fake = FakeLLMClient([{"role_id": "R008"}])
        classify_role(fake, "devops please", _roles())
        assert fake.calls[0]["schema"]["properties"]["role_id"]["enum"] == ["R008", "R002", "unclear"]


class TestClassifyDecision:
    def test_approve_detected(self):
        fake = FakeLLMClient([{"decision": "approve", "detail": ""}])
        out = classify_decision(fake, "looks good, approve it", ["approve", "amend", "ask", "reject"])
        assert out["decision"] == "approve"

    def test_amend_carries_guidance_verbatim(self):
        fake = FakeLLMClient([{"decision": "amend", "detail": "weight availability higher"}])
        out = classify_decision(fake, "can you weight availability higher", ["approve", "amend", "ask", "reject"])
        assert out["detail"] == "weight availability higher"

    def test_schema_restricted_to_allowed_decisions(self):
        fake = FakeLLMClient([{"decision": "approve", "detail": ""}])
        classify_decision(fake, "ok", ["approve", "reject"])
        assert fake.calls[0]["schema"]["properties"]["decision"]["enum"] == ["approve", "reject"]


class TestResumePayload:
    def test_approve(self):
        assert resume_payload({"decision": "approve", "detail": ""}) == {"decision": "approve"}

    def test_reject(self):
        assert resume_payload({"decision": "reject", "detail": "x"}) == {"decision": "reject"}

    def test_amend_maps_detail_to_guidance(self):
        assert resume_payload({"decision": "amend", "detail": "text"}) == {"decision": "amend", "guidance": "text"}

    def test_change_rubric_maps_detail_to_guidance(self):
        assert resume_payload({"decision": "change_rubric", "detail": "text"}) == {"decision": "change_rubric", "guidance": "text"}

    def test_ask_maps_detail_to_question(self):
        assert resume_payload({"decision": "ask", "detail": "why?"}) == {"decision": "ask", "question": "why?"}

    def test_edit_maps_detail_to_note(self):
        assert resume_payload({"decision": "edit", "detail": "shorter"}) == {"decision": "edit", "note": "shorter"}

    def test_unknown_decision_raises(self):
        with pytest.raises(ValueError):
            resume_payload({"decision": "bogus", "detail": ""})
