from datetime import date

from agent.observability import build_run_config, is_configured, wrap_llm
from engine.config import Config
from engine.llm import FakeLLMClient


def _cfg(**kw) -> Config:
    defaults = dict(
        openrouter_api_key="k", openrouter_base_url="https://openrouter.ai/api/v1",
        model_link="m", model_summary="m", reference_date=date(2026, 1, 1),
        slack_webhook_url=None, langfuse_host=None, langfuse_public_key=None, langfuse_secret_key=None,
    )
    defaults.update(kw)
    return Config(**defaults)


class TestIsConfigured:
    def test_false_when_unset(self):
        assert is_configured(_cfg()) is False

    def test_false_when_partially_set(self):
        assert is_configured(_cfg(langfuse_host="http://localhost:3000")) is False

    def test_true_when_all_three_set(self):
        cfg = _cfg(langfuse_host="http://localhost:3000", langfuse_public_key="pk", langfuse_secret_key="sk")
        assert is_configured(cfg) is True


class TestBuildRunConfig:
    def test_empty_dict_when_not_configured(self):
        assert build_run_config(_cfg(), "session-1") == {}

    def test_never_raises_when_not_configured_even_with_role_info(self):
        # callers should be able to always call this and merge the result,
        # no branching required at the call site
        result = build_run_config(_cfg(), "s1", role_id="R008", taxonomy_version="v1", assessment_key="abc")
        assert result == {}

    def test_configured_returns_callbacks_tags_and_session_metadata(self):
        cfg = _cfg(langfuse_host="http://localhost:3000", langfuse_public_key="pk-x", langfuse_secret_key="sk-x")
        result = build_run_config(cfg, "session-42", role_id="R008", taxonomy_version="v1", assessment_key="abc123")
        assert "callbacks" in result and len(result["callbacks"]) == 1
        assert set(result["tags"]) == {"role:R008", "taxonomy:v1", "assessment:abc123"}
        assert result["metadata"] == {"langfuse_session_id": "session-42"}

    def test_configured_with_no_role_info_still_returns_callback_with_empty_tags(self):
        cfg = _cfg(langfuse_host="http://localhost:3000", langfuse_public_key="pk-x", langfuse_secret_key="sk-x")
        result = build_run_config(cfg, "session-1")
        assert result["tags"] == []
        assert result["metadata"] == {"langfuse_session_id": "session-1"}

    def test_merges_cleanly_into_a_thread_id_config(self):
        cfg = _cfg(langfuse_host="http://localhost:3000", langfuse_public_key="pk-x", langfuse_secret_key="sk-x")
        base = {"configurable": {"thread_id": "t1"}}
        merged = {**base, **build_run_config(cfg, "s1", role_id="R008")}
        assert merged["configurable"]["thread_id"] == "t1"
        assert merged["tags"] == ["role:R008"]


class TestWrapLlm:
    def test_returns_the_same_object_when_not_configured(self):
        fake = FakeLLMClient([{"a": 1}])
        assert wrap_llm(_cfg(), fake) is fake

    def test_wrapped_client_still_returns_the_real_result(self):
        # unset LANGFUSE_* env means get_client() returns Langfuse's own
        # disabled-client no-op, so this exercises the wrapping logic without
        # needing a live Langfuse server
        cfg = _cfg(langfuse_host="http://localhost:3000", langfuse_public_key="pk-x", langfuse_secret_key="sk-x")
        fake = FakeLLMClient([{"answer": "hello"}])
        traced = wrap_llm(cfg, fake)
        assert traced is not fake
        result = traced.call("sys", "user", {"type": "object"})
        assert result == {"answer": "hello"}
        assert len(fake.calls) == 1

    def test_wrapped_client_usage_delegates_to_inner(self):
        cfg = _cfg(langfuse_host="http://localhost:3000", langfuse_public_key="pk-x", langfuse_secret_key="sk-x")
        fake = FakeLLMClient([{"a": 1}])
        traced = wrap_llm(cfg, fake)
        traced.call("sys", "user", {})
        assert traced.usage.calls == 1
        assert traced.usage is fake.usage

    def test_wrapped_client_propagates_exceptions(self):
        cfg = _cfg(langfuse_host="http://localhost:3000", langfuse_public_key="pk-x", langfuse_secret_key="sk-x")
        fake = FakeLLMClient([])  # no scripted response -> raises inside call()
        traced = wrap_llm(cfg, fake)
        try:
            traced.call("sys", "user", {})
            assert False, "expected an exception"
        except RuntimeError:
            pass
