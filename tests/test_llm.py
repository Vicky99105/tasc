import pytest

from engine.llm import FakeLLMClient, Usage


class TestUsage:
    def test_cost_zero_when_empty(self):
        assert Usage().cost_usd(0.375, 1.875) == 0.0

    def test_cost_computed_per_million_tokens(self):
        u = Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
        assert u.cost_usd(0.375, 1.875) == pytest.approx(2.25)

    def test_cost_scales_linearly(self):
        u = Usage(prompt_tokens=500_000, completion_tokens=0)
        assert u.cost_usd(0.375, 1.875) == pytest.approx(0.1875)


class TestFakeLLMClient:
    def test_returns_scripted_responses_in_order(self):
        fake = FakeLLMClient([{"a": 1}, {"a": 2}])
        assert fake.call("sys", "u1", {}) == {"a": 1}
        assert fake.call("sys", "u2", {}) == {"a": 2}

    def test_records_every_call(self):
        fake = FakeLLMClient([{"a": 1}])
        fake.call("sys", "user text", {"type": "object"}, model="m")
        assert len(fake.calls) == 1
        assert fake.calls[0]["user"] == "user text"
        assert fake.calls[0]["model"] == "m"

    def test_tracks_call_count_in_usage(self):
        fake = FakeLLMClient([{"a": 1}, {"a": 2}])
        fake.call("s", "u", {})
        fake.call("s", "u", {})
        assert fake.usage.calls == 2

    def test_raises_when_responses_exhausted(self):
        fake = FakeLLMClient([{"a": 1}])
        fake.call("s", "u", {})
        with pytest.raises(RuntimeError):
            fake.call("s", "u", {})
