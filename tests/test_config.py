import pytest

from engine.config import load_config


class TestLoadConfig:
    def test_loads_real_env_file(self):
        cfg = load_config(".env")
        assert cfg.openrouter_api_key
        assert cfg.openrouter_base_url == "https://openrouter.ai/api/v1"
        assert cfg.reference_date.isoformat() == "2026-08-19"

    def test_model_summary_falls_back_to_model_link(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text(
            "OPENROUTER_API_KEY=k\nOPENROUTER_MODEL=m1\nREFERENCE_DATE=2026-01-01\n"
        )
        cfg = load_config(str(env))
        assert cfg.model_summary == "m1"

    def test_model_summary_explicit_wins(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text(
            "OPENROUTER_API_KEY=k\nOPENROUTER_MODEL=m1\nMODEL_SUMMARY=m2\nREFERENCE_DATE=2026-01-01\n"
        )
        cfg = load_config(str(env))
        assert cfg.model_summary == "m2"

    def test_missing_required_key_raises(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("OPENROUTER_MODEL=m1\n")
        with pytest.raises(RuntimeError):
            load_config(str(env))

    def test_optional_keys_default_to_none(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text(
            "OPENROUTER_API_KEY=k\nREFERENCE_DATE=2026-01-01\n"
        )
        cfg = load_config(str(env))
        assert cfg.slack_webhook_url is None
        assert cfg.langfuse_host is None
