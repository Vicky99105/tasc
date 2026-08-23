import pytest

from engine.build_taxonomy import PROMPT_PATH, build_pillar, load_prompt_sections, raw_to_terms, render_prompt
from engine.llm import FakeLLMClient


class TestLoadPromptSections:
    def test_real_template_has_all_three_sections(self):
        sections = load_prompt_sections(PROMPT_PATH)
        assert set(sections) == {"system", "user", "schema"}

    def test_schema_section_contains_fenced_json(self):
        sections = load_prompt_sections(PROMPT_PATH)
        assert "```json" in sections["schema"]


class TestRenderPrompt:
    def test_unknown_pillar_raises(self):
        with pytest.raises(ValueError):
            render_prompt("bogus", [], [])

    def test_substitutes_pillar_noun_and_concept(self):
        system, user, schema = render_prompt("skill", ["Python"], ["Python"])
        assert "{PILLAR_NOUN}" not in system
        assert "skill" in system.lower()
        assert "{OUT_OF_SCOPE}" not in system

    def test_occupation_pillar_uses_occupation_language(self):
        system, _, _ = render_prompt("occupation", ["Backend Engineer"], [])
        assert "job title" in system.lower()
        assert "a job title must never imply a skill" in system.lower()

    def test_requirement_and_candidate_strings_appear_in_user_prompt(self):
        _, user, _ = render_prompt("skill", ["Python", "SQL"], ["Django"])
        assert "- Python" in user
        assert "- SQL" in user
        assert "- Django" in user
        assert "2" in user  # n_requirements

    def test_schema_is_valid_json_object(self):
        _, _, schema = render_prompt("skill", ["Python"], [])
        assert schema["type"] == "object"
        assert "terms" in schema["properties"]


class TestRawToTerms:
    def _raw(self):
        return {
            "terms": [
                {
                    "id": "SQL", "canonical": "SQL", "aliases": ["Structured Query Language"],
                    "narrower": ["PostgreSQL", "MySQL"], "adjacent": [{"term": "NOSQL", "weight": 0.6}],
                    "related": [{"phrase": "wrote complex joins", "weight": 0.5}],
                    "from_requirements": ["SQL"],
                },
                {
                    "id": "NOSQL", "canonical": "NoSQL", "aliases": [], "narrower": [],
                    "adjacent": [{"term": "SQL", "weight": 0.6}], "related": [],
                    "from_requirements": ["NoSQL"],
                },
            ],
            "unmapped": [],
        }

    def test_narrower_strings_become_surfaces_not_edges(self):
        terms, edges = raw_to_terms("skill", self._raw())
        sql = next(t for t in terms if t.id == "SQL")
        assert "PostgreSQL" in sql.surfaces
        assert "MySQL" in sql.surfaces
        assert not any(e.kind == "narrower" for e in edges)

    def test_related_becomes_partial_credit_tuple(self):
        terms, _ = raw_to_terms("skill", self._raw())
        sql = next(t for t in terms if t.id == "SQL")
        assert sql.related == (("wrote complex joins", 0.5),)

    def test_adjacent_edge_deduped_across_both_directions(self):
        _, edges = raw_to_terms("skill", self._raw())
        adjacent = [e for e in edges if e.kind == "adjacent"]
        assert len(adjacent) == 1
        assert {adjacent[0].src, adjacent[0].dst} == {"SQL", "NOSQL"}
        assert adjacent[0].weight == 0.6

    def test_dangling_adjacent_reference_is_dropped_not_kept(self):
        raw = self._raw()
        raw["terms"][0]["adjacent"] = [{"term": "GHOST_TERM", "weight": 0.6}]
        raw["terms"][1]["adjacent"] = []
        terms, edges = raw_to_terms("skill", raw)
        assert edges == []

    def test_pillar_is_tagged_on_every_term_and_edge(self):
        terms, edges = raw_to_terms("occupation", self._raw())
        assert all(t.pillar == "occupation" for t in terms)
        assert all(e.pillar == "occupation" for e in edges)


class TestBuildPillar:
    def test_calls_llm_with_rendered_prompt_and_returns_raw_response(self):
        fake = FakeLLMClient([{"terms": [], "unmapped": []}])
        out = build_pillar(fake, "skill", ["Python"], ["Python"])
        assert out == {"terms": [], "unmapped": []}
        assert len(fake.calls) == 1
        assert "- Python" in fake.calls[0]["user"]
