import pytest

from engine.extract import (
    FIELD_GROUPS,
    CandidateRole,
    LinkResult,
    Mention,
    RejectedMention,
    extract_candidate,
    extract_group,
    field_text,
    link_all,
    load_links,
    render_extraction_prompt,
    render_vocabulary_block,
    resolve_candidate_roles,
    resolve_occupation,
    save_links,
)
from engine.ingest import Candidate, PastRole
from engine.llm import FakeLLMClient
from engine.taxonomy import Edge, Taxonomy, Term


def _tax() -> Taxonomy:
    terms = (
        Term(id="PYTHON", pillar="skill", label="Python", surfaces=("Python",)),
        Term(id="DOCKER", pillar="skill", label="Docker", surfaces=("Docker",),
             related=(("wrote a Dockerfile", 0.7),)),
        Term(id="SQL", pillar="skill", label="SQL", surfaces=("SQL", "PostgreSQL", "MySQL")),
        Term(id="BACKEND_ENGINEER", pillar="occupation", label="Backend Engineer",
             surfaces=("Backend Engineer",), related=(("Software Engineer", 0.7),)),
        Term(id="DEVOPS_ENGINEER", pillar="occupation", label="DevOps Engineer",
             surfaces=("DevOps Engineer",)),
    )
    edges = (Edge(pillar="occupation", src="BACKEND_ENGINEER", dst="DEVOPS_ENGINEER", kind="adjacent", weight=0.6),)
    return Taxonomy(version="t", terms=terms, edges=edges)


def _candidate(
    cid="C001", headline="Backend dev", skills="Python, SQL",
    projects="-", extra_curriculars="-", education="-", certifications="-",
    past_roles=(),
) -> Candidate:
    return Candidate(
        candidate_id=cid, headline=headline, skills_raw=skills,
        experience_years_stated="3", experience_years_stated_numeric=3.0,
        past_roles=past_roles, certifications=certifications, education=education,
        projects=projects, extra_curriculars=extra_curriculars,
        city="X", country="Y", notice_period_days=0, notice_period_raw="Immediate",
    )


class TestRenderVocabularyBlock:
    def test_includes_surfaces_and_related(self):
        block = render_vocabulary_block(_tax(), "skill")
        assert "PYTHON | surfaces: Python" in block
        assert "DOCKER" in block and "wrote a Dockerfile(0.7)" in block

    def test_only_requested_pillar(self):
        block = render_vocabulary_block(_tax(), "occupation")
        assert "PYTHON" not in block
        assert "BACKEND_ENGINEER" in block


class TestFieldText:
    def test_maps_field_name_to_candidate_attribute(self):
        c = _candidate(skills="Python, SQL", projects="Built X")
        assert field_text(c, "skills") == "Python, SQL"
        assert field_text(c, "projects") == "Built X"


class TestRenderExtractionPrompt:
    def test_skips_call_when_group_entirely_blank(self):
        c = _candidate(projects="-", extra_curriculars="-", education="-", certifications="-")
        assert render_extraction_prompt(c, "prose", _tax()) is None

    def test_renders_when_group_has_content(self):
        c = _candidate(skills="Python, SQL")
        rendered = render_extraction_prompt(c, "list", _tax())
        assert rendered is not None
        system, user, schema = rendered
        assert "C001" in user
        assert "skills: Python, SQL" in user
        assert "PYTHON" in schema["properties"]["mentions"]["items"]["properties"]["term"]["enum"]
        assert set(schema["properties"]["mentions"]["items"]["properties"]["field"]["enum"]) == {"headline", "skills"}

    def test_unknown_group_raises(self):
        with pytest.raises(ValueError):
            render_extraction_prompt(_candidate(), "bogus", _tax())


class TestExtractGroup:
    def test_returns_empty_when_group_skipped(self):
        fake = FakeLLMClient([])
        c = _candidate(projects="-", extra_curriculars="-", education="-", certifications="-")
        result = extract_group(fake, _tax(), c, "prose")
        assert result == ([], [])
        assert fake.usage.calls == 0

    def test_valid_mention_kept(self):
        fake = FakeLLMClient([{"mentions": [
            {"term": "PYTHON", "relation": "met", "phrase": "Python", "field": "skills"}
        ]}])
        c = _candidate(skills="Python, SQL")
        mentions, rejected = extract_group(fake, _tax(), c, "list")
        assert len(mentions) == 1
        assert mentions[0].term_id == "PYTHON"
        assert rejected == []

    def test_mention_with_ungrounded_phrase_is_dropped_and_recorded(self):
        fake = FakeLLMClient([{"mentions": [
            {"term": "PYTHON", "relation": "met", "phrase": "Rust", "field": "skills"}
        ]}])
        c = _candidate(skills="Python, SQL")
        mentions, rejected = extract_group(fake, _tax(), c, "list")
        assert mentions == []
        assert len(rejected) == 1
        assert "not found verbatim" in rejected[0].reason

    def test_mention_with_wrong_field_for_group_is_dropped_and_recorded(self):
        fake = FakeLLMClient([{"mentions": [
            {"term": "PYTHON", "relation": "met", "phrase": "Built X", "field": "projects"}
        ]}])
        c = _candidate(skills="Python, SQL", projects="Built X")
        mentions, rejected = extract_group(fake, _tax(), c, "list")
        assert mentions == []
        assert len(rejected) == 1
        assert "not in this call's group" in rejected[0].reason

    def test_call_failure_returns_none(self):
        fake = FakeLLMClient([])  # exhausted immediately -> raises inside call()
        c = _candidate(skills="Python")
        result = extract_group(fake, _tax(), c, "list")
        assert result is None


class TestExtractCandidate:
    def test_both_groups_succeed(self):
        fake = FakeLLMClient([
            {"mentions": [{"term": "PYTHON", "relation": "met", "phrase": "Python", "field": "skills"}]},
            {"mentions": [{"term": "DOCKER", "relation": "adjacent", "phrase": "wrote a Dockerfile", "field": "projects"}]},
        ])
        c = _candidate(skills="Python", projects="wrote a Dockerfile for the pipeline")
        result = extract_candidate(fake, _tax(), c)
        assert not result.partial
        assert {m.term_id for m in result.mentions} == {"PYTHON", "DOCKER"}

    def test_prose_group_skipped_when_blank_list_group_succeeds(self):
        fake = FakeLLMClient([
            {"mentions": [{"term": "PYTHON", "relation": "met", "phrase": "Python", "field": "skills"}]},
        ])
        c = _candidate(skills="Python", projects="-", extra_curriculars="-", education="-", certifications="-")
        result = extract_candidate(fake, _tax(), c)
        assert not result.partial
        assert len(result.mentions) == 1

    def test_rejected_mentions_are_aggregated_across_groups(self):
        fake = FakeLLMClient([
            {"mentions": [{"term": "PYTHON", "relation": "met", "phrase": "Rust", "field": "skills"}]},
            {"mentions": [{"term": "DOCKER", "relation": "adjacent", "phrase": "not there", "field": "projects"}]},
        ])
        c = _candidate(skills="Python", projects="wrote something")
        result = extract_candidate(fake, _tax(), c)
        assert result.mentions == ()
        assert len(result.rejected) == 2
        assert not result.partial

    def test_one_group_failing_marks_partial_but_keeps_the_other(self):
        fake = FakeLLMClient([
            {"mentions": [{"term": "PYTHON", "relation": "met", "phrase": "Python", "field": "skills"}]},
        ])  # only one scripted response; prose call (if attempted) raises
        c = _candidate(skills="Python", projects="something real", extra_curriculars="-", education="-", certifications="-")
        result = extract_candidate(fake, _tax(), c)
        # exactly one group had content->call; the other's call exhausted the fake -> failed
        assert len(result.mentions) == 1
        assert result.partial


class TestLinkAll:
    def test_runs_over_multiple_candidates(self):
        fake = FakeLLMClient([
            {"mentions": [{"term": "PYTHON", "relation": "met", "phrase": "Python", "field": "skills"}]},
            {"mentions": [{"term": "SQL", "relation": "met", "phrase": "SQL", "field": "skills"}]},
        ])
        candidates = [_candidate("C001", skills="Python"), _candidate("C002", skills="SQL")]
        results = link_all(fake, _tax(), candidates, concurrency=2)
        assert len(results) == 2
        assert {r.candidate_id for r in results} == {"C001", "C002"}


class TestLinksRoundTrip:
    def test_save_and_load_preserves_mentions_and_rejected(self, tmp_path):
        results = [
            LinkResult(
                candidate_id="C001",
                mentions=(Mention("C001", "PYTHON", "skills", "Python", "met"),),
                failed_groups=(),
                rejected=(RejectedMention("C001", "prose", {"term": "SQL", "phrase": "x"}, "not found"),),
            ),
            LinkResult(candidate_id="C002", mentions=(), failed_groups=("prose",)),
        ]
        path = str(tmp_path / "links.json")
        save_links(results, path)
        back = load_links(path)
        assert back == results


class TestResolveOccupation:
    def test_exact_surface_is_full_credit(self):
        occ_id, weight = resolve_occupation("Backend Engineer", _tax())
        assert occ_id == "BACKEND_ENGINEER"
        assert weight == 1.0

    def test_related_title_is_partial_credit(self):
        occ_id, weight = resolve_occupation("Software Engineer", _tax())
        assert occ_id == "BACKEND_ENGINEER"
        assert weight == 0.7

    def test_unknown_title_is_unmapped(self):
        occ_id, weight = resolve_occupation("Astronaut", _tax())
        assert occ_id is None
        assert weight == 0.0

    def test_case_and_separator_insensitive(self):
        occ_id, _ = resolve_occupation("  backend engineer  ", _tax())
        assert occ_id == "BACKEND_ENGINEER"


class TestResolveCandidateRoles:
    def test_resolves_each_past_role(self):
        roles = (
            PastRole(title="Backend Engineer", company="X", city="Y", is_current=True,
                      start_year=2022, duration_years=3.0, raw=""),
            PastRole(title="Software Engineer", company="Z", city="Y", is_current=False,
                      start_year=None, duration_years=2.0, raw=""),
        )
        c = _candidate(past_roles=roles)
        resolved = resolve_candidate_roles(c, _tax())
        assert resolved[0] == CandidateRole("Backend Engineer", "BACKEND_ENGINEER", 1.0, 3.0, True)
        assert resolved[1] == CandidateRole("Software Engineer", "BACKEND_ENGINEER", 0.7, 2.0, False)

    def test_no_past_roles_yields_empty(self):
        c = _candidate(past_roles=())
        assert resolve_candidate_roles(c, _tax()) == ()

    def test_unmapped_title_still_recorded_with_zero_weight(self):
        roles = (PastRole(title="Astronaut", company="X", city="Y", is_current=True,
                           start_year=2020, duration_years=1.0, raw=""),)
        c = _candidate(past_roles=roles)
        resolved = resolve_candidate_roles(c, _tax())
        assert resolved[0].occupation_id is None
        assert resolved[0].own_weight == 0.0
