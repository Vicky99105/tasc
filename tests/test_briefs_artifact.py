"""Guards the checked-in data/briefs.json — the P6 output artifact."""
import json

from engine.ingest import load_candidates, load_roles
from engine.config import load_config


def _briefs():
    return json.load(open("data/briefs.json"))


class TestBriefsArtifact:
    def test_every_role_has_at_least_one_brief(self):
        briefs = _briefs()
        roles = load_roles("data/open_roles.csv")
        role_ids = {b["role_id"] for b in briefs}
        assert role_ids == {r.role_id for r in roles}

    def test_every_candidate_id_is_a_real_candidate(self):
        briefs = _briefs()
        cfg = load_config(".env")
        result = load_candidates("data/candidate_profiles.csv", cfg.reference_date)
        known = {c.candidate_id for c in result.candidates}
        for b in briefs:
            assert b["candidate_id"] in known

    def test_question_count_matches_gaps_asked_count(self):
        for b in _briefs():
            assert len(b["questions"]) == len(b["gaps_asked"])

    def test_no_candidate_has_zero_strengths_and_zero_gaps_and_a_positive_score(self):
        # sanity: a scored finalist should have SOME evidence trail, one way or the other
        for b in _briefs():
            if b["total"] > 0:
                assert b["strengths"] or b["gaps_asked"]

    def test_c076_against_r008_asks_about_docker_and_kubernetes_first(self):
        briefs = {b["candidate_id"]: b for b in _briefs() if b["role_id"] == "R008"}
        c076 = briefs.get("C076")
        assert c076 is not None
        gap_details = " ".join(g["detail"] for g in c076["gaps_asked"][:2]).lower()
        assert "docker" in gap_details and "kubernetes" in gap_details
