"""The P3 gate, operationalized: does the checked-in taxonomy/links/rubric/match
stack still pass the regression baseline and both audits from the plan's "The P3
gate" section? If this fails after a re-link or a taxonomy edit, review before
building P6 on top of it — do not just re-run it until it passes.
"""
from engine.config import load_config
from engine.extract import field_text, load_links, resolve_candidate_roles
from engine.ingest import load_candidates, load_roles
from engine.match import role_occupation, score_all
from engine.rubric import compile_rubric
from engine.taxonomy import load_taxonomy


def _setup():
    cfg = load_config(".env")
    result = load_candidates("data/candidate_profiles.csv", cfg.reference_date)
    roles = load_roles("data/open_roles.csv")
    tax = load_taxonomy("data/taxonomy.json")
    links = {r.candidate_id: r for r in load_links("data/links.json")}
    cand_by_id = {c.candidate_id: c for c in result.candidates}
    rubrics = {r.role_id: compile_rubric(r, tax) for r in roles}
    scores = score_all(roles, rubrics, result.candidates, links, tax)
    return result, roles, tax, links, cand_by_id, rubrics, scores


class TestRegressionFloors:
    def test_scores_recompute_identically(self):
        result, roles, tax, links, cand_by_id, rubrics, scores1 = _setup()
        scores2 = score_all(roles, rubrics, result.candidates, links, tax)
        assert scores1 == scores2

    def test_requirement_slots_satisfied_at_least_p0_baseline(self):
        *_, scores = _setup()
        total = sum(len(s.requirement_scores) for s in scores)
        satisfied = sum(1 for s in scores for rs in s.requirement_scores if rs.score > 0)
        assert satisfied >= 682, f"only {satisfied} of {total} slots satisfied, P0 baseline was 682"

    def test_every_nonzero_requirement_score_traces_to_real_text(self):
        *_, cand_by_id, _, scores = _setup()
        for s in scores:
            c = cand_by_id[s.candidate_id]
            for rs in s.requirement_scores:
                if rs.score <= 0 or not rs.phrase or rs.kind == "experience":
                    continue
                text = field_text(c, rs.field)
                assert rs.phrase.lower() in text.lower()

    def test_shortlist_with_two_or_more_holes_share_not_above_p0_baseline(self):
        *_, rubrics, scores = _setup()
        shortlisted = [s for s in scores if s.total >= rubrics[s.role_id].shortlist_threshold]
        holes = [
            s for s in shortlisted
            if sum(1 for rs in s.requirement_scores if rs.score == 0 and rs.tier == "required") >= 2
        ]
        share = len(holes) / len(shortlisted)
        assert share <= 6 / 90, f"{share:.1%} of shortlist has 2+ holes, P0 baseline share was 6/90 = 6.7%"


class TestSameOccupationSweep:
    """Every candidate who held a role's exact occupation but was rejected must have
    a real, traceable gap (an absent required skill) — not a silent linking bug."""

    def test_every_same_occupation_reject_has_an_absent_required_skill(self):
        result, roles, tax, links, cand_by_id, rubrics, scores = _setup()
        scores_by_pair = {(s.role_id, s.candidate_id): s for s in scores}
        unexplained = []
        for role in roles:
            target = role_occupation(role, tax)
            rubric = rubrics[role.role_id]
            for c in result.candidates:
                held = any(
                    cr.occupation_id == target and cr.own_weight == 1.0
                    for cr in resolve_candidate_roles(c, tax)
                )
                if not held:
                    continue
                s = scores_by_pair[(role.role_id, c.candidate_id)]
                if s.total >= rubric.shortlist_threshold:
                    continue
                absent = [rs for rs in s.requirement_scores if rs.score == 0 and rs.tier == "required"]
                if not absent:
                    unexplained.append((role.role_id, c.candidate_id, s.total))
        assert unexplained == []


class TestDuplicateContentConsistency:
    """Candidates who share the exact same headline+skills text (real people with
    templated self-descriptions, per P1) must link identically on the skills field."""

    def test_duplicate_content_groups_link_consistently(self):
        import csv
        from collections import defaultdict

        result, roles, tax, links, cand_by_id, rubrics, scores = _setup()
        with open("data/candidate_profiles.csv") as f:
            rows = [r for r in csv.DictReader(f) if r["candidate_id"].strip()]
        groups = defaultdict(list)
        for r in rows:
            groups[(r["headline"], r["skills"])].append(r["candidate_id"])

        inconsistent = []
        for key, ids in groups.items():
            present = [i for i in ids if i in links]
            if len(present) < 2:
                continue
            term_sets = {frozenset(m.term_id for m in links[i].mentions if m.field == "skills") for i in present}
            if len(term_sets) > 1:
                inconsistent.append((key, ids))
        assert inconsistent == []
