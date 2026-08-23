from eval.linking import (
    GoldenLinkCase,
    build_golden_set,
    check_regression,
    consistency_across_duplicates,
    evaluate,
)
from engine.extract import LinkResult, Mention
from engine.ingest import Candidate
from engine.taxonomy import Taxonomy, Term


def _tax() -> Taxonomy:
    terms = (
        Term(id="DOCKER", pillar="skill", label="Docker", surfaces=("Docker",),
             related=(("wrote a Dockerfile", 0.7),)),
        Term(id="SQL", pillar="skill", label="SQL", surfaces=("SQL",)),
    )
    return Taxonomy(version="t", terms=terms, edges=())


def _candidate(cid="C001", skills="Docker", projects="-") -> Candidate:
    return Candidate(
        candidate_id=cid, headline="h", skills_raw=skills,
        experience_years_stated="1", experience_years_stated_numeric=1.0,
        past_roles=(), certifications="-", education="-", projects=projects,
        extra_curriculars="-", city="X", country="Y",
        notice_period_days=0, notice_period_raw="Immediate",
    )


class TestBuildGoldenSet:
    def test_known_skill_string_gets_expected_term(self):
        cases = build_golden_set([_candidate(skills="Docker")], _tax())
        skills_case = next(c for c in cases if c.field == "skills")
        assert skills_case.expected_term == "DOCKER"

    def test_unknown_skill_string_expects_none(self):
        cases = build_golden_set([_candidate(skills="Rust")], _tax())
        skills_case = next(c for c in cases if c.field == "skills")
        assert skills_case.expected_term is None

    def test_related_phrase_in_prose_field_detected(self):
        cases = build_golden_set([_candidate(skills="-", projects="I wrote a Dockerfile for CI")], _tax())
        prose_cases = [c for c in cases if c.field == "projects"]
        assert len(prose_cases) == 1
        assert prose_cases[0].expected_term == "DOCKER"

    def test_related_phrase_as_a_list_item_expects_that_term_not_none(self):
        # "wrote a Dockerfile" sitting verbatim in the comma-split skills list,
        # not just in prose - must still expect DOCKER via the related phrase.
        cases = build_golden_set([_candidate(skills="wrote a Dockerfile")], _tax())
        skills_case = next(c for c in cases if c.field == "skills")
        assert skills_case.expected_term == "DOCKER"

    def test_no_duplicate_cases_for_same_candidate_field_text(self):
        cases = build_golden_set([_candidate(skills="Docker, Docker")], _tax())
        skills_cases = [c for c in cases if c.field == "skills"]
        assert len(skills_cases) == 1


class TestEvaluate:
    def test_true_positive(self):
        golden = [GoldenLinkCase("C001", "Docker", "skills", "DOCKER")]
        links = {"C001": LinkResult("C001", (Mention("C001", "DOCKER", "skills", "Docker", "met"),), ())}
        m = evaluate(golden, links)
        assert m.true_positive == 1 and m.precision == 1.0 and m.recall == 1.0

    def test_false_negative_when_expected_link_missing(self):
        golden = [GoldenLinkCase("C001", "Docker", "skills", "DOCKER")]
        links = {"C001": LinkResult("C001", (), ())}
        m = evaluate(golden, links)
        assert m.false_negative == 1 and m.recall == 0.0

    def test_false_positive_when_wrong_term_linked(self):
        golden = [GoldenLinkCase("C001", "Docker", "skills", "DOCKER")]
        links = {"C001": LinkResult("C001", (Mention("C001", "SQL", "skills", "Docker", "met"),), ())}
        m = evaluate(golden, links)
        assert m.false_positive == 1

    def test_true_negative_when_expected_none_and_none_found(self):
        golden = [GoldenLinkCase("C001", "Rust", "skills", None)]
        links = {"C001": LinkResult("C001", (), ())}
        m = evaluate(golden, links)
        assert m.true_negative == 1

    def test_false_positive_when_expected_none_but_something_linked(self):
        golden = [GoldenLinkCase("C001", "Rust", "skills", None)]
        links = {"C001": LinkResult("C001", (Mention("C001", "DOCKER", "skills", "Rust", "adjacent"),), ())}
        m = evaluate(golden, links)
        assert m.false_positive == 1

    def test_missing_candidate_in_links_counts_as_absent(self):
        golden = [GoldenLinkCase("CGHOST", "Docker", "skills", "DOCKER")]
        m = evaluate(golden, {})
        assert m.false_negative == 1


class TestConsistencyAcrossDuplicates:
    def test_identical_linking_is_consistent(self):
        candidates = [_candidate("C001", "Docker"), _candidate("C002", "Docker")]
        links = {
            "C001": LinkResult("C001", (Mention("C001", "DOCKER", "skills", "Docker", "met"),), ()),
            "C002": LinkResult("C002", (Mention("C002", "DOCKER", "skills", "Docker", "met"),), ()),
        }
        checked, inconsistent = consistency_across_duplicates(candidates, links)
        assert checked == 1 and inconsistent == 0

    def test_differing_linking_is_inconsistent(self):
        candidates = [_candidate("C001", "Docker"), _candidate("C002", "Docker")]
        links = {
            "C001": LinkResult("C001", (Mention("C001", "DOCKER", "skills", "Docker", "met"),), ()),
            "C002": LinkResult("C002", (), ()),
        }
        checked, inconsistent = consistency_across_duplicates(candidates, links)
        assert checked == 1 and inconsistent == 1

    def test_no_duplicates_nothing_checked(self):
        candidates = [_candidate("C001", "Docker"), _candidate("C002", "SQL")]
        links = {"C001": LinkResult("C001", (), ()), "C002": LinkResult("C002", (), ())}
        checked, inconsistent = consistency_across_duplicates(candidates, links)
        assert checked == 0


class TestCheckRegression:
    def test_no_violations_when_equal(self):
        baseline = {"precision": 0.9, "recall": 0.8, "consistency_inconsistent": 1}
        assert check_regression(dict(baseline), baseline) == []

    def test_precision_drop_flagged(self):
        baseline = {"precision": 0.9, "recall": 0.8, "consistency_inconsistent": 1}
        current = {"precision": 0.8, "recall": 0.8, "consistency_inconsistent": 1}
        violations = check_regression(current, baseline)
        assert any("precision" in v for v in violations)

    def test_recall_drop_flagged(self):
        baseline = {"precision": 0.9, "recall": 0.8, "consistency_inconsistent": 1}
        current = {"precision": 0.9, "recall": 0.7, "consistency_inconsistent": 1}
        violations = check_regression(current, baseline)
        assert any("recall" in v for v in violations)

    def test_consistency_regression_flagged(self):
        baseline = {"precision": 0.9, "recall": 0.8, "consistency_inconsistent": 1}
        current = {"precision": 0.9, "recall": 0.8, "consistency_inconsistent": 3}
        violations = check_regression(current, baseline)
        assert any("consistency" in v for v in violations)

    def test_improvement_never_flagged(self):
        baseline = {"precision": 0.9, "recall": 0.8, "consistency_inconsistent": 3}
        current = {"precision": 0.95, "recall": 0.85, "consistency_inconsistent": 0}
        assert check_regression(current, baseline) == []
