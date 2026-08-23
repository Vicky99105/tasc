"""Guards the checked-in eval artifacts and re-runs the regression gate against
the real data — if this fails after a re-link or a taxonomy edit, the P9
harnesses caught something before it reached production."""
import json

from engine.config import load_config
from engine.extract import load_links
from engine.ingest import load_candidates
from engine.taxonomy import load_taxonomy
from eval.linking import build_golden_set, check_regression, consistency_across_duplicates, evaluate


class TestLinkingBaselineArtifact:
    def test_baseline_file_is_well_formed(self):
        baseline = json.load(open("data/eval_linking_baseline.json"))
        for key in ("precision", "recall", "consistency_inconsistent", "n"):
            assert key in baseline

    def test_current_linking_does_not_regress_the_checked_in_baseline(self):
        cfg = load_config(".env")
        result = load_candidates("data/candidate_profiles.csv", cfg.reference_date)
        tax = load_taxonomy("data/taxonomy.json")
        links = {r.candidate_id: r for r in load_links("data/links.json")}

        golden = build_golden_set(result.candidates, tax)
        metrics = evaluate(golden, links)
        checked, inconsistent = consistency_across_duplicates(result.candidates, links)
        current = {**metrics.as_dict(), "consistency_inconsistent": inconsistent}

        baseline = json.load(open("data/eval_linking_baseline.json"))
        violations = check_regression(current, baseline)
        assert violations == [], violations


class TestSteeringResultArtifact:
    def test_all_recorded_cases_passed(self):
        results = json.load(open("data/eval_steering_result.json"))
        assert len(results) > 0
        failed = [r for r in results if not r["passed"]]
        assert failed == [], failed


class TestMatchingGoldenArtifact:
    def test_grades_are_in_range(self):
        grades = json.load(open("data/golden_matching.json"))
        assert len(grades) > 0
        for g in grades:
            assert 0 <= g["grade"] <= 3
