import pytest

from eval.matching import GoldenMatchGrade, evaluate, ndcg_at_k, precision_at_k, sample_across_ranking
from engine.match import MatchScore


def _score(cid, total):
    return MatchScore(role_id="R008", candidate_id=cid, total=total,
                       required_component=0, preferred_component=0,
                       availability_component=0, location_component=0,
                       requirement_scores=(), availability_score=0, location_score=0)


def _scores(n=20):
    return [_score(f"C{i:03d}", 100 - i) for i in range(n)]  # C000=100 down to C019=81


class TestSampleAcrossRanking:
    def test_includes_top_and_bottom(self):
        sample = sample_across_ranking(_scores(20), 5)
        ids = [s.candidate_id for s in sample]
        assert ids[0] == "C000"
        assert ids[-1] == "C019"

    def test_not_just_top_n(self):
        sample = sample_across_ranking(_scores(20), 5)
        ids = {s.candidate_id for s in sample}
        assert not ids.issubset({f"C{i:03d}" for i in range(5)})

    def test_n_greater_than_population_returns_all(self):
        sample = sample_across_ranking(_scores(3), 10)
        assert len(sample) == 3

    def test_deterministic(self):
        s1 = sample_across_ranking(_scores(20), 7)
        s2 = sample_across_ranking(_scores(20), 7)
        assert s1 == s2


class TestPrecisionAtK:
    def test_all_relevant_is_one(self):
        scores = _scores(10)
        grades = {s.candidate_id: 3 for s in scores[:5]}
        assert precision_at_k(scores, grades, 5) == 1.0

    def test_none_relevant_is_zero(self):
        scores = _scores(10)
        grades = {s.candidate_id: 0 for s in scores[:5]}
        assert precision_at_k(scores, grades, 5) == 0.0

    def test_partial(self):
        scores = _scores(10)
        grades = {scores[0].candidate_id: 3, scores[1].candidate_id: 0, scores[2].candidate_id: 2}
        # only 3 of top 5 graded; 2 of those 3 are relevant (>=2)
        assert precision_at_k(scores, grades, 5) == pytest.approx(2 / 3)

    def test_ungraded_top_k_returns_zero_not_error(self):
        scores = _scores(10)
        assert precision_at_k(scores, {}, 5) == 0.0


class TestNdcgAtK:
    def test_perfect_ranking_is_one(self):
        scores = _scores(5)
        grades = {s.candidate_id: 3 - i for i, s in enumerate(scores)}
        assert ndcg_at_k(scores, grades, 5) == pytest.approx(1.0)

    def test_worst_ranking_scores_below_one(self):
        scores = _scores(5)
        grades = {s.candidate_id: i for i, s in enumerate(scores)}  # inverted: worst first
        assert ndcg_at_k(scores, grades, 5) < 1.0

    def test_no_grades_returns_one_by_convention(self):
        scores = _scores(5)
        assert ndcg_at_k(scores, {}, 5) == 1.0


class TestEvaluate:
    def test_returns_both_metrics(self):
        scores = _scores(10)
        grades = [GoldenMatchGrade("R008", s.candidate_id, 3 - (i % 4)) for i, s in enumerate(scores)]
        result = evaluate(scores, grades)
        assert "precision_at_5" in result and "ndcg_at_10" in result
        assert result["n_graded"] == 10
