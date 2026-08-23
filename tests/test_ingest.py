from datetime import date

import pytest

from engine.ingest import (
    Candidate,
    DedupeGroup,
    Role,
    dedupe_full_row,
    load_candidates,
    load_roles,
    parse_experience_years_stated,
    parse_location,
    parse_notice_period,
    parse_past_roles,
    strip_html,
)

REF = date(2026, 8, 19)

ALL_NOTICE_STRINGS = [
    "", "1 month", "2 months", "2 weeks notice", "30 days notice", "45 days",
    "60 days", "90 days notice", "Available immediately", "Immediate",
    "Negotiable", "starts in 2027",
]


class TestParseNoticePeriod:
    def test_every_real_string_maps_without_raising(self):
        for raw in ALL_NOTICE_STRINGS:
            parse_notice_period(raw, REF)  # must not raise

    def test_blank_is_none(self):
        assert parse_notice_period("", REF) is None

    def test_immediate_variants_are_zero(self):
        assert parse_notice_period("Immediate", REF) == 0
        assert parse_notice_period("Available immediately", REF) == 0

    def test_negotiable_maps_to_a_number(self):
        assert parse_notice_period("Negotiable", REF) == 0

    def test_weeks_days_months(self):
        assert parse_notice_period("2 weeks notice", REF) == 14
        assert parse_notice_period("60 days", REF) == 60
        assert parse_notice_period("90 days notice", REF) == 90
        assert parse_notice_period("45 days", REF) == 45
        assert parse_notice_period("1 month", REF) == 30
        assert parse_notice_period("2 months", REF) == 60
        assert parse_notice_period("30 days notice", REF) == 30

    def test_starts_in_future_year_counts_days_from_reference(self):
        days = parse_notice_period("starts in 2027", REF)
        assert days == (date(2027, 1, 1) - REF).days

    def test_unrecognised_string_raises(self):
        with pytest.raises(ValueError):
            parse_notice_period("whenever", REF)


class TestParseExperienceYearsStated:
    def test_negative_does_not_raise(self):
        assert parse_experience_years_stated("-2") == -2.0

    def test_word_form_does_not_raise_and_is_none(self):
        assert parse_experience_years_stated("five years") is None

    def test_blank_is_none(self):
        assert parse_experience_years_stated("") is None
        assert parse_experience_years_stated(None) is None

    def test_normal_integer(self):
        assert parse_experience_years_stated("7") == 7.0


class TestParseLocation:
    def test_comma_with_space(self):
        assert parse_location("Cairo, Egypt") == ("Cairo", "Egypt")

    def test_comma_no_space(self):
        assert parse_location("Alexandria,Egypt") == ("Alexandria", "Egypt")
        assert parse_location("Riyadh,Saudi Arabia") == ("Riyadh", "Saudi Arabia")
        assert parse_location("Sharjah,UAE") == ("Sharjah", "UAE")
        assert parse_location("Beirut,Lebanon") == ("Beirut", "Lebanon")

    def test_blank(self):
        assert parse_location("") == ("", "")


class TestStripHtml:
    def test_removes_tags_and_unescapes_entities(self):
        raw = ("<b>HR Business Partner</b>, Emirates Group (Dubai)<br/>"
               "2019-Present: managed employee relations.&nbsp;")
        out = strip_html(raw)
        assert "<" not in out and ">" not in out
        assert "&nbsp;" not in out and "&amp;" not in out
        assert "HR Business Partner" in out
        assert "Emirates Group (Dubai)" in out


class TestParsePastRoles:
    def test_two_role_normal_case(self):
        raw = ("Talent Acquisition Specialist, Kitopi (Jeddah) — 2022–Present: "
               "relevant accomplishment. | Technical Recruiter, Property Finder "
               "(Jeddah) — earlier tenure, 2+ years.")
        roles = parse_past_roles(raw, REF)
        assert len(roles) == 2
        r1, r2 = roles
        assert r1.title == "Talent Acquisition Specialist"
        assert r1.company == "Kitopi"
        assert r1.city == "Jeddah"
        assert r1.is_current is True
        assert r1.start_year == 2022
        assert r1.duration_years == pytest.approx((REF - date(2022, 1, 1)).days / 365.25, abs=0.05)
        assert r2.is_current is False
        assert r2.duration_years == 2.0

    def test_html_row_yields_one_role(self):
        raw = ("<b>HR Business Partner</b>, Emirates Group (Dubai)<br/>"
               "2019-Present: managed employee relations.&nbsp;")
        roles = parse_past_roles(raw, REF)
        assert len(roles) == 1
        assert roles[0].title == "HR Business Partner"
        assert roles[0].company == "Emirates Group"
        assert roles[0].city == "Dubai"
        assert roles[0].start_year == 2019

    def test_blank_yields_no_roles(self):
        assert parse_past_roles("", REF) == ()

    def test_every_role_in_corpus_parses_without_raising(self):
        import csv
        with open("data/candidate_profiles.csv") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            parse_past_roles(r["past_roles"], REF)  # must not raise


class TestDedupeFullRow:
    def test_removes_exactly_five_from_real_corpus(self):
        import csv
        with open("data/candidate_profiles.csv") as f:
            rows = list(csv.DictReader(f))
        rows = [r for r in rows if r["candidate_id"].strip()]
        kept, groups = dedupe_full_row(rows)
        assert len(rows) - len(kept) == 5
        assert sum(len(g.dropped_ids) for g in groups) == 5

    def test_identical_content_different_id_collapses(self):
        rows = [
            {"candidate_id": "A", "headline": "x", "skills": "y"},
            {"candidate_id": "B", "headline": "x", "skills": "y"},
        ]
        kept, groups = dedupe_full_row(rows)
        assert len(kept) == 1
        assert groups == [DedupeGroup(kept_id="A", dropped_ids=("B",))]

    def test_no_duplicates_returns_all(self):
        rows = [
            {"candidate_id": "A", "headline": "x"},
            {"candidate_id": "B", "headline": "z"},
        ]
        kept, groups = dedupe_full_row(rows)
        assert len(kept) == 2
        assert groups == []


class TestLoadCandidates:
    def test_end_to_end_counts(self):
        result = load_candidates("data/candidate_profiles.csv", REF)
        assert len(result.exclusions) == 1
        assert result.exclusions[0].reason == "blank candidate_id"
        assert sum(len(g.dropped_ids) for g in result.dedupe_groups) == 5
        assert len(result.candidates) == 120 - 1 - 5
        for c in result.candidates:
            assert isinstance(c, Candidate)
            assert c.candidate_id

    def test_html_row_is_present_and_clean(self):
        result = load_candidates("data/candidate_profiles.csv", REF)
        c120 = next((c for c in result.candidates if c.candidate_id == "C120"), None)
        assert c120 is not None
        assert "<" not in c120.past_roles[0].title

    def test_blank_past_roles_candidate_has_zero_roles(self):
        result = load_candidates("data/candidate_profiles.csv", REF)
        c118 = next((c for c in result.candidates if c.candidate_id == "C118"), None)
        assert c118 is not None
        assert c118.past_roles == ()

    def test_conflict_report_names_every_contradiction(self):
        result = load_candidates("data/candidate_profiles.csv", REF)
        ids = {c.candidate_id for c in result.conflicts}
        assert len(ids) == len(result.conflicts)  # one entry per candidate, no dupes
        for conflict in result.conflicts:
            assert conflict.candidate_id
            assert conflict.stated is not None


class TestLoadRoles:
    def test_loads_all_ten(self):
        roles = load_roles("data/open_roles.csv")
        assert len(roles) == 10
        r001 = next(r for r in roles if r.role_id == "R001")
        assert r001.title == "Backend Engineer"
        assert "Python" in r001.required_skills
        assert "Kubernetes" in r001.nice_to_have_skills
        assert r001.experience_min == 3
        assert r001.experience_max == 6

    def test_every_role_is_typed(self):
        for r in load_roles("data/open_roles.csv"):
            assert isinstance(r, Role)
            assert r.experience_min <= r.experience_max
