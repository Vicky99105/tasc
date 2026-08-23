import json

from engine.taxonomy import resolve_surface, validate_taxonomy
from engine.taxonomy_patches import (
    apply_occupation_patches,
    apply_skill_patches,
    build_patched_taxonomy,
)

SKILL_RAW = json.load(open("data/taxonomy_skill_raw.json"))
OCC_RAW = json.load(open("data/taxonomy_occupation_raw.json"))


class TestApplySkillPatches:
    def test_mysql_added_to_sql_surfaces(self):
        patched = apply_skill_patches(SKILL_RAW)
        sql = next(t for t in patched["terms"] if t["id"] == "SQL")
        assert "MySQL" in sql["narrower"]

    def test_mysql_removed_from_unmapped_once_resolved(self):
        patched = apply_skill_patches(SKILL_RAW)
        assert "MySQL" not in {u["string"] for u in patched["unmapped"]}

    def test_postgres_sql_adjacent_edge_removed_both_directions(self):
        patched = apply_skill_patches(SKILL_RAW)
        sql = next(t for t in patched["terms"] if t["id"] == "SQL")
        postgres = next(t for t in patched["terms"] if t["id"] == "POSTGRESQL")
        assert not any(a["term"] == "POSTGRESQL" for a in sql["adjacent"])
        assert not any(a["term"] == "SQL" for a in postgres["adjacent"])

    def test_bilingual_merged_into_arabic_language(self):
        patched = apply_skill_patches(SKILL_RAW)
        ids = [t["id"] for t in patched["terms"]]
        assert "BILINGUAL_ARABIC_ENGLISH" not in ids
        arabic = next(t for t in patched["terms"] if t["id"] == "ARABIC_LANGUAGE")
        assert "bilingual Arabic/English" in arabic["aliases"]
        assert "Arabic + English bilingual" in arabic["from_requirements"]

    def test_no_dangling_reference_to_removed_term(self):
        patched = apply_skill_patches(SKILL_RAW)
        for t in patched["terms"]:
            assert all(a["term"] != "BILINGUAL_ARABIC_ENGLISH" for a in t["adjacent"])


class TestApplyOccupationPatches:
    def test_rescued_titles_removed_from_unmapped(self):
        patched = apply_occupation_patches(OCC_RAW)
        unmapped = {u["string"] for u in patched["unmapped"]}
        assert "Sales Executive" not in unmapped
        assert "Account Executive" not in unmapped
        assert "HR Generalist" not in unmapped

    def test_full_credit_titles_become_narrower_surfaces(self):
        patched = apply_occupation_patches(OCC_RAW)
        recruiter = next(t for t in patched["terms"] if t["id"] == "TECHNICAL_RECRUITER")
        assert "Recruiter" in recruiter["narrower"]
        assert "Talent Acquisition Specialist" in recruiter["narrower"]

    def test_partial_credit_titles_become_related_not_narrower(self):
        patched = apply_occupation_patches(OCC_RAW)
        sdr = next(t for t in patched["terms"] if t["id"] == "SALES_DEVELOPMENT_REPRESENTATIVE")
        phrases = {r["phrase"]: r["weight"] for r in sdr["related"]}
        assert phrases["Sales Executive"] == 0.7
        assert "Sales Executive" not in sdr["narrower"]

    def test_idempotent_on_double_apply(self):
        once = apply_occupation_patches(OCC_RAW)
        twice = apply_occupation_patches(once)
        sdr = next(t for t in twice["terms"] if t["id"] == "SALES_DEVELOPMENT_REPRESENTATIVE")
        assert sdr["related"].count({"phrase": "Sales Executive", "weight": 0.7}) == 1


class TestBuildPatchedTaxonomy:
    def test_produces_a_valid_taxonomy(self):
        tax = build_patched_taxonomy("v-test", SKILL_RAW, OCC_RAW)
        assert validate_taxonomy(tax) == []

    def test_postgresql_resolves_full_credit_toward_sql_via_narrower_closure(self):
        from engine.taxonomy import narrower_closure

        tax = build_patched_taxonomy("v-test", SKILL_RAW, OCC_RAW)
        assert "SQL" in narrower_closure(tax, "POSTGRESQL")

    def test_mysql_resolves_directly_to_sql(self):
        tax = build_patched_taxonomy("v-test", SKILL_RAW, OCC_RAW)
        t = resolve_surface(tax, "skill", "MySQL")
        assert t is not None and t.id == "SQL"

    def test_no_surface_collision_between_sql_and_postgresql(self):
        tax = build_patched_taxonomy("v-test", SKILL_RAW, OCC_RAW)
        t = resolve_surface(tax, "skill", "PostgreSQL")
        assert t is not None and t.id == "POSTGRESQL"

    def test_both_pillars_present(self):
        tax = build_patched_taxonomy("v-test", SKILL_RAW, OCC_RAW)
        assert len(tax.by_pillar("skill")) > 0
        assert len(tax.by_pillar("occupation")) == 10
