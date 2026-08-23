import os

import pytest

from engine.taxonomy import (
    Edge,
    Taxonomy,
    Term,
    adjacent_weight,
    audit_requirements,
    build_db,
    is_dag,
    load_db,
    load_taxonomy,
    narrower_closure,
    normalize,
    resolve_surface,
    save_taxonomy,
    taxonomy_from_dict,
    taxonomy_to_dict,
    validate_taxonomy,
)


class TestNormalize:
    def test_lowercases(self):
        assert normalize("Python") == "python"

    def test_transliterates_cpp(self):
        assert normalize("C++") == "c-plus-plus"

    def test_transliterates_csharp(self):
        assert normalize("C#") == "c-sharp"

    def test_transliterates_dotnet(self):
        assert normalize(".NET") == "dot-net"

    def test_c_alone_stays_distinct(self):
        assert normalize("C") == "c"
        assert normalize("C") != normalize("C++")
        assert normalize("C") != normalize("C#")

    def test_other_separators_become_single_hyphen(self):
        assert normalize("REST APIs") == "rest-apis"
        assert normalize("CI/CD") == "ci-cd"
        assert normalize("B2B  outreach") == "b2b-outreach"

    def test_no_leading_trailing_hyphens(self):
        assert normalize("  Docker  ") == "docker"


def _sample_taxonomy() -> Taxonomy:
    terms = (
        Term(id="PYTHON", pillar="skill", label="Python", surfaces=("Python",),
             from_requirements=("Python",)),
        Term(id="DJANGO", pillar="skill", label="Django", surfaces=("Django",),
             related=(("flask", 0.6),)),
        Term(id="SQL", pillar="skill", label="SQL", surfaces=("SQL", "PostgreSQL")),
        Term(id="OCC_BACKEND", pillar="occupation", label="Backend Engineer",
             surfaces=("Backend Engineer",)),
        Term(id="OCC_DEVOPS", pillar="occupation", label="DevOps Engineer",
             surfaces=("DevOps Engineer",)),
    )
    edges = (
        Edge(pillar="skill", src="DJANGO", dst="PYTHON", kind="narrower", weight=1.0),
        Edge(pillar="occupation", src="OCC_BACKEND", dst="OCC_DEVOPS", kind="adjacent", weight=0.6),
    )
    return Taxonomy(version="test-1", terms=terms, edges=edges)


class TestResolveSurface:
    def test_exact_match(self):
        tax = _sample_taxonomy()
        t = resolve_surface(tax, "skill", "python")
        assert t is not None and t.id == "PYTHON"

    def test_case_and_separator_insensitive(self):
        tax = _sample_taxonomy()
        t = resolve_surface(tax, "skill", "  Python  ")
        assert t is not None and t.id == "PYTHON"

    def test_unknown_surface_is_none(self):
        tax = _sample_taxonomy()
        assert resolve_surface(tax, "skill", "Rust") is None

    def test_wrong_pillar_is_none(self):
        tax = _sample_taxonomy()
        assert resolve_surface(tax, "occupation", "Python") is None

    def test_alias_does_not_resolve_as_exact(self):
        tax = _sample_taxonomy()
        assert resolve_surface(tax, "skill", "flask") is None


class TestNarrowerClosure:
    def test_transitive_closure(self):
        terms = (
            Term(id="A", pillar="skill", label="A", surfaces=("a",)),
            Term(id="B", pillar="skill", label="B", surfaces=("b",)),
            Term(id="C", pillar="skill", label="C", surfaces=("c",)),
        )
        edges = (
            Edge(pillar="skill", src="A", dst="B", kind="narrower", weight=1.0),
            Edge(pillar="skill", src="B", dst="C", kind="narrower", weight=1.0),
        )
        tax = Taxonomy(version="t", terms=terms, edges=edges)
        assert narrower_closure(tax, "A") == {"B", "C"}
        assert narrower_closure(tax, "B") == {"C"}
        assert narrower_closure(tax, "C") == set()


class TestIsDag:
    def test_acyclic_is_true(self):
        assert is_dag(_sample_taxonomy(), "skill") is True

    def test_cycle_is_false(self):
        terms = (
            Term(id="A", pillar="skill", label="A", surfaces=("a",)),
            Term(id="B", pillar="skill", label="B", surfaces=("b",)),
        )
        edges = (
            Edge(pillar="skill", src="A", dst="B", kind="narrower", weight=1.0),
            Edge(pillar="skill", src="B", dst="A", kind="narrower", weight=1.0),
        )
        tax = Taxonomy(version="t", terms=terms, edges=edges)
        assert is_dag(tax, "skill") is False


class TestAdjacentWeight:
    def test_symmetric_lookup(self):
        tax = _sample_taxonomy()
        assert adjacent_weight(tax, "occupation", "OCC_BACKEND", "OCC_DEVOPS") == 0.6
        assert adjacent_weight(tax, "occupation", "OCC_DEVOPS", "OCC_BACKEND") == 0.6

    def test_unknown_pair_is_none(self):
        tax = _sample_taxonomy()
        assert adjacent_weight(tax, "occupation", "OCC_BACKEND", "OCC_BACKEND") is None


class TestValidateTaxonomy:
    def test_clean_taxonomy_has_no_errors(self):
        assert validate_taxonomy(_sample_taxonomy()) == []

    def test_edge_weight_out_of_range_rejected_at_construction(self):
        with pytest.raises(ValueError):
            Edge(pillar="skill", src="A", dst="B", kind="narrower", weight=1.5)

    def test_cycle_detected(self):
        terms = (
            Term(id="A", pillar="skill", label="A", surfaces=("a",)),
            Term(id="B", pillar="skill", label="B", surfaces=("b",)),
        )
        edges = (
            Edge(pillar="skill", src="A", dst="B", kind="narrower", weight=1.0),
            Edge(pillar="skill", src="B", dst="A", kind="narrower", weight=1.0),
        )
        tax = Taxonomy(version="t", terms=terms, edges=edges)
        errs = validate_taxonomy(tax)
        assert any(e.kind == "cycle" for e in errs)

    def test_dangling_edge_detected(self):
        terms = (Term(id="A", pillar="skill", label="A", surfaces=("a",)),)
        edges = (Edge(pillar="skill", src="A", dst="GHOST", kind="narrower", weight=1.0),)
        tax = Taxonomy(version="t", terms=terms, edges=edges)
        errs = validate_taxonomy(tax)
        assert any(e.kind == "dangling_edge" for e in errs)

    def test_orphan_term_detected(self):
        terms = (Term(id="A", pillar="skill", label="A", surfaces=()),)
        tax = Taxonomy(version="t", terms=terms, edges=())
        errs = validate_taxonomy(tax)
        assert any(e.kind == "orphan_term" for e in errs)

    def test_cross_pillar_edge_is_dangling_not_silently_accepted(self):
        terms = (
            Term(id="PYTHON", pillar="skill", label="Python", surfaces=("Python",)),
            Term(id="OCC_BACKEND", pillar="occupation", label="Backend Engineer", surfaces=("Backend Engineer",)),
        )
        # a skill-pillar edge pointing at an occupation id must not resolve
        edges = (Edge(pillar="skill", src="PYTHON", dst="OCC_BACKEND", kind="narrower", weight=1.0),)
        tax = Taxonomy(version="t", terms=terms, edges=edges)
        errs = validate_taxonomy(tax)
        assert any(e.kind == "dangling_edge" for e in errs)


class TestAuditRequirements:
    def test_all_covered_returns_empty(self):
        tax = _sample_taxonomy()
        assert audit_requirements(tax, "skill", ["Python"]) == []

    def test_uncovered_string_is_reported(self):
        tax = _sample_taxonomy()
        missing = audit_requirements(tax, "skill", ["Python", "Rust"])
        assert len(missing) == 1
        assert missing[0].requirement == "Rust"


class TestJsonRoundTrip:
    def test_round_trips_exactly(self):
        tax = _sample_taxonomy()
        d = taxonomy_to_dict(tax)
        back = taxonomy_from_dict(d)
        assert back == tax

    def test_save_and_load(self, tmp_path):
        tax = _sample_taxonomy()
        path = str(tmp_path / "taxonomy.json")
        save_taxonomy(tax, path)
        back = load_taxonomy(path)
        assert back == tax


class TestSqliteIndex:
    def test_build_and_reload_preserves_terms_and_edges(self, tmp_path):
        tax = _sample_taxonomy()
        db_path = str(tmp_path / "taxonomy.db")
        build_db(tax, db_path)
        back = load_db(db_path)
        assert back.version == tax.version
        assert {t.id for t in back.terms} == {t.id for t in tax.terms}
        assert {(e.pillar, e.src, e.dst, e.kind) for e in back.edges} == \
               {(e.pillar, e.src, e.dst, e.kind) for e in tax.edges}

    def test_related_surfaces_carry_partial_weight(self, tmp_path):
        tax = _sample_taxonomy()
        db_path = str(tmp_path / "taxonomy.db")
        build_db(tax, db_path)
        back = load_db(db_path)
        django = back.term("DJANGO")
        assert ("flask", 0.6) in django.related

    def test_cross_pillar_fk_is_rejected(self, tmp_path):
        terms = (
            Term(id="PYTHON", pillar="skill", label="Python", surfaces=("Python",)),
            Term(id="OCC_BACKEND", pillar="occupation", label="Backend Engineer", surfaces=("Backend Engineer",)),
        )
        edges = (Edge(pillar="skill", src="PYTHON", dst="OCC_BACKEND", kind="narrower", weight=1.0),)
        tax = Taxonomy(version="t", terms=terms, edges=edges)
        with pytest.raises(ValueError):
            build_db(tax, str(tmp_path / "bad.db"))

    def test_invalid_taxonomy_refuses_to_build(self, tmp_path):
        terms = (Term(id="A", pillar="skill", label="A", surfaces=()),)  # orphan
        tax = Taxonomy(version="t", terms=terms, edges=())
        with pytest.raises(ValueError):
            build_db(tax, str(tmp_path / "bad.db"))

    def test_adjacent_edge_is_queryable_from_either_side(self, tmp_path):
        tax = _sample_taxonomy()
        db_path = str(tmp_path / "taxonomy.db")
        build_db(tax, db_path)
        conn = __import__("sqlite3").connect(db_path)
        rows = conn.execute(
            "SELECT src, dst FROM occupation_edge WHERE kind='adjacent'"
        ).fetchall()
        conn.close()
        pairs = {tuple(r) for r in rows}
        assert ("OCC_BACKEND", "OCC_DEVOPS") in pairs
        assert ("OCC_DEVOPS", "OCC_BACKEND") in pairs
