"""The taxonomy: two pillars (skill, occupation), each with surfaces and edges.

taxonomy.json is the reviewed source. taxonomy.db is the built index, regenerated
from the JSON by build_db() and never hand-edited. Two pillars, separate tables, so
a cross-pillar edge (a job title implying a skill) is rejected by a foreign key.

score = relation x field_strength. FIELD_STRENGTH lives here because it is part of
the taxonomy's contract with the scorer, not a scoring-time decision.
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass, field

FIELD_STRENGTH: dict[str, float] = {
    "skills": 1.00,
    "projects": 1.00,
    "past_roles": 1.00,
    "headline": 0.90,
    "certifications": 0.75,
    "education": 0.70,
    "extra_curriculars": 0.60,
}

PILLARS = ("skill", "occupation")

_TRANSLITERATE = {
    "c++": "c-plus-plus",
    "c#": "c-sharp",
    ".net": "dot-net",
}
_SEP_RE = re.compile(r"[^a-z0-9]+")
_HYPHEN_TRIM_RE = re.compile(r"^-+|-+$")


def normalize(s: str) -> str:
    """Canonical surface form: lowercase, known symbols transliterated, every
    other separator collapsed to a single hyphen. C++/C#/C all stay distinct."""
    s = s.strip().lower()
    for symbol, word in _TRANSLITERATE.items():
        s = s.replace(symbol, f" {word} ")
    s = _SEP_RE.sub("-", s)
    return _HYPHEN_TRIM_RE.sub("", s)


@dataclass(frozen=True)
class Term:
    id: str
    pillar: str
    label: str
    surfaces: tuple[str, ...] = ()
    related: tuple[tuple[str, float], ...] = ()  # surface -> partial-credit weight
    from_requirements: tuple[str, ...] = ()

    def __post_init__(self):
        if self.pillar not in PILLARS:
            raise ValueError(f"unknown pillar: {self.pillar!r}")


@dataclass(frozen=True)
class Edge:
    pillar: str
    src: str
    dst: str
    kind: str  # "narrower" | "adjacent"
    weight: float

    def __post_init__(self):
        if self.pillar not in PILLARS:
            raise ValueError(f"unknown pillar: {self.pillar!r}")
        if self.kind not in ("narrower", "adjacent"):
            raise ValueError(f"unknown edge kind: {self.kind!r}")
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"edge weight out of [0,1]: {self.weight!r}")


@dataclass(frozen=True)
class Taxonomy:
    version: str
    terms: tuple[Term, ...]
    edges: tuple[Edge, ...]

    def by_pillar(self, pillar: str) -> tuple[Term, ...]:
        return tuple(t for t in self.terms if t.pillar == pillar)

    def edges_by_pillar(self, pillar: str) -> tuple[Edge, ...]:
        return tuple(e for e in self.edges if e.pillar == pillar)

    def term(self, term_id: str) -> Term | None:
        return next((t for t in self.terms if t.id == term_id), None)


def resolve_surface(taxonomy: Taxonomy, pillar: str, surface: str) -> Term | None:
    """Exact match on a normalised surface form. No fuzzy matching — a mention
    that doesn't hit a known surface is unmatched, not guessed at."""
    key = normalize(surface)
    for t in taxonomy.by_pillar(pillar):
        if key in (normalize(s) for s in t.surfaces):
            return t
    return None


def narrower_closure(taxonomy: Taxonomy, term_id: str) -> set[str]:
    """Every term term_id is narrower-of, transitively."""
    edges = [e for e in taxonomy.edges if e.kind == "narrower"]
    by_src = {}
    for e in edges:
        by_src.setdefault(e.src, []).append(e.dst)
    seen: set[str] = set()
    frontier = [term_id]
    while frontier:
        cur = frontier.pop()
        for nxt in by_src.get(cur, ()):
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return seen


def is_dag(taxonomy: Taxonomy, pillar: str, kind: str = "narrower") -> bool:
    edges = [e for e in taxonomy.edges if e.pillar == pillar and e.kind == kind]
    by_src: dict[str, list[str]] = {}
    for e in edges:
        by_src.setdefault(e.src, []).append(e.dst)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {t.id: WHITE for t in taxonomy.by_pillar(pillar)}

    def visit(node: str) -> bool:
        color[node] = GRAY
        for nxt in by_src.get(node, ()):
            if color.get(nxt, WHITE) == GRAY:
                return False
            if color.get(nxt, WHITE) == WHITE and not visit(nxt):
                return False
        color[node] = BLACK
        return True

    return all(visit(n) for n, c in list(color.items()) if c == WHITE)


def adjacent_weight(taxonomy: Taxonomy, pillar: str, a: str, b: str) -> float | None:
    """Symmetric: an adjacent edge stored as (a, b) resolves in either direction."""
    for e in taxonomy.edges:
        if e.pillar == pillar and e.kind == "adjacent":
            if (e.src, e.dst) == (a, b) or (e.src, e.dst) == (b, a):
                return e.weight
    return None


@dataclass(frozen=True)
class ValidationError:
    kind: str
    detail: str


def validate_taxonomy(taxonomy: Taxonomy) -> list[ValidationError]:
    errors: list[ValidationError] = []
    ids_by_pillar = {p: {t.id for t in taxonomy.by_pillar(p)} for p in PILLARS}

    for pillar in PILLARS:
        if not is_dag(taxonomy, pillar, "narrower"):
            errors.append(ValidationError("cycle", f"{pillar}: narrower edges contain a cycle"))

    for e in taxonomy.edges:
        ids = ids_by_pillar[e.pillar]
        if e.src not in ids:
            errors.append(ValidationError("dangling_edge", f"{e.pillar} edge src {e.src!r} is not a known term"))
        if e.dst not in ids:
            errors.append(ValidationError("dangling_edge", f"{e.pillar} edge dst {e.dst!r} is not a known term"))
        if e.src == e.dst:
            errors.append(ValidationError("self_edge", f"{e.pillar} edge {e.src!r} -> itself"))

    referenced: dict[str, set[str]] = {p: set() for p in PILLARS}
    for e in taxonomy.edges:
        referenced[e.pillar].add(e.src)
        referenced[e.pillar].add(e.dst)
    for t in taxonomy.terms:
        if not t.surfaces and t.id not in referenced[t.pillar]:
            errors.append(ValidationError("orphan_term", f"{t.pillar} term {t.id!r} has no surfaces and no edges"))

    return errors


@dataclass(frozen=True)
class UnmatchedRequirement:
    requirement: str
    pillar: str


def audit_requirements(taxonomy: Taxonomy, pillar: str, requirement_strings: list[str]) -> list[UnmatchedRequirement]:
    """Every requirement string must have created at least one term. A string
    with nothing in from_requirements pointing to it was silently dropped."""
    covered = {req for t in taxonomy.by_pillar(pillar) for req in t.from_requirements}
    return [
        UnmatchedRequirement(requirement=req, pillar=pillar)
        for req in requirement_strings
        if req not in covered
    ]


# --- JSON round-trip -------------------------------------------------------

def taxonomy_to_dict(taxonomy: Taxonomy) -> dict:
    return {
        "version": taxonomy.version,
        "terms": [
            {
                "id": t.id,
                "pillar": t.pillar,
                "label": t.label,
                "surfaces": list(t.surfaces),
                "related": {k: v for k, v in t.related},
                "from_requirements": list(t.from_requirements),
            }
            for t in taxonomy.terms
        ],
        "edges": [asdict(e) for e in taxonomy.edges],
    }


def taxonomy_from_dict(d: dict) -> Taxonomy:
    terms = tuple(
        Term(
            id=t["id"],
            pillar=t["pillar"],
            label=t["label"],
            surfaces=tuple(t.get("surfaces", ())),
            related=tuple((t.get("related", {}) or {}).items()),
            from_requirements=tuple(t.get("from_requirements", ())),
        )
        for t in d["terms"]
    )
    edges = tuple(
        Edge(pillar=e["pillar"], src=e["src"], dst=e["dst"], kind=e["kind"], weight=e["weight"])
        for e in d["edges"]
    )
    return Taxonomy(version=d["version"], terms=terms, edges=edges)


def load_taxonomy(path: str) -> Taxonomy:
    with open(path, encoding="utf-8") as f:
        return taxonomy_from_dict(json.load(f))


def save_taxonomy(taxonomy: Taxonomy, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(taxonomy_to_dict(taxonomy), f, indent=2, ensure_ascii=False)
        f.write("\n")


# --- SQLite index ------------------------------------------------------------

_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE skill (id TEXT PRIMARY KEY, label TEXT NOT NULL);
CREATE TABLE skill_surface (surface TEXT NOT NULL, term_id TEXT NOT NULL REFERENCES skill(id), weight REAL NOT NULL DEFAULT 1.0);
CREATE TABLE skill_edge (src TEXT NOT NULL REFERENCES skill(id), dst TEXT NOT NULL REFERENCES skill(id), kind TEXT NOT NULL, weight REAL NOT NULL);

CREATE TABLE occupation (id TEXT PRIMARY KEY, label TEXT NOT NULL);
CREATE TABLE occupation_surface (surface TEXT NOT NULL, term_id TEXT NOT NULL REFERENCES occupation(id), weight REAL NOT NULL DEFAULT 1.0);
CREATE TABLE occupation_edge (src TEXT NOT NULL REFERENCES occupation(id), dst TEXT NOT NULL REFERENCES occupation(id), kind TEXT NOT NULL, weight REAL NOT NULL);

CREATE INDEX idx_skill_surface ON skill_surface(surface);
CREATE INDEX idx_occupation_surface ON occupation_surface(surface);
"""


def build_db(taxonomy: Taxonomy, db_path: str) -> None:
    errors = validate_taxonomy(taxonomy)
    if errors:
        raise ValueError(f"cannot build db from invalid taxonomy: {errors}")

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript("DROP TABLE IF EXISTS meta; DROP TABLE IF EXISTS skill_edge; "
                            "DROP TABLE IF EXISTS skill_surface; DROP TABLE IF EXISTS skill; "
                            "DROP TABLE IF EXISTS occupation_edge; DROP TABLE IF EXISTS occupation_surface; "
                            "DROP TABLE IF EXISTS occupation;")
        conn.executescript(_SCHEMA)
        conn.execute("INSERT INTO meta VALUES ('version', ?)", (taxonomy.version,))

        for pillar in PILLARS:
            table = pillar
            for t in taxonomy.by_pillar(pillar):
                conn.execute(f"INSERT INTO {table} (id, label) VALUES (?, ?)", (t.id, t.label))
                for surf in t.surfaces:
                    conn.execute(
                        f"INSERT INTO {table}_surface (surface, term_id, weight) VALUES (?, ?, ?)",
                        (normalize(surf), t.id, 1.0),
                    )
                for surf, w in t.related:
                    conn.execute(
                        f"INSERT INTO {table}_surface (surface, term_id, weight) VALUES (?, ?, ?)",
                        (normalize(surf), t.id, w),
                    )
            for e in taxonomy.edges_by_pillar(pillar):
                conn.execute(
                    f"INSERT INTO {table}_edge (src, dst, kind, weight) VALUES (?, ?, ?, ?)",
                    (e.src, e.dst, e.kind, e.weight),
                )
                if e.kind == "adjacent":
                    conn.execute(
                        f"INSERT INTO {table}_edge (src, dst, kind, weight) VALUES (?, ?, ?, ?)",
                        (e.dst, e.src, e.kind, e.weight),
                    )
        conn.commit()
    finally:
        conn.close()


def load_db(db_path: str) -> Taxonomy:
    conn = sqlite3.connect(db_path)
    try:
        version = conn.execute("SELECT value FROM meta WHERE key='version'").fetchone()[0]
        terms: list[Term] = []
        edges: list[Edge] = []
        for pillar in PILLARS:
            table = pillar
            surfaces_by_term: dict[str, list[tuple[str, float]]] = {}
            for surf, term_id, weight in conn.execute(f"SELECT surface, term_id, weight FROM {table}_surface"):
                surfaces_by_term.setdefault(term_id, []).append((surf, weight))
            for term_id, label in conn.execute(f"SELECT id, label FROM {table}"):
                own = surfaces_by_term.get(term_id, [])
                exact = tuple(s for s, w in own if w == 1.0)
                related = tuple((s, w) for s, w in own if w != 1.0)
                terms.append(Term(id=term_id, pillar=pillar, label=label, surfaces=exact, related=related))
            seen_pairs: set[tuple[str, str, str]] = set()
            for src, dst, kind, weight in conn.execute(f"SELECT src, dst, kind, weight FROM {table}_edge"):
                if kind == "adjacent":
                    canon = tuple(sorted((src, dst)))
                    key = (canon[0], canon[1], kind)
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    edges.append(Edge(pillar=pillar, src=canon[0], dst=canon[1], kind=kind, weight=weight))
                else:
                    edges.append(Edge(pillar=pillar, src=src, dst=dst, kind=kind, weight=weight))
        return Taxonomy(version=version, terms=tuple(terms), edges=tuple(edges))
    finally:
        conn.close()
