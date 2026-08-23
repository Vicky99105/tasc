"""Loaders and normalizers for the two source CSVs.

experience_years is parsed and carried for display but never scored — it
contradicts the candidate's own dates on a large share of rows (see
ExperienceConflict). Relevance-weighted duration is derived downstream in
P5 from PastRole.duration_years, not from this field.
"""
from __future__ import annotations

import csv
import html as htmllib
import re
from dataclasses import dataclass
from datetime import date

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]+")

_ROLE1_RE = re.compile(
    r"^(?P<title>[^,]+),\s*(?P<company>.+?)\s*\((?P<city>[^)]+)\)\s*"
    r"[—-]?\s*(?P<start>\d{4})[–-]Present:?\s*(?P<blurb>.*)$"
)
_ROLE2_RE = re.compile(
    r"^(?P<title>[^,]+),\s*(?P<company>.+?)\s*\((?P<city>[^)]+)\)\s*"
    r"—\s*earlier tenure,\s*(?P<years>\d+)\+?\s*years?\.?$"
)

_NOTICE_IMMEDIATE = {"immediate", "available immediately"}
_NOTICE_NEGOTIABLE = {"negotiable"}
_NOTICE_STARTS_IN_RE = re.compile(r"starts in (\d{4})")
_NOTICE_WEEK_RE = re.compile(r"(\d+)\s*week")
_NOTICE_MONTH_RE = re.compile(r"(\d+)\s*month")
_NOTICE_DAY_RE = re.compile(r"(\d+)\s*day")


@dataclass(frozen=True)
class PastRole:
    title: str
    company: str
    city: str
    is_current: bool
    start_year: int | None
    duration_years: float
    raw: str


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    headline: str
    skills_raw: str
    experience_years_stated: str
    experience_years_stated_numeric: float | None
    past_roles: tuple[PastRole, ...]
    certifications: str
    education: str
    projects: str
    extra_curriculars: str
    city: str
    country: str
    notice_period_days: int | None
    notice_period_raw: str


@dataclass(frozen=True)
class Role:
    role_id: str
    title: str
    department: str
    required_skills: tuple[str, ...]
    nice_to_have_skills: tuple[str, ...]
    experience_min: int
    experience_max: int
    seniority: str
    location: str


@dataclass(frozen=True)
class ExclusionReason:
    row_index: int
    candidate_id: str
    reason: str


@dataclass(frozen=True)
class DedupeGroup:
    kept_id: str
    dropped_ids: tuple[str, ...]


@dataclass(frozen=True)
class ExperienceConflict:
    candidate_id: str
    stated: str
    derived_years: float


@dataclass(frozen=True)
class LoadResult:
    candidates: list[Candidate]
    exclusions: list[ExclusionReason]
    dedupe_groups: list[DedupeGroup]
    conflicts: list[ExperienceConflict]


def strip_html(text: str) -> str:
    no_tags = _TAG_RE.sub(" ", text)
    unescaped = htmllib.unescape(no_tags)
    return _WS_RE.sub(" ", unescaped).strip()


def parse_location(raw: str) -> tuple[str, str]:
    if not raw or not raw.strip():
        return ("", "")
    parts = raw.split(",")
    if len(parts) == 1:
        return (parts[0].strip(), "")
    city = parts[0].strip()
    country = ",".join(parts[1:]).strip()
    return (city, country)


def parse_experience_years_stated(raw: str | None) -> float | None:
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_notice_period(raw: str, reference_date: date) -> int | None:
    if not raw or not raw.strip():
        return None
    s = raw.strip().lower()
    if s in _NOTICE_IMMEDIATE:
        return 0
    if s in _NOTICE_NEGOTIABLE:
        return 0
    m = _NOTICE_STARTS_IN_RE.search(s)
    if m:
        target = date(int(m.group(1)), 1, 1)
        return (target - reference_date).days
    m = _NOTICE_WEEK_RE.search(s)
    if m:
        return int(m.group(1)) * 7
    m = _NOTICE_MONTH_RE.search(s)
    if m:
        return int(m.group(1)) * 30
    m = _NOTICE_DAY_RE.search(s)
    if m:
        return int(m.group(1))
    raise ValueError(f"unparseable notice_period: {raw!r}")


def _years_between(start_year: int, reference_date: date) -> float:
    days = (reference_date - date(start_year, 1, 1)).days
    return round(max(days, 0) / 365.25, 2)


def parse_past_roles(raw: str, reference_date: date) -> tuple[PastRole, ...]:
    if raw is None or not raw.strip():
        return ()
    cleaned = strip_html(raw) if "<" in raw else raw
    segments = [s.strip() for s in cleaned.split("|") if s.strip()]
    roles: list[PastRole] = []
    for seg in segments:
        m1 = _ROLE1_RE.match(seg)
        if m1:
            start_year = int(m1.group("start"))
            roles.append(
                PastRole(
                    title=m1.group("title").strip(),
                    company=m1.group("company").strip(),
                    city=m1.group("city").strip(),
                    is_current=True,
                    start_year=start_year,
                    duration_years=_years_between(start_year, reference_date),
                    raw=seg,
                )
            )
            continue
        m2 = _ROLE2_RE.match(seg)
        if m2:
            roles.append(
                PastRole(
                    title=m2.group("title").strip(),
                    company=m2.group("company").strip(),
                    city=m2.group("city").strip(),
                    is_current=False,
                    start_year=None,
                    duration_years=float(m2.group("years")),
                    raw=seg,
                )
            )
            continue
        raise ValueError(f"unparseable past_roles segment: {seg!r}")
    return tuple(roles)


def dedupe_full_row(rows: list[dict]) -> tuple[list[dict], list[DedupeGroup]]:
    seen: dict[tuple, str] = {}
    ids_by_key: dict[tuple, list[str]] = {}
    kept: list[dict] = []
    for r in rows:
        key = tuple(v for k, v in r.items() if k != "candidate_id")
        if key not in seen:
            seen[key] = r["candidate_id"]
            ids_by_key[key] = [r["candidate_id"]]
            kept.append(r)
        else:
            ids_by_key[key].append(r["candidate_id"])
    groups = [
        DedupeGroup(kept_id=ids[0], dropped_ids=tuple(ids[1:]))
        for ids in ids_by_key.values()
        if len(ids) > 1
    ]
    return kept, groups


def _build_candidate(row: dict, reference_date: date) -> tuple[Candidate, ExperienceConflict | None]:
    city, country = parse_location(row["location"])
    stated_raw = row["experience_years"]
    stated_numeric = parse_experience_years_stated(stated_raw)
    past_roles = parse_past_roles(row["past_roles"], reference_date)
    derived_total = round(sum(pr.duration_years for pr in past_roles), 2)

    candidate = Candidate(
        candidate_id=row["candidate_id"],
        headline=row["headline"],
        skills_raw=row["skills"],
        experience_years_stated=stated_raw,
        experience_years_stated_numeric=stated_numeric,
        past_roles=past_roles,
        certifications=row["certifications"],
        education=row["education"],
        projects=row["projects"],
        extra_curriculars=row["extra_curriculars"],
        city=city,
        country=country,
        notice_period_days=parse_notice_period(row["notice_period"], reference_date),
        notice_period_raw=row["notice_period"],
    )

    # >2y matches the relevant-experience conflict threshold used at score time (P4/P5):
    # dates are only known to whole-year granularity, so a smaller gap is noise, not a contradiction.
    conflict = None
    if stated_numeric is None or abs(stated_numeric - derived_total) > 2:
        conflict = ExperienceConflict(
            candidate_id=row["candidate_id"],
            stated=stated_raw,
            derived_years=derived_total,
        )
    return candidate, conflict


def load_candidates(path: str, reference_date: date) -> LoadResult:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    exclusions: list[ExclusionReason] = []
    usable_rows: list[dict] = []
    for i, row in enumerate(rows):
        if not row["candidate_id"].strip():
            exclusions.append(
                ExclusionReason(row_index=i, candidate_id="", reason="blank candidate_id")
            )
            continue
        usable_rows.append(row)

    kept_rows, dedupe_groups = dedupe_full_row(usable_rows)

    candidates: list[Candidate] = []
    conflicts: list[ExperienceConflict] = []
    for row in kept_rows:
        candidate, conflict = _build_candidate(row, reference_date)
        candidates.append(candidate)
        if conflict is not None:
            conflicts.append(conflict)

    return LoadResult(
        candidates=candidates,
        exclusions=exclusions,
        dedupe_groups=dedupe_groups,
        conflicts=conflicts,
    )


def _split_skills(raw: str) -> tuple[str, ...]:
    if not raw or not raw.strip() or raw.strip() == "-":
        return ()
    return tuple(s.strip() for s in raw.split(",") if s.strip())


def load_roles(path: str) -> list[Role]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    roles: list[Role] = []
    for row in rows:
        m = re.match(r"(\d+)\s*-\s*(\d+)\s*years?", row["experience_range"].strip())
        if not m:
            raise ValueError(f"unparseable experience_range: {row['experience_range']!r}")
        roles.append(
            Role(
                role_id=row["role_id"],
                title=row["title"],
                department=row["department"],
                required_skills=_split_skills(row["required_skills"]),
                nice_to_have_skills=_split_skills(row["nice_to_have_skills"]),
                experience_min=int(m.group(1)),
                experience_max=int(m.group(2)),
                seniority=row["seniority"],
                location=row["location"],
            )
        )
    return roles
