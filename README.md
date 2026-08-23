# Candidate–Role Match Intelligence

An agent that matches 120 candidate profiles against 10 open roles, explains every verdict with the
evidence it came from, and produces a Markdown summary a recruiter can send to a hiring manager —
after they approve it.

It does not do keyword search. Both sides normalise into one shared vocabulary first, and matching is
then a dictionary lookup that costs **zero model calls**.

---

## Cost

**Everything except the LLM is free and runs locally.**

| Component | Where it runs | Cost |
|---|---|---|
| Matching engine | your machine | free |
| Taxonomy store | SQLite, one 4 KB file in this repo | free, ships with Python |
| Langfuse (tracing) | Docker, on your machine | free, self-hosted |
| Slack delivery | an incoming webhook | free |
| LLM | OpenRouter | **~$0.15 for a full 120-candidate run** |

No managed service and no paid tier anywhere. The taxonomy lives in SQLite, which is part of the Python
standard library — no server, no install, one file you can open with any SQLite viewer. Semantic search
is not needed at 60 terms; the exact index plus SQLite's built-in FTS5 trigram search covers candidate
generation. When the vocabulary grows past a few thousand terms, `sqlite-vec` adds vector recall to the
same file, still free and still local.

---

## Setup

Five commands. Python 3.11+, and Docker only if you want traces.

```bash
git clone <this repo> && cd tasc-match
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then fill in the two required keys
python -m engine.build                                 # one-time: link all candidates, ~2 min, ~$0.15
```

Then run it:

```bash
python -m agent.cli                                    # terminal
# or
uvicorn agent.server:app --reload                      # web UI at http://localhost:8000
```

---

## Keys

Copy `.env.example` to `.env` and fill these in. **Never commit `.env`** — it is gitignored from the
first commit.

### Required

```bash
OPENROUTER_API_KEY=sk-or-v1-...        # https://openrouter.ai/keys
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=google/gemini-3.7-flash
REFERENCE_DATE=2026-08-19              # pins "Present" in a CV so scores never drift
```

`REFERENCE_DATE` matters more than it looks. Profiles say "2021–Present". Resolved against `now()`,
every candidate's tenure grows daily and yesterday's scores stop reproducing.

### Optional — Slack delivery

Only needed if you want the approved summary posted to a channel. Without it, the summary prints to the
terminal and is written to `out/`.

```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

To get one, five minutes:

1. <https://api.slack.com/apps> → **Create New App → From scratch** → pick your workspace
2. **Incoming Webhooks** → toggle **On**
3. **Add New Webhook to Workspace** → choose the channel you want posts to land in
4. Copy the URL into `.env`

The URL is bound to that one channel at creation, so there is no channel-picking code and no way to
post somewhere unintended. **It is a bearer credential** — anyone holding it can post to that channel.

### Optional — Langfuse tracing

```bash
cd observability && docker compose up -d
```

Six containers — `langfuse-web`, `langfuse-worker`, `postgres`, `clickhouse`, `redis`, `minio` — all
self-hosted, all local, data never leaves the machine. The compose file auto-provisions one org, project,
user and API key pair on first boot, so the `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` already in
`.env.example` work immediately — no manual sign-up. Open <http://localhost:3001> (`dev@example.com`,
password in `observability/docker-compose.yml`) to see traces. Port 3001, not Langfuse's usual 3000 —
something else was already holding 3000 on the machine this was built on; change it back in
`docker-compose.yml` and `.env` if that's not true for you.

LangGraph is LangChain-compatible, so one `CallbackHandler` (`agent/observability.py`) instruments the
whole graph — every node becomes a span. Model calls do NOT come for that free: `engine/llm.py` makes a
raw `urllib` call, not a LangChain Runnable, so the callback handler can't see inside a node — confirmed
live, every trace showed real node spans but $0 model cost with zero token counts. `wrap_llm()` closes
that gap by reporting one `generation` observation per call explicitly, nested under whichever node is
running. Verified live against a real Docker stack: token counts land correctly (1,213 → 1,212 for one
real `render_brief` call, checked directly in ClickHouse) — cost still shows $0, because Langfuse computes
cost from its own registered model-price table and doesn't have a price for `google/gemini-3.7-flash`,
overwriting whatever cost value the client sends. Registering that model's price in Langfuse's own Models
settings would fix the dollar figure; the token counts (and everything else — the span tree, the tags,
the session grouping) are already correct without it.

Every trace is tagged with the role id (and, from the CLI, the taxonomy version); traces from the same
chat session are grouped together in the Langfuse UI, since each interrupt resume is a separate
`graph.invoke()` call and can only be tied to the others by session, not by one continuous span.
Everything runs without this — traces are for understanding the run, not for producing it, which is why
`build_run_config()` and `wrap_llm()` are both no-ops when the three env vars aren't set.

Two Docker-specific fixes worth knowing about if you rebuild this stack from scratch: ClickHouse needs
`CLICKHOUSE_CLUSTER_ENABLED: "false"` explicitly, or its migration tries a `ReplicatedMergeTree`/`ON
CLUSTER` path that needs a Zookeeper this compose file doesn't have; and plain MinIO doesn't auto-create
buckets, so a one-shot `minio-init` service runs `mc mb` before `langfuse-web`/`langfuse-worker` are
allowed to start. Both are already handled in `observability/docker-compose.yml` — noted here because
they weren't obvious from Langfuse's own docs and cost real debugging time to find.

---

## What you will see in a trace

This is the fastest way to understand the architecture. A session over ten roles shows:

```
~1 call   per candidate      once, ever
 1 call   per role           compile the rubric
 1 call   per role           write the briefs
 0 calls  under run_match    ← 1,190 candidate-role pairs scored here
```

**That last line is the whole design.** Matching does not call the model, so adding an eleventh opening
to a database of ten thousand candidates costs two calls, not ten thousand.

---

## Using it

```
> list roles
> pick R008
    the agent compiles a rubric and shows it in plain English with its cost
    ⏸  approve / amend / reject
> amend: containers are not essential here, weight CI/CD higher
    recompiled and re-scored — re-scoring is free, so steer as much as you like
    ⏸  approve / amend / reject
> approve
    17 candidates above the threshold, each with covered requirements,
    gaps, and three questions that name a real gap
> send
    ⏸  approve / edit / reject the final summary
> approve
    posted to Slack
```

**Nothing is sent until you approve at the second gate.** That path is reachable from exactly one edge
in the graph, and that edge requires an explicit approval.

---

## How it works

```
10 roles ─┐                                   ┌─ 119 candidates
          │                                   │
   52 requirement strings              675 field values
          │                                   │
          └────────▶  ONE TAXONOMY  ◀─────────┘
                   50 skills · 10 occupations
                            │
                            ▼
                MATCH — 1,190 pairs, 0 model calls
```

- **The role side defines the vocabulary.** A term exists only because some role asked for it. A
  candidate's skill attaches to a term or is recorded as unmatched — it never creates one.
- **Scores are `relation × field_strength`.** `relation` asks how close the concept is (met 1.0,
  adjacent 0.6). `field_strength` asks how strong the claim is: shipping it on a project counts 1.00,
  passing an exam on it counts 0.75.
- **Experience is derived from the record**, never from the self-reported number, which contradicts the
  candidate's own dates on 67 of 120 profiles. Each past role contributes its own duration weighted by
  how relevant that job is to this opening.
- **The taxonomy is stored in two layers.** `data/taxonomy.json` is the source a human reviews in a pull
  request; `data/taxonomy.db` is a SQLite index built from it, and that is what the engine queries. Same
  relationship as documents to a search index — you edit the documents, you query the index.
  `python -m engine.build_taxonomy` regenerates the index; never hand-edit the `.db`.
- **The vocabulary is never pasted into a prompt.** The store resolves a mention to a handful of
  candidate terms, and only those go to the model. Prompt size stays constant whether the vocabulary
  holds 60 terms or 60,000.

---

## Assumptions

- **Silence is not evidence.** A DevOps job title does not imply Docker. This favours candidates who
  wrote more about themselves, and that is a deliberate, stated trade rather than an oversight.
- **Presence is judgeable, proficiency is not.** 118 of 120 profiles carry placeholder prose in
  `past_roles`, so there is no depth signal to model.
- **Only the current role is dated.** The second role states "N+ years", which is a floor.
- **Duplicates are removed on the full row only.** 23 groups share headline and skills but differ
  elsewhere — deduping on content would delete about 43 real people.
- **One row has no `candidate_id`** and is excluded with a reason. 119 are usable.
- **Location never excludes anyone.** The corpus contains no relocation, remote or visa signals, so
  there is nothing to filter on. It scores, and the brief says where the person is.

---

## Honest limits

- **A score gap under 3 points is a tie.** Tested: an independent reader shown two raw profiles and the
  role, never our scores, agreed with our order 90% of the time when the scores were 8+ points apart,
  and 40% when under 3 apart. Read anything closer than 3 points as a band.
- **No human has labelled anything.** Every accuracy number in this repo measures agreement between
  independent readings of the same records — self-consistency, not ground truth.
- **The threshold of 56 is derived from the score distribution**, not from anyone saying who deserved an
  interview.
- **Two vocabulary terms are unreachable** — `KAFKA` and `SAAS_SALES`. Nobody in the corpus has them, so
  they score zero for everyone and slightly depress two roles.

Full detail, including what verification caught and what it changed, is in `docs/`.

---

## Evaluating match quality at scale

Matching is subjective, so "is this a good match" splits into two different questions that need
different tools: **is the pipeline itself working correctly** (a QA problem, answerable before launch),
and **are the matches actually good** (a product problem, answerable only in production, continuously).

### Before launch: three harnesses, because the failures are at three different layers

A single end-to-end accuracy number hides where a bug actually lives. `eval/` has three separate
checks, one per layer, and **the grading rubric for each is written before any output is read** — grading
after seeing the ranking is how a golden set quietly becomes a description of what the system already
does, not an independent check on it.

| harness | what it checks | real result on this build |
|---|---|---|
| `eval/linking.py` | Does a candidate string resolve to the right taxonomy term, and — just as important — do the strings the taxonomy says *shouldn't* resolve stay unresolved? The golden set (494 cases) is built from `taxonomy.json` and the real corpus, without ever running extraction, so it can't drift toward describing the linker's own behaviour. | precision 1.0, recall 1.0, 0/22 duplicate-content groups link inconsistently |
| `eval/steering.py` | Does a natural-language rubric instruction change the *right* knob? 8 fixed utterances against the real model. This harness exists directly because a live browser test in P8 found the model narrating a change ("moved Docker and Kubernetes to preferred") that didn't match what it actually did (only moved Docker, and moved AWS/Azure unprompted) — no scripted unit test could have caught that, because scripted tests script the model's answer. | 8/8 passed, including the exact scenario that surfaced the bug |
| `eval/matching.py` | precision@5 / nDCG@10 against a hand-graded sample taken across the **whole** ranking, not just the top — a top-only sample can't see a wrongly-excluded candidate, only a wrong shortlist. 22 candidates read blind to the system's own score breakdown, graded 0–3 against the role's raw requirements. | nDCG@10 of 0.86 (R001) and 0.79 (R008); grades tracked score order almost perfectly (3→87–90, 2→69, 1→42, 0→≤10) |

Two things worth being honest about, not glossing over: the `eval/linking.py` golden set is
self-graded (built from the same taxonomy the linker uses, not an independent labeller — see
"No human has labelled anything" above), and `eval/matching.py`'s sample is scaled down from
what a production version would run (22 candidates across 2 roles here; production would want
30 per role across all 10). Both are stated limits, not hidden ones.

**Regression gate**: `eval/linking.py` also exposes `check_regression()` — a prompt or model change
must not drop precision, recall, or consistency below the checked-in baseline
(`data/eval_linking_baseline.json`). That is what turns "is the cheaper model good enough?" from an
opinion into a number that either moves or doesn't.

### At real scale, none of the above is the primary signal

Once the system is actually being used, the harnesses above only re-run as a pre-deploy gate. The
signal that actually matters in production:

- **Override rate** — how often a recruiter rejects someone the system shortlisted, or manually adds
  someone it didn't surface. It needs no labelling, arrives continuously from real usage, and measures
  the only thing that matters: does the shortlist match what a recruiter would have picked. This is the
  metric to actually optimise once the system is live.
- **Measure inter-rater agreement between recruiters first.** If two human recruiters disagree with each
  other more than the system disagrees with either of them, the ceiling on "match quality" is the task's
  inherent subjectivity, not the model — and no amount of prompt tuning fixes that. This number should be
  gathered *before* concluding the system needs to improve.
- Every accuracy number this repo reports is self-consistency between independent readings of the same
  records, stated plainly above — not ground truth. At scale, override rate against real hiring outcomes
  is the closest thing to ground truth this problem has.

---

## Repo

```
data/           the two CSVs
                taxonomy.json — the artifact worth reading first, reviewed in PRs
                taxonomy.db   — built from it, queried by the engine, never hand-edited
engine/         ingest, taxonomy, extract, rubric, steering, match, brief, render
agent/          the LangGraph graph, its state, the six tools, intent classification,
                Slack delivery, the FastAPI + SSE chat server, a minimal CLI
prompts/        the four named prompts plus intent classification, as Markdown/inline
observability/  docker-compose.yml for Langfuse
eval/           linking, matching and steering harnesses
docs/           the build plan and the verification write-ups
```

## Tests

```bash
pytest                      # unit and contract tests
python -m eval.linking      # does "Jenkins" reach CI_CD, and do the discards stay discarded
python -m eval.matching     # precision@5 and nDCG@10 per role
python -m eval.steering     # does a sentence produce the right rubric change
```
