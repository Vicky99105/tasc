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
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

```bash
cd observability && docker compose up -d
```

Open <http://localhost:3000>, create an account (it is your own instance, the data never leaves the
machine), make a project, and paste its two keys into `.env`. Everything runs without this — traces are
for understanding the run, not for producing it.

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

## Repo

```
data/           the two CSVs
                taxonomy.json — the artifact worth reading first, reviewed in PRs
                taxonomy.db   — built from it, queried by the engine, never hand-edited
engine/         ingest, taxonomy, extract, rubric, match, brief, render
agent/          the LangGraph graph, its state, the five tools, Slack delivery
prompts/        the four prompts, as Markdown, versioned
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
