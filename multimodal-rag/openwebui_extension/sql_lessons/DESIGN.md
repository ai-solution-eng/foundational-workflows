# SQL-Lessons — design notes

Design of the self-improving SQL-agent lesson loop shipped under
`openwebui_extension/sql_lessons/` (first deployment: Toromont).

## Problem

LLM SQL agents fail in the same ways repeatedly: an entity referenced by a
display name gets matched with ILIKE against raw columns, a fact table gets
full-scanned for a "top-N over everything" question, a date window is missing.
The Toromont "who travelled the most" resolution fixed this by hand — a human
noticed the failure shape, wrote the rule down, and the agent stopped making
that class of mistake. This loop captures that fix automatically, for every
domain, without trusting the agent to grade itself.

## The loop

    resolve ──► distill ──► store(candidate) ──► promote ──► recall(next question)

- **resolve**: the OWUI agent answers a question (running SQL through the
  governed surface).
- **distill** (outlet): a second LLM decides whether the exchange established a
  durable lesson — positive evidence only. Trivial exchanges emit `NOTHING`.
- **store**: candidates land in a write-only staging dataset.
- **promote**: a separate gated pass (human, LLM, or auto) moves validated
  candidates into the curated set — checking the evidence shape again and
  skipping semantic near-duplicates of existing lessons.
- **recall** (inlet): the agent recalls top-k curated lessons before writing
  SQL for a similar question.

## The two-dataset guardrail

The core risk in every self-improvement loop is the system learning to trust
its own output. The guardrail here is structural, not behavioral:

| Dataset | Writes | Reads |
|---|---|---|
| `sql-lessons` (curated) | promotion gate only | agent recall |
| `sql-lessons-candidates` | distill step only | promotion gate |

The agent never reads candidates, and the distill step never writes curated.
A bad lesson can therefore only enter the prompt by passing the promotion gate
(evidence shape + near-dup + review), and can be demoted by the same gate.

## Lesson schema

One document per lesson, flat Qdrant payload:

    { "text": "<imperative instruction>",
      "kind": "schema-map | perf-guard | resolution-pattern | fail-fix",
      "trigger": "<class of question this applies to>",
      "tables": [...], "tags": [...],
      "status": "candidate | curated | stale",
      "source": "openwebui:sql-lesson" }

`kind` semantics: **schema-map** (intent term → table/column), **perf-guard**
(bounded IN-list / window / LIMIT / no free-text scans), **resolution-pattern**
(how to resolve a class of question), **fail-fix** (concrete fix for a concrete
error).

## Staleness (designed, not yet closed)

Curated lessons carry `hits`/`successes` counters. `promote.py
--demote-stale` demotes a lesson with enough samples and a success rate below
the floor (schema drift; a lesson that stopped helping). **Open item:** the
recall path is read-only, so nothing increments those counters yet — closing
the loop needs a small document-PATCH endpoint on the RAG API. Until then the
loop accumulates behind the gate but has no decay signal.

## Adapters (adding a domain)

An adapter is a directory: `adapters/<name>/<name>.jsonl` (domain lessons in
the schema above) plus `eval/<name>/`-style held-out checks. Seed with
`seed_sql_lessons.py --adapter <name>`; nothing in the loop is domain-aware.

## Eval methodology

Improvement is measured by `eval/replay.py`: replay the held-out question set
through the agent (`--executor`), score with an LLM judge or a human, and
compare PASS rates before seeding vs after each promotion batch.
