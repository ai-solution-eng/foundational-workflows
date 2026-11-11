# Playbook memory — procedural memory across sessions

Status: **built 2026-11-10** (feature wave). The gateway half ships as an overlay because the
PCAI-LLM checkout is mounted read-only from this workspace — see "Applying the gateway half".

## The problem it solves

The session distiller already writes episodic memory ("what we did") and semantic memory
("decisions", "gotchas"). Neither is *procedural*: "## How to proceed" is the fourth section of a
project summary, with no task signature to match a new task against, no outcome label, and it
arrives wrapped in one project's context when retrieved.

A **playbook** is a procedure that is structured, outcome-labelled and independently retrievable.

## The record

    {"task_signature": "recover a CrashLoopBackOff pod whose finalizer never clears",
     "applies_when":   ["deletionTimestamp set", "kubectl delete hangs"],
     "steps":          ["check finalizers", "check webhook", "check subresource"],
     "failure_modes":  ["assumed RBAC"],
     "outcome":        "success" | "abandoned"}

Stored as an ordinary memory document with **memory_kind = "playbook"** plus payload fields:
playbook_key (sha256 of the normalized task signature), outcome, outcome_source, evidence_thread,
last_validated, success_count, fail_count, last_note.

The store supersedes a playbook only when BOTH the task signature (playbook_key) and the
procedure (a sha256 fingerprint of the normalized steps) match: that is a re-emission of the same
procedure, so it replaces the prior record and inherits its success/fail counters. A playbook with
the same task signature but different steps is a second variant, not a rewrite — both are kept,
ranked by empirical success, and recall reports the disagreement. Because the replacement is a
delete-then-write (a write-first re-emission would be skipped by near-dup suppression), the removed
record is snapshotted first and re-added if the new write fails, so a failed write can never
silently destroy a playbook and its counters. Counters merge with max(), so a replayed or retried
supersession can never inflate the evidence.

## Emission (gateway side)

The distiller gains an OPTIONAL "## Playbook" section (fenced JSON). Optional is deliberate: making
it required would raise DistillTruncatedError on every legitimately short document and re-run the
whole map-reduce.

Two things are enforced **server-side**, not left to the model:

1. **The emission gate.** A session qualifies only with at least 4 tool calls, no budget_exceeded
   marker, no denied-tool storm (3+ denied results), and no repeated identical call (3+). Below the
   bar the section is stripped and no playbook is written — even if the model emitted one.
2. **The outcome label is INFERRED and overrides the model's claim.** It is derived from trace
   signals — any denied tool result, a repeated call, a budget marker, or a tool-error ratio above
   the line — and the model's own "outcome" field is discarded, not merged. outcome_source is always
   "inferred". A model grading its own session is the failure mode this field exists to avoid.

The playbook is written as a SECOND memory document (kind "playbook") with the section stripped from
the summary body, so the four-section summary is byte-identical to what it would have been. A failed
playbook write never fails the summary write.

## Recall (mm-rag side)

**recall_playbooks(task, k=3)** — on demand, not at session start. Injecting top-k playbooks into
every preamble is the one design that contradicts progressive disclosure: most sessions need none,
and a stuffed preamble is paid on every request.

Ranking is by **smoothed success rate** (success+1)/(success+fail+2), tie-broken by recency. An
unproven playbook (0/0) always sorts last and is labelled "evidence": "unproven" regardless of how
similar it looks.

**record_playbook_outcome(memory_id, outcome)** is the write half: it increments success/fail and
stamps last_validated. Before this feature nothing could update a memory after write, which is
exactly why no counter could ever accumulate evidence.

**Disagreement is a state, not an error.** Two playbooks for the same task signature with opposite
steps are BOTH returned, with the disagreement stated explicitly and their counts shown. Do not
collapse them with a summarizer.

## Consolidation (decay, importance, contradiction)

**consolidate_memories(dry_run=True)** plans and (only on request) applies:

    importance = kind_weight * (1 + log1p(access_count)) * recency_boost
    decayed    = importance * exp(-ln2 * age_days / half_life)

Defaults: half-life 30 days, floor 0.15. A record is ARCHIVED only when all three hold: decayed below
the floor, never accessed, and kind not in PROTECTED_KINDS — **"playbook" is never auto-archived**,
because a playbook's value is its counters. Undated records are never archived on a guess.

Contradictions (near-duplicate pairs at cosine >= 0.92 whose advice points in opposite directions)
are **always report-only** — two memories disagreeing is a state to show, never one to resolve
silently.

Dry run is the default and writes nothing. Applying deletes the archive set and writes decay scores
back as a payload patch, so the next pass is explainable.

## Applying the gateway half

The PCAI-LLM checkout is read-only from this workspace, so the gateway half is an overlay:

    integrations/pcai_llm_playbook/
      playbook.py               # pure: prompt text, parse, validate, key, gate, outcome inference
      apply_playbook_patch.py   # idempotent patcher: --check | --apply [--root PATH]
      tests/                    # overlay tests + vendored upstream fixtures

From a writable gateway checkout:

    python integrations/pcai_llm_playbook/apply_playbook_patch.py --check --root /path/to/PCAI_LLM
    python integrations/pcai_llm_playbook/apply_playbook_patch.py --apply --root /path/to/PCAI_LLM

--check prints a per-edit diff, exits 0 when the patch would apply or is already applied, and exits 1
on a broken anchor. Re-running --apply after a successful apply writes nothing. One edit is an
informational skip by design: the /v1/distill inline shim has no message rows in scope, so it keeps
the pre-overlay behaviour and the gate is prompt-side at that single site.

The overlay was verified against the real gateway: 11 edits apply, the gateway's own distiller suites
pass before and after (103 tests), and a live probe shows an ineligible session writing no playbook,
an eligible-but-dirty session storing outcome="abandoned" against the model's claimed "success", and
the summary body unchanged in every case.

## What is deliberately NOT here

- **No scheduled consolidation job.** consolidate_memories is on-demand (dry run by default). A cron
  entry is a deployment decision, not a code one; if you add one, set concurrencyPolicy: Forbid so
  passes never overlap.
- **No auto-merge of contradictory memories** — by design (see above).
- **No training-loop consumption of playbooks.** outcome_source is "inferred"; do not feed these
  labels into a training loop until they are verified rather than inferred.
- **Yield will be low at first.** Most sessions are not procedures. Expect tens, not thousands, in
  the first month; judge the feature by whether the ones it produces get reused. success_count on a
  playbook that is never recalled is a signal the gate is wrong, not that memory is broken.
