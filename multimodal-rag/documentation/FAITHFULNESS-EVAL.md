# Faithfulness Evaluation (Feature E)

CI-gated faithfulness evaluation for RAG answers.  The feature is **opt-in and
default-off**: it is entirely standalone tooling (this directory, one script,
one workflow, one test tree).  Nothing in the server runtime imports it, no
existing behaviour changes, and it only acts when someone runs the commands
below (or the CI workflow is present).

## Files

| Path | Purpose |
| --- | --- |
| evaluations/eval_pipelines/faithfulness.py | The deterministic lexical scorer (stdlib only) |
| evaluations/eval_pipelines/run_faithfulness_eval.py | CLI: score the golden fixtures, write a results JSON |
| evaluations/eval_pipelines/fixtures/faithfulness_golden.jsonl | 30 hand-written golden cases |
| evaluations/eval_pipelines/baseline/faithfulness_baseline.json | Committed baseline results (what the gate compares against) |
| scripts/ci_eval_gate.py | The gate: compare fresh results to the baseline, exit 0/1/2 |
| tests/evaluations/test_faithfulness_scorer.py | Scorer unit tests (imported by path, no package deps) |
| tests/evaluations/test_faithfulness_gate.py | Runner + gate integration tests (subprocesses; this is what makes it CI-gated inside the normal pytest suite) |
| .github/workflows/faithfulness-gate.yml | GitHub Actions workflow for when CI exists |

## What the scorer measures

For each case (an answer plus its source contexts) the scorer:

1. splits the answer into **claims** (regex sentence split: . ! ? followed by
   whitespace, plus newline boundaries for bullet lists);
2. tokenizes claim and contexts (lowercase word/number tokens; apostrophes
   stay inside a token, so "doesn't" is one token);
3. for each claim, compares its **content tokens** (stopwords removed)
   against each context, with light **prefix matching** for inflection
   ("support"~"supports", "cluster"~"clusters"; the shorter token must have
   at least 4 characters, and prefix matching is disabled for negation
   markers so "does" never matches "doesn't");
4. marks the claim **supported** when at least **0.7** of its content tokens
   are found in a single context, **and** no matched token carries opposite
   negation on the two sides — a **negation guard** that looks at the token
   itself plus the 2 tokens before it ("does not require" vs "requires" is a
   conflict, never support; a marker that is itself a matched token counts
   as self-negating).  Negation is detected by single-token markers (not,
   no, never, cannot, won't, the isn't/doesn't/didn't/... contractions) and
   by two-token patterns ("fails to", "failed to", "unable to", "no
   longer", "instead of", "rather than" — the last two negate their
   OBJECT, which follows the phrase), **and** there is no **numeric
   contradiction**: every
   number in the claim (digit numbers like 768, 4096, 1.19.0 — dotted
   versions stay whole — and number words like three, five, matched exactly)
   must appear in that same best context, so "4096-dimensional" vs
   "768-dimensional" vetoes a claim no matter how high the overlap is;
5. a context rejected by the negation guard or the numeric guard is not
   reported as best_context (None instead); a context that merely falls
   below the coverage bar is still reported as best_context, even when the
   claim ends up unsupported;
6. the case **score** is the fraction of supported claims (0..1); an answer
   with no extractable claim scores 1.0 vacuously (recorded with a note).
   A claim made up purely of stopwords is unmeasurable, not supported: it
   is vacuously scored and carries a per-claim "note": "no scorable
   claim; vacuously scored" (F7), so a reader can tell "supported" from
   "unmeasurable";
7. the results document also carries per-claim detail
   (claim / supported / best_context) and aggregates (mean over all cases,
   mean over the faithful set, per-category means).

The scorer is **deterministic** (same inputs, byte-identical output, no
timestamps), **offline** (no network), **LLM-free**, and **stdlib-only**.
Results documents carry "method": "lexical" and a "scorer_version" (bump the
constant when the rules change; the gate warns on a mismatch).

## What lexical grounding does NOT measure

Read this before trusting a score:

- **Not entailment.** The scorer does not understand meaning. Synonyms do
  not match ("cars" vs "automobiles"); a fully supported claim only means
  its words appear in a context.
- **Not factuality.** A claim can reach the 0.7 content-coverage bar with
  words that appear in the context while the *statement* is wrong. An
  entity swap (API to CLI) or a direction flip (scales up to scales down)
  slips through when no number changes and the overlap stays high. A 1.0
  score is "these words are grounded", never "this is true". The numeric
  contradiction guard closes the wrong-*number* case (768 vs 4096, 429 vs
  500, 1.19.0 vs 1.29.0, three vs five), which is the most damaging
  hallucination class; wrong *words* with high overlap remain a known gap.
- **The disjoint-vocabulary cases are the easy half.** A hallucination that
  shares no words with the context scores 0.0 trivially — that is what the
  old "hallucinated" fixtures prove. The harder class is context vocabulary
  reused around a wrong number or entity (the "plausible-contradiction"
  fixtures), which needs the 0.7 bar and the numeric guard to catch.
- **Negation is a heuristic.** The guard is a fixed 2-token window before
each matched token, over a closed list of single-token markers and
six two-token patterns ("fails to", "failed to", "unable to", "no
longer", "instead of", "rather than").  The actual, named gaps:
negations farther than 2 tokens away ("It is not, after all, supported
by X" — pinned as a known miss by a unit test); double negation ("not
unavailable" reads as negation-flipped); and failure forms outside the
pattern list — bare "fails" without "to", "short of", "barring",
"lacking", "refuses to" — which a lexical scorer cannot enumerate
safely.  When no number changes, none of the guards can see any of
these.
- **Sentence splitting is a regex.** Abbreviations ("e.g.", "i.e.") and
  decimals ("version 1.2.") mis-split into extra claims.
- **No completeness.** A short answer that repeats one context verbatim
  scores 1.0 even if it omits every important caveat in the other
  contexts.
- **Not multimodal.** Only text contexts are scored; image/audio content is
  out of scope for this scorer.
- **Numbers are literal.** "1,000" does not match "1000"; no unit
  conversion ("1 hour" vs "60 minutes" does not match).
- **The LLM judge is separate and never runs in CI.**
  judge_faithfulness(answer, contexts, judge_fn) exists for out-of-band
  semantic evaluation, but an LLM judge is non-deterministic, needs network
  access, and its cost does not belong in a regression gate. The gate
  rejects (exit 2) any results document whose method is not "lexical".

## Commands

Run the golden fixtures (deterministic; two runs are byte-identical):

```
python evaluations/eval_pipelines/run_faithfulness_eval.py \
    --fixtures evaluations/eval_pipelines/fixtures/faithfulness_golden.jsonl \
    --out /tmp/faithfulness_results.json
```

Gate the fresh results against the committed baseline (the exact gate
command; exit 0 = pass, 1 = gate failure, 2 = usage/input error):

```
python scripts/ci_eval_gate.py \
    --baseline evaluations/eval_pipelines/baseline/faithfulness_baseline.json \
    --results /tmp/faithfulness_results.json \
    --min-score 0.80 --max-regression 0.02
```

Run the in-suite tests (the pytest integration that gates CI today):

```
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u no_proxy -u NO_PROXY \
    .venv/bin/python -m pytest tests/evaluations/ -q
```

### Gate semantics

The gate reads two results documents (baseline, fresh) and fails (exit 1) when:

- any case's fresh score is more than --max-regression (default 0.02) below
  its baseline score — reported as a per-case "regression:" line with
  baseline / results / delta;
- the mean fresh score over the **faithful set** is below --min-score
  (default 0.80).  The faithful set is defined by the **baseline's**
  "should_be_faithful" flags, not the fresh file's: the baseline is the
  reviewed artifact, and the fresh file's flags are validated for presence
  and type but ignored for scoring (F6 — a producer cannot relabel cases
  to launder a bad result into a pass);
- a baseline case is missing from the fresh results (an unverifiable
  regression is a failure);
- the fresh results contain no faithful cases at all (nothing to gate);
- the total number of **unsupported claims** in the fresh results
  (sum of n_claims − n_supported_claims over all cases) exceeds the
  baseline's by more than --max-unsupported-growth (default 0, i.e.
  zero-tolerance growth, F8).  This aggregate catches a new case that is
  wrong from birth, which a per-case regression check cannot see because
  there is no baseline score to regress from.

Case ids present in the fresh results but absent from the baseline are a
hard input error (exit 2), not a free pass: the gate will not score
unreviewed cases (refresh the baseline deliberately instead).  A scorer_version mismatch prints a warning and does not
fail (it is a hint to refresh the baseline).  Unreadable/malformed files,
missing required fields (including per-case n_claims /
n_supported_claims), or a non-lexical method are exit 2.

## Refreshing the baseline

The committed baseline is exactly one deterministic run of the runner over
the committed fixtures.  To refresh it (only when the scorer rules or the
fixtures intentionally change):

1. bump SCORER_VERSION in evaluations/eval_pipelines/faithfulness.py if the
   scoring rules changed;
2. run the runner (command above) writing to a scratch path;
3. inspect the per-case scores — every change must be explainable;
4. copy the scratch file over
   evaluations/eval_pipelines/baseline/faithfulness_baseline.json;
5. run pytest tests/evaluations/ -q — the drift canary test
   test_fresh_results_match_committed_baseline fails if baseline, scorer,
   and fixtures disagree;
6. commit baseline + fixtures + scorer together.

## Adding golden cases

The fixtures file is one JSON object per line:

```
{"id": "fs-009", "category": "fully-supported", "should_be_faithful": true,
 "contexts": ["..."], "answer": "...", "note": "why this case exists"}
```

- id: unique, non-empty (the gate matches cases by id);
- category: fully-supported | partially-supported | hallucinated |
  negation-flipped | empty-context | plausible-contradiction (free-form;
  the gate does not care);
- should_be_faithful: true for cases that SHOULD score high (the faithful
  set the min-score gate is computed over);
- contexts: list of strings (may be empty for empty-context cases);
- answer: the generated answer text.

Hand-write cases around a small set of recurring "facts" so that
fully-supported cases reuse the context vocabulary (the scorer rewards
verbatim grounding — that is by design for a lexical baseline) and the
negative categories deviate in exactly one controlled way.  After adding
cases, refresh the baseline (above).

## LLM judge (optional, never in CI)

```
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # adjust to your checkout
spec = importlib.util.spec_from_file_location(
    "faithfulness",
    REPO_ROOT / "evaluations" / "eval_pipelines" / "faithfulness.py",
)
faithfulness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(faithfulness)

def my_llm_judge(answer, contexts):
    # call your model endpoint; must return {"score": 0..1, "reason": str}
    ...

result = faithfulness.judge_faithfulness(answer, contexts, my_llm_judge)
```

**Import by path, not through the package.**
evaluations/eval_pipelines/__init__.py imports the ViDoRe benchmark stack
(torch, PIL, vidore_benchmark), which is not installed in the server venv or
in CI, so
`from evaluations.eval_pipelines.faithfulness import ...` fails there with
`ModuleNotFoundError: No module named 'torch'`.  The path-based loader
above is THE way to import the scorer: it is what the tests and the
runner's direct-execution fallback do, and a test
(test_path_import_works_with_torch_absent) pins that the import works in a
fresh interpreter without torch and never pulls torch in.

The judge must return a dict with a numeric score in [0, 1]; anything else
raises ValueError/TypeError.  Use it to sample semantic quality out of band;
do not feed its output to scripts/ci_eval_gate.py — the gate rejects
non-lexical method documents by design.
