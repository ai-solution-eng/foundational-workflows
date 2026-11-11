"""Deterministic, offline lexical faithfulness scoring (Feature E, 2026-11-10).

Scores whether an answer is grounded in its source contexts using pure
lexical overlap: no network, no LLM, no third-party imports.  This is the
default scorer for the CI gate: every function here is deterministic, so
repeated runs over the same inputs produce identical results, which is what
makes a committed baseline comparable in CI.

What this measures
------------------
An answer is split into sentence-level claims.  Each claim is compared
against every context: content tokens (lowercased, stopword-free) found in
a context count as grounding, with light prefix matching so that
"support"/"supports" and "cluster"/"clusters" agree.  A claim is supported
when at least OVERLAP_THRESHOLD (0.7) of its content tokens are found in a
single context, no matched token carries opposite negation on the two
sides (context "does not require X" vs claim "requires X" is a conflict,
never support; negation is detected by single-token markers such as not,
no, never, cannot, won't and by two-token patterns such as "fails to",
"unable to", "no longer", "instead of", "rather than"), and every number
in the claim (digit numbers such as 768, 4096, 1.19.0 and number words
such as three, five) appears in that same best context -- a numeric
contradiction vetoes the claim no matter how high the token overlap is.
A context rejected by the negation guard or the numeric guard is not
reported as best_context (None instead).  A claim made up purely of
stopwords carries a per-claim "note" ("no scorable claim; vacuously
scored") so readers can tell it apart from a genuinely supported claim.
The case score is the fraction of supported claims.

What this does NOT measure (see documentation/FAITHFULNESS-EVAL.md)
-------------------------------------------------------------------
This is not entailment and not factuality.  Synonyms are not recognised
("cars" vs "automobiles" does not match); a claim can reach the 0.7
content-coverage bar while still being semantically wrong (an entity or
direction swap that keeps the token overlap high is missed when no number
changes); the negation guard is a fixed 2-token window over a closed list
of markers and two-token patterns (negations farther than 2 tokens away,
double negation, and failure forms outside the list -- "short of",
"barring", bare "fails" without "to", "lacking" -- are missed); sentence
splitting is a regex heuristic that mis-splits abbreviations such as
"e.g.".  For semantic scoring use judge_faithfulness with an LLM judge;
that path is deliberately never run in CI.

Public API
----------
extract_claims(answer) -> list[str]
score_faithfulness(answer, contexts) -> dict
judge_faithfulness(answer, contexts, judge_fn) -> dict
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

__all__ = [
    "METHOD",
    "NEGATION_MARKERS",
    "OVERLAP_THRESHOLD",
    "SCORER_VERSION",
    "STOPWORDS",
    "extract_claims",
    "judge_faithfulness",
    "score_faithfulness",
]

#: Value of the "method" field in result documents.  The CI gate only
#: evaluates documents produced by this scorer; anything else (e.g. an LLM
#: judge) is rejected at the gate by construction.
METHOD = "lexical"

#: Bump when the scoring rules change.  The gate warns (does not fail) when
#: the baseline and a results file disagree on this value.
SCORER_VERSION = 3

#: A claim is supported when at least this fraction of its content tokens
#: is found in one context (and no number in the claim is contradicted by
#: that context -- see _NUM_RE / _NUMBER_WORDS).
OVERLAP_THRESHOLD = 0.7

#: Prefix matches are accepted only when the shorter token has at least
#: this many characters (avoids "art"~"article", "can"~"cannot", ...).
_PREFIX_MIN_LEN = 4

#: The negation guard looks at most this many tokens before a matched token
#: for a negation marker.
_NEGATION_WINDOW = 2

#: Numeric contradiction guard: every digit number in a claim (e.g. 768,
#: 4096, 1.19.0 -- the regex keeps dotted versions as one number) must
#: appear in the best context, or the claim is unsupported.  This catches
#: the most damaging RAG hallucination class: context vocabulary reused
#: around a wrong number.
_NUM_RE = re.compile(r"\b\d+(?:\.\d+)*\b")

#: Number words in a claim must appear (exactly, no prefix matching) among
#: the best context's tokens, or the claim is unsupported ("five" vs "three").
_NUMBER_WORDS: frozenset[str] = frozenset(
    {
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "hundred",
        "thousand",
        "million",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])[\s]+|\n+")

#: English stopwords removed before overlap scoring.  Negation markers that
#: are also stopwords ("not", "no", "never", ...) are still seen by the
#: negation guard, which operates on the full token sequence before
#: stopword removal.
STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "about",
        "above",
        "across",
        "after",
        "again",
        "against",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "behind",
        "below",
        "beneath",
        "beside",
        "between",
        "beyond",
        "but",
        "by",
        "can",
        "cannot",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "done",
        "down",
        "during",
        "each",
        "even",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "like",
        "made",
        "makes",
        "make",
        "making",
        "may",
        "me",
        "might",
        "more",
        "most",
        "must",
        "my",
        "myself",
        "neither",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "one",
        "only",
        "onto",
        "or",
        "other",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "per",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "upon",
        "us",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "within",
        "without",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }
)

#: Tokens that flip the polarity of what they negate.  A matched token that
#: is one of these is negated by itself; a marker within _NEGATION_WINDOW
#: tokens before a matched token negates that token too.
NEGATION_MARKERS: frozenset[str] = frozenset(
    {
        "cannot",
        "can't",
        "couldnt",
        "couldn't",
        "didnt",
        "didn't",
        "dont",
        "don't",
        "doesnt",
        "doesn't",
        "isnt",
        "isn't",
        "wasnt",
        "wasn't",
        "werent",
        "weren't",
        "arent",
        "aren't",
        "wont",
        "won't",
        "wouldnt",
        "wouldn't",
        "shouldnt",
        "shouldn't",
        "mustnt",
        "mustn't",
        "no",
        "not",
        "never",
        "neither",
        "nor",
        "none",
    }
)

#: Two-token negation patterns (token tuples, over the full token sequence,
#: stopwords included).  A token is negated when a pattern ends on the
#: token itself or within _NEGATION_WINDOW tokens before it.  These cover
#: the failure forms a single marker token does not: "fails to X",
#: "failed to X", "unable to X" (no marker token at all), "instead of X" /
#: "rather than X" (the phrase negates its OBJECT, which follows it), and
#: "no longer X" (explicit; the lone "no" already reaches it at distance 2).
#: Auxiliary+"not" pairs ("is not", "does not", ...) need no pattern
#: because the lone "not" token falls inside the window.
_NEGATION_PATTERNS: tuple[tuple[str, ...], ...] = (
    ("fails", "to"),
    ("failed", "to"),
    ("unable", "to"),
    ("no", "longer"),
    ("instead", "of"),
    ("rather", "than"),
)


def _tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens.  Apostrophes stay inside a token (so
    "doesn't" is one token); hyphens and other punctuation split."""
    return _TOKEN_RE.findall(text.lower())


def extract_claims(answer: str) -> list[str]:
    """Split an answer into sentence-level claims.

    Deterministic regex split: sentences end at '.'/'!'/'?' followed by
    whitespace, and at newlines (which also covers bullet lists).
    Abbreviations such as "e.g." mis-split; that is a known, documented
    limitation of lexical scoring.  Empty/whitespace answers yield [].
    """
    if not isinstance(answer, str):
        raise TypeError(f"answer must be str, got {type(answer).__name__}")
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(answer) if s.strip()]


def _negated_at(tokens: Sequence[str], i: int) -> bool:
    """True when the token at index i is negated: it is a negation marker
    itself, a marker occurs within _NEGATION_WINDOW tokens before it, a
    _NEGATION_PATTERNS phrase ends on it (self-negating), or a phrase ends
    within _NEGATION_WINDOW tokens before it (over the full token
    sequence, stopwords included)."""
    if tokens[i] in NEGATION_MARKERS:
        return True
    lo = max(0, i - _NEGATION_WINDOW)
    if any(t in NEGATION_MARKERS for t in tokens[lo:i]):
        return True
    for pat in _NEGATION_PATTERNS:
        ln = len(pat)
        if i + 1 >= ln and tuple(tokens[i - ln + 1 : i + 1]) == pat:
            return True
        for d in range(1, _NEGATION_WINDOW + 1):
            end = i - d
            if end - ln + 1 < 0:
                continue
            if tuple(tokens[end - ln + 1 : end + 1]) == pat:
                return True
    return False


def _tokens_match(a: str, b: str) -> bool:
    """Exact match, or light prefix matching for inflection ("support"~
    "supports").  Prefix matching is disabled when either token is a
    negation marker, so "does" never matches "doesn't"."""
    if a == b:
        return True
    if a in NEGATION_MARKERS or b in NEGATION_MARKERS:
        return False
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) < _PREFIX_MIN_LEN:
        return False
    return longer.startswith(shorter)


def score_faithfulness(answer: str, contexts: list[str]) -> dict:
    """Score one answer against its contexts.

    Returns:

        {
          "score": float,               # fraction of supported claims, 0..1
          "claims": [
            {"claim": str, "supported": bool, "best_context": int | None,
             "note": str},              # "note" only on vacuous claims
            ...
          ],
          "method": "lexical",
        }

    plus a top-level "note" key only when the answer has no extractable
    claim, in which case the score is vacuously 1.0 (there is nothing to
    contradict).  A claim made up purely of stopwords is vacuously scored
    too and carries its own per-claim "note": "no scorable claim;
    vacuously scored".

    Deterministic: the same inputs always yield the same output.  No
    network, no LLM.  An empty contexts list can never support a claim.
    A context string that is empty or whitespace is skipped.  Support
    requires all three of: content-token coverage of at least
    OVERLAP_THRESHOLD in one context, no negation conflict, and no numeric
    contradiction with that context.  A context rejected by a guard is not
    reported as best_context (None).  Raises TypeError on a non-str answer
    or contexts that are not a list of str.
    """
    claims = extract_claims(answer)
    if not isinstance(contexts, list) or not all(isinstance(c, str) for c in contexts):
        raise TypeError(f"contexts must be a list of str, got {type(contexts).__name__}")

    ctx_tokens = [_tokenize(c) for c in contexts]
    claim_results: list[dict[str, Any]] = []
    supported_count = 0

    for claim in claims:
        full = _tokenize(claim)
        content = [(j, t) for j, t in enumerate(full) if t not in STOPWORDS]
        best_idx: int | None = None
        best_overlap = 0.0
        claim_note: str | None = None

        if not content:
            # No content tokens: nothing lexical can contradict the claim.
            # Vacuously supported when some context is non-empty.  Flagged
            # with a note so a vacuous claim is distinguishable from a
            # genuinely supported one (F7: "unmeasurable" != "supported").
            supported = any(ctx_tokens)
            claim_note = "no scorable claim; vacuously scored"
        else:
            supported = False
            for ci, xt in enumerate(ctx_tokens):
                if not xt:
                    continue
                matched = 0
                conflict = False
                for j, ct in content:
                    for k, xt2 in enumerate(xt):
                        if _tokens_match(ct, xt2):
                            matched += 1
                            if _negated_at(full, j) != _negated_at(xt, k):
                                conflict = True
                            break
                if conflict:
                    continue
                overlap = matched / len(content)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = ci
            supported = best_overlap >= OVERLAP_THRESHOLD
            if supported and best_idx is not None:
                # Numeric contradiction guard: every number in the claim
                # must appear in the best context.  A context with high
                # token overlap but a different number (768 vs 4096,
                # 429 vs 500, 1.19.0 vs 1.29.0, three vs five) vetoes.
                claim_nums = set(_NUM_RE.findall(claim))
                claim_num_words = {t for _, t in content if t in _NUMBER_WORDS}
                ctx_text = contexts[best_idx]
                ctx_nums = set(_NUM_RE.findall(ctx_text))
                digit_conflict = bool(claim_nums) and not claim_nums.issubset(ctx_nums)
                word_conflict = bool(claim_num_words) and not claim_num_words.issubset(set(ctx_tokens[best_idx]))
                if digit_conflict or word_conflict:
                    supported = False
                    best_idx = None

        if supported:
            supported_count += 1
        cr = {"claim": claim, "supported": supported, "best_context": best_idx}
        if claim_note is not None:
            cr["note"] = claim_note
        claim_results.append(cr)

    if not claim_results:
        return {
            "score": 1.0,
            "claims": [],
            "method": METHOD,
            "note": "no extractable claims; vacuously supported",
        }
    return {
        "score": supported_count / len(claim_results),
        "claims": claim_results,
        "method": METHOD,
    }


def judge_faithfulness(
    answer: str,
    contexts: list[str],
    judge_fn: Callable[[str, list[str]], dict],
) -> dict:
    """Run an external (typically LLM) judge on the same case.

    judge_fn(answer, contexts) must return a dict with at least
    {"score": number in [0, 1]}; an optional "reason" string is passed
    through.  The returned document mirrors score_faithfulness' shape with
    "method": "llm-judge" and no lexical claims.

    This function is for out-of-band semantic evaluation only.  It is
    intentionally never used by the CI gate: an LLM judge is
    non-deterministic, needs network access, and its cost does not belong
    in a regression gate.  Raises TypeError if judge_fn is not callable or
    its verdict is not a dict with a numeric 'score'; ValueError if the
    score is outside [0, 1].
    """
    if not callable(judge_fn):
        raise TypeError("judge_fn must be callable")
    verdict = judge_fn(answer, contexts)
    if not isinstance(verdict, dict):
        raise TypeError(f"judge_fn must return a dict, got {type(verdict).__name__}")
    score = verdict.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise TypeError("judge verdict must contain a numeric 'score'")
    score = float(score)
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"judge score {score} outside [0, 1]")
    result: dict[str, Any] = {"score": score, "claims": [], "method": "llm-judge"}
    if "reason" in verdict:
        result["reason"] = str(verdict["reason"])
    return result
