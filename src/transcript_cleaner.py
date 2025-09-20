"""Utilities for transcript clean-up heuristics.

This module intentionally focuses on pure helper routines so that each
transformation can be tested in isolation.  The new requirement from the user
centres on recognising short false starts where one question opener is
immediately followed by another.  The heuristics below embrace the project's
"50% rule" by layering several simple cues (lexical classes, suffix matches,
and punctuation context) so that even if one heuristic misses a corner case,
the combination still succeeds more than half of the time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Token model and basic text utilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Token:
    """Represents a single word in the transcript.

    The clean-up rules in this code base operate only on the textual tokens and
    never mutate global state.  Storing both the original slice and a normalised
    lower-case version keeps the downstream logic side-effect free and allows us
    to describe the behaviour in tests without worrying about accidental
    lower/upper-case mismatches.
    """

    text: str
    lower: str
    start: int
    end: int


_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*")


def _normalise(word: str) -> str:
    """Normalise a token for comparison.

    Apostrophes appear in several different Unicode forms inside transcripts.
    Normalisation keeps comparisons straightforward and predictable.
    """

    return word.lower().replace("’", "'")


def _tokenise_words(text: str) -> List[Token]:
    """Return a list of tokens extracted from *text*.

    The function purposefully avoids altering the source string; instead it
    exposes start/end spans so that higher level heuristics can decide what to
    replace while keeping those replacements pure.
    """

    return [
        Token(match.group(0), _normalise(match.group(0)), match.start(), match.end())
        for match in _WORD_RE.finditer(text)
    ]


# ---------------------------------------------------------------------------
# Word classification helpers
# ---------------------------------------------------------------------------

# Question words give us the interrogative style that frequently appears inside
# false starts.
QUESTION_WORDS = {
    "what",
    "when",
    "where",
    "who",
    "whom",
    "whose",
    "why",
    "how",
    "which",
}

QUESTION_AUXILIARIES = {
    "am",
    "are",
    "is",
    "was",
    "were",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "can",
    "could",
    "shall",
    "should",
    "will",
    "would",
    "may",
    "might",
    "must",
    "ought",
    "need",
}

PERSONAL_PRONOUNS = {
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "it",
    "me",
    "him",
    "her",
    "us",
    "them",
}

CONTRACTION_PRONOUNS = {
    "i'm",
    "i'd",
    "i'll",
    "i've",
    "we're",
    "we've",
    "we'll",
    "we'd",
    "they're",
    "they've",
    "they'll",
    "they'd",
    "you're",
    "you've",
    "you'll",
    "you'd",
    "he's",
    "she's",
    "it's",
    "that's",
}

DETERMINERS = {"this", "that", "these", "those"}

LEADING_NOISE = {
    "and",
    "but",
    "so",
    "or",
    "nor",
    "yet",
    "then",
    "uh",
    "um",
    "erm",
    "er",
    "ah",
    "oh",
    "hmm",
    "mm",
    "well",
}

INSERTABLE_PRONOUNS = PERSONAL_PRONOUNS | {"i'm", "we're", "they're", "you're"}


def _categorise(word: str) -> str:
    """Classify *word* into a coarse lexical category.

    Returning explicit categories keeps the decision surface transparent.  The
    heuristics lean on the category overlap (pronoun vs. question word, etc.) to
    decide when a repeated suffix is likely a false start rather than two genuine
    sentences.
    """

    normalised = _normalise(word)
    if normalised in QUESTION_WORDS:
        return "question"
    if normalised in QUESTION_AUXILIARIES:
        return "auxiliary"
    if normalised in PERSONAL_PRONOUNS or normalised in CONTRACTION_PRONOUNS:
        return "pronoun"
    if normalised in DETERMINERS:
        return "determiner"
    if word and word[0].isupper() and not word.isupper():
        return "proper"
    return "other"


def _lead_words_compatible(left: Token, right: Token) -> bool:
    """Return ``True`` if the words preceding the suffix form a false start."""

    if left.lower in LEADING_NOISE or right.lower in LEADING_NOISE:
        return False

    if left.lower == right.lower:
        return False

    left_category = _categorise(left.text)
    right_category = _categorise(right.text)

    if left_category == right_category and left_category in {"question", "auxiliary", "pronoun"}:
        return True

    if {left_category, right_category} <= {"question", "auxiliary"}:
        return True

    if left_category == right_category == "pronoun":
        return True

    if left_category == right_category == "proper":
        return True

    return False


# ---------------------------------------------------------------------------
# Pattern detectors
# ---------------------------------------------------------------------------

MAX_SEQUENCE = 6


def _tokens_equal(tokens_a: Sequence[Token], tokens_b: Sequence[Token]) -> bool:
    """Case-insensitive token comparison helper."""

    if len(tokens_a) != len(tokens_b):
        return False
    return all(token_a.lower == token_b.lower for token_a, token_b in zip(tokens_a, tokens_b))


def _detect_suffix_swap(left_tokens: Sequence[Token], right_tokens: Sequence[Token]) -> bool:
    """Detect ``A rest`` followed by ``B rest`` patterns.

    The rule mirrors the examples ``"Do you can you"`` and
    ``"You can't go. He can't go"``.  Both phrases share a trailing ``rest`` but
    differ in the immediately preceding word, indicating a false start that the
    user wants joined with ``--``.
    """

    left_len = len(left_tokens)
    right_len = len(right_tokens)

    for l_size in range(2, left_len + 1):
        left_seq = left_tokens[left_len - l_size : left_len]
        for r_size in range(2, right_len + 1):
            right_seq = right_tokens[:r_size]
            max_common = min(l_size, r_size) - 1
            for common in range(max_common, 0, -1):
                left_suffix = left_seq[-common:]
                right_suffix = right_seq[-common:]
                if not _tokens_equal(left_suffix, right_suffix):
                    continue

                left_prefix_len = len(left_seq) - common
                right_prefix_len = len(right_seq) - common

                if left_prefix_len != 1 or right_prefix_len != 1:
                    continue

                left_lead = left_seq[-common - 1]
                right_lead = right_seq[-common - 1]

                if not _lead_words_compatible(left_lead, right_lead):
                    continue

                return True
    return False


def _detect_inserted_word_pattern(left_tokens: Sequence[Token], right_tokens: Sequence[Token]) -> bool:
    """Detect ``Word rest`` followed by ``Word insert rest`` patterns.

    This rule handles the user's ``"When was.  When I was"`` scenario where the
    second clause repeats the opener but inserts an additional pronoun.  We keep
    the insert short and limited to personal pronouns to avoid catching unrelated
    sentences.
    """

    left_len = len(left_tokens)
    right_len = len(right_tokens)

    for l_size in range(2, left_len + 1):
        left_seq = left_tokens[left_len - l_size : left_len]
        if l_size < 2:
            continue
        for r_size in range(l_size + 1, right_len + 1):
            right_seq = right_tokens[:r_size]
            if left_seq[0].lower != right_seq[0].lower:
                continue

            # The first sequence must be prefix + rest with no insertions.
            for rest_size in range(1, l_size):
                prefix_size = 1  # limit to a single leading opener for safety
                if l_size != prefix_size + rest_size:
                    continue

                left_rest = left_seq[-rest_size:]
                right_rest = right_seq[-rest_size:]
                if not _tokens_equal(left_rest, right_rest):
                    continue

                inserted = right_seq[prefix_size : r_size - rest_size]
                if not inserted or len(inserted) > 2:
                    continue

                if not all(token.lower in INSERTABLE_PRONOUNS for token in inserted):
                    continue

                opener_category = _categorise(left_seq[0].text)
                if opener_category not in {"question", "auxiliary"}:
                    continue

                return True
    return False


# ---------------------------------------------------------------------------
# Gap replacement helpers
# ---------------------------------------------------------------------------

PUNCT_THAT_ENDS_SENTENCE = {".", "?", "!"}


def _gap_requires_lowercase(gap: str) -> bool:
    """Return ``True`` when the removed punctuation implies lower-casing."""

    return any(symbol in gap for symbol in PUNCT_THAT_ENDS_SENTENCE)


def _should_lowercase_second(token: Token, gap_text: str) -> bool:
    """Decide whether to lowercase the next token after replacing the gap."""

    if not _gap_requires_lowercase(gap_text):
        return False
    if not token.text or not token.text[0].isalpha() or not token.text[0].isupper():
        return False
    if token.text.isupper():
        return False
    if token.lower == "i" or token.lower.startswith("i'"):
        return False

    category = _categorise(token.text)
    return category in {"question", "auxiliary", "pronoun"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_question_opener_dash(text: str) -> str:
    """Insert ``" -- "`` between consecutive question openers.

    The function keeps the rest of the transcript intact, replacing only the gap
    between two matching phrases.  Each candidate boundary is evaluated against
    two complementary heuristics: one that recognises swapped openers (``A rest``
    vs ``B rest``) and a second that recognises simple insertions
    (``When was`` → ``When I was``).  This layered approach mirrors the 50% rule
    guidance by providing multiple chances to confirm that a boundary is truly a
    false start.
    """

    tokens = _tokenise_words(text)
    if len(tokens) < 2:
        return text

    modifications: List[Tuple[int, int, str]] = []
    reserved_spans: set[Tuple[int, int]] = set()

    for boundary in range(len(tokens) - 1):
        left_index_start = max(0, boundary - (MAX_SEQUENCE - 1))
        left_slice = tokens[left_index_start : boundary + 1]
        right_slice = tokens[boundary + 1 : boundary + 1 + MAX_SEQUENCE]
        if not right_slice:
            continue

        gap_start = tokens[boundary].end
        gap_end = tokens[boundary + 1].start
        gap_text = text[gap_start:gap_end]

        if "--" in gap_text or "—" in gap_text or "–" in gap_text:
            continue
        if any(char.isalpha() or char.isdigit() for char in gap_text):
            # Unexpected text between the two word spans – skip the boundary.
            continue

        matched = _detect_suffix_swap(left_slice, right_slice)
        if not matched:
            matched = _detect_inserted_word_pattern(left_slice, right_slice)

        if not matched:
            continue

        gap_span = (gap_start, gap_end)
        if gap_span not in reserved_spans:
            modifications.append((gap_start, gap_end, " -- "))
            reserved_spans.add(gap_span)

        next_token = tokens[boundary + 1]
        if _should_lowercase_second(next_token, gap_text):
            lowercase_span = (next_token.start, next_token.start + 1)
            if lowercase_span not in reserved_spans:
                new_letter = next_token.text[0].lower()
                modifications.append((next_token.start, next_token.start + 1, new_letter))
                reserved_spans.add(lowercase_span)

    if not modifications:
        return text

    modifications.sort(key=lambda item: item[0], reverse=True)
    result = text
    for start, end, replacement in modifications:
        result = result[:start] + replacement + result[end:]
    return result


__all__ = ["apply_question_opener_dash"]
