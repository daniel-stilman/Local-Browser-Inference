"""Regression tests for the question-opener dash heuristic."""

import pathlib
import sys

import pytest

# The project keeps application logic inside ``src`` to stay monolithic.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from transcript_cleaner import apply_question_opener_dash  # noqa: E402


def apply(text: str) -> str:
    """Helper for readability inside tests."""

    return apply_question_opener_dash(text)


def test_inserts_dash_between_auxiliaries():
    text = "Do you can you remember the day?"
    expected = "Do you -- can you remember the day?"
    assert apply(text) == expected


def test_handles_period_and_lowercases_second_clause():
    text = "When was.  When I was younger."
    expected = "When was -- when I was younger."
    assert apply(text) == expected


def test_handles_complex_suffix_match():
    text = (
        "And how often do you have.  Would you have to take shelter in the bunker?"
    )
    expected = (
        "And how often do you have -- would you have to take shelter in the bunker?"
    )
    assert apply(text) == expected


def test_handles_pronoun_swap_with_lowercasing():
    text = "You can't go. He can't go there."
    expected = "You can't go -- he can't go there."
    assert apply(text) == expected


def test_handles_names_without_lowercasing():
    text = "David is really. Daryll is really fast."
    expected = "David is really -- Daryll is really fast."
    assert apply(text) == expected


def test_respects_existing_dash():
    text = "Do you -- can you remember the day?"
    assert apply(text) == text


def test_does_not_modify_regular_sentences():
    text = "This is nice. That is better."
    assert apply(text) == text


def test_handles_commas_and_spacing():
    text = "Can they, should they stay?"
    expected = "Can they -- should they stay?"
    assert apply(text) == expected


def test_handles_newline_boundaries():
    text = "When was.\nWhen I was younger"
    expected = "When was -- when I was younger"
    assert apply(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("When was when I was there", "When was -- when I was there"),
        ("And then we left. They stayed behind.", "And then we left. They stayed behind."),
    ],
)
def test_avoids_unrelated_sequences(text: str, expected: str):
    assert apply(text) == expected
