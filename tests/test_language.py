"""Tests for language code normalization.

These exercise the function the language-mismatch warning depends on.
Without coverage here, a regression in the normalizer silently disabled
the entire mismatch UX (which is exactly what happened before).
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from transcriber.language import MULTI_SENTINEL, normalize_language_to_iso


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # Direct ISO-639-1 passthrough
        ("en", "en"),
        ("de", "de"),
        ("fr", "fr"),
        ("es", "es"),
        ("zh", "zh"),
        # Locale suffix stripped
        ("en-US", "en"),
        ("fr-FR", "fr"),
        ("de-DE", "de"),
        ("pt-BR", "pt"),
        # Whitespace and case variation
        ("  EN  ", "en"),
        ("DE", "de"),
        # Whisper full names (English)
        ("english", "en"),
        ("german", "de"),
        ("french", "fr"),
        ("spanish", "es"),
        ("italian", "it"),
        ("portuguese", "pt"),
        ("dutch", "nl"),
        ("russian", "ru"),
        ("japanese", "ja"),
        ("chinese", "zh"),
        # Native-name aliases
        ("deutsch", "de"),
        ("français", "fr"),
        ("francais", "fr"),
        ("español", "es"),
        ("italiano", "it"),
    ],
)
def test_known_codes_normalize(value, expected):
    assert normalize_language_to_iso(value) == expected


def test_multi_sentinel_returned_unchanged():
    """Deepgram Nova-3 multilingual returns "multi" — must surface as-is so
    the UI knows to skip the mismatch comparison rather than treating it
    as a single detected language."""
    assert normalize_language_to_iso("multi") == MULTI_SENTINEL
    assert normalize_language_to_iso("MULTI") == MULTI_SENTINEL


@pytest.mark.parametrize("value", ["", "   ", None])
def test_empty_input_returns_none(value):
    assert normalize_language_to_iso(value) is None


@pytest.mark.parametrize(
    "value",
    [
        # Two-letter strings that look like codes but aren't ISO-639-1.
        # The old fast-path accepted these and produced spurious warnings.
        "zz",
        "xx",
        "qq",
        # Locale tags whose head isn't a valid ISO code
        "zh-Hant",  # script subtag but head "zh" *is* valid → see separate test
        # Random non-language strings
        "klingon",
        "auto",
        "unknown",
        "english/german",
    ],
)
def test_unknown_values_return_none_or_valid(value):
    """Anything we can't confidently map should return None so the UI can
    fall back to displaying the raw string. The one quirk: zh-Hant has a
    valid head ('zh') so it normalizes to 'zh' — that's correct behavior."""
    result = normalize_language_to_iso(value)
    if value == "zh-Hant":
        assert result == "zh"
    else:
        assert result is None


def test_two_letter_garbage_no_longer_passes():
    """Regression guard for the original review finding: the previous
    implementation accepted any two-alpha string. 'zz' would normalize to
    'zz' and trigger a false mismatch warning when the user selected 'en'.
    Now it returns None and the UI falls back to the raw string."""
    assert normalize_language_to_iso("zz") is None
    assert normalize_language_to_iso("ZZ") is None