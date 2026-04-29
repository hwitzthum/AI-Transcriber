"""Unit tests for text_processor.render_transcript_html.

Covers the XSS-prevention escape pass and the paired-underscore italic
conversion that replaced the previous global `_`→`<em>` substitution.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from transcriber import text_processor


def test_escapes_raw_html_in_transcript():
    """A <script> tag in the transcript must be rendered as literal text."""
    result = text_processor.render_transcript_html("<script>alert(1)</script>")
    assert "<script>" not in result, "raw <script> must be escaped"
    assert "&lt;script&gt;" in result, f"expected escaped tag, got: {result}"


def test_escapes_attribute_injection():
    """Ampersands and quotes must be escaped to prevent attribute breakouts."""
    result = text_processor.render_transcript_html('Hello "world" & friends')
    assert '"' not in result.replace('color: #9ca3af;', ''), "quotes must be escaped"
    assert "&amp;" in result, "ampersand must be escaped"


def test_paired_bold_markers_become_strong():
    result = text_processor.render_transcript_html("Hello **world** and **again**")
    assert "<strong>world</strong>" in result
    assert "<strong>again</strong>" in result


def test_paired_underscores_become_em():
    result = text_processor.render_transcript_html("This is _important_ stuff")
    assert "<em" in result and "important</em>" in result


def test_filename_with_multiple_underscores_is_not_corrupted():
    """The bug case: 'test_audio_file' must NOT become '<em>audio<em>file'."""
    result = text_processor.render_transcript_html("Open test_audio_file in editor")
    # Either the filename renders as literal text OR a single paired match —
    # what must NOT happen is the old behavior where every `_` became `<em>`,
    # producing unbalanced tags.
    assert "<em>audio<em>file" not in result
    # Make sure no orphaned/unclosed em tags remain
    assert result.count("<em") == result.count("</em>"), (
        f"unbalanced <em> tags in output: {result}"
    )


def test_speaker_label_renders_correctly():
    """**Speaker 1:** must convert to <strong>Speaker 1:</strong>."""
    result = text_processor.render_transcript_html("**Speaker 1:** Hello there")
    assert "<strong>Speaker 1:</strong>" in result


def test_search_highlight_marks_query():
    """Search query must be wrapped in <mark> in the rendered HTML."""
    result = text_processor.render_transcript_html(
        "Find the word here please", search_query="word"
    )
    assert "<mark" in result and "</mark>" in result
    assert "word" in result


def test_search_query_does_not_match_inside_escaped_html():
    """Searching for '<script>' should not produce a literal <script> tag."""
    result = text_processor.render_transcript_html(
        "<script>", search_query="<script>"
    )
    assert "<script>" not in result, "literal <script> must not appear"


def test_empty_transcript_returns_empty_string():
    assert text_processor.render_transcript_html("") == ""


def test_no_search_query_renders_normally():
    result = text_processor.render_transcript_html("Plain text without markers")
    assert result == "Plain text without markers"
