"""Tests for low-confidence word highlighting.

The user-visible behaviour is: words at or below the chosen confidence
threshold are wrapped in ``~~word~~`` markers; the editor preview
renders them with an amber background; the DOCX export gives them a
yellow text-highlight; the PDF export drops the markers cleanly.

Coverage: the per-word renderer, the diarized + non-diarized
words-loop paths, the timestamped paragraph extractor, the preview
HTML renderer, the reading-stats stripper, and the DOCX/PDF round-trip.
"""

import os
import sys
from types import SimpleNamespace

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from transcriber import cloud_engine, exporter, text_processor


def _word(text, confidence, start=0.0, end=0.5, speaker=None):
    """Build a Deepgram-shaped word object for the loop helpers."""
    return SimpleNamespace(
        punctuated_word=text,
        word=text.lower().rstrip(".,!?;:"),
        confidence=confidence,
        start=start,
        end=end,
        speaker=speaker,
    )


# ---------------------------------------------------------------------------
# _render_word_token
# ---------------------------------------------------------------------------

def test_render_word_token_passthrough_when_no_threshold():
    w = _word("Hello.", 0.42)
    assert cloud_engine._render_word_token(w, None) == "Hello."


def test_render_word_token_wraps_below_threshold():
    """Words at the threshold are wrapped — defensive choice. A score
    *exactly* at the threshold is on the suspect side, not the
    confident one. Otherwise users who set 0.5 would see nothing
    wrapped on a model that consistently emits 0.5."""
    w = _word("uncertain", 0.55)
    assert cloud_engine._render_word_token(w, 0.6) == "~~uncertain~~"


def test_render_word_token_keeps_high_confidence_unwrapped():
    w = _word("clear.", 0.95)
    assert cloud_engine._render_word_token(w, 0.6) == "clear."


def test_render_word_token_handles_missing_confidence():
    """Words without a confidence score (older models / partial
    responses) must NOT be wrapped — wrapping would imply we have a
    signal we don't actually have."""
    w = SimpleNamespace(punctuated_word="word", word="word", confidence=None)
    assert cloud_engine._render_word_token(w, 0.6) == "word"


def test_render_word_token_empty_word_returns_empty():
    w = SimpleNamespace(punctuated_word=None, word="", confidence=0.1)
    assert cloud_engine._render_word_token(w, 0.6) == ""


# ---------------------------------------------------------------------------
# Diarized words-loop produces wrapped output
# ---------------------------------------------------------------------------

def test_format_diarized_from_words_wraps_low_confidence():
    """The text returned by the diarized formatter must contain the
    ``~~word~~`` marker for the suspect word and not for the others."""
    alt = SimpleNamespace(words=[
        _word("Hello.", 0.95, 0.0, 0.5, speaker=0),
        _word("possibly", 0.40, 0.6, 1.2, speaker=0),
        _word("yes.", 0.92, 1.3, 1.8, speaker=0),
    ])
    rendered = cloud_engine._format_diarized_from_words(alt, low_confidence_threshold=0.6)
    assert "**Speaker 0:**" in rendered
    assert "~~possibly~~" in rendered
    # The confident words must not be wrapped.
    assert "~~Hello.~~" not in rendered
    assert "~~yes.~~" not in rendered


# ---------------------------------------------------------------------------
# Non-diarized words-loop
# ---------------------------------------------------------------------------

def test_format_undiarized_from_words_inserts_paragraph_breaks():
    """Pause ≥1.5 s must produce a paragraph break, matching the
    diarized path's rhythm."""
    alt = SimpleNamespace(words=[
        _word("First.", 0.9, 0.0, 1.0),
        _word("sentence.", 0.9, 1.0, 1.5),
        # 2 s pause → new paragraph.
        _word("Second.", 0.9, 3.5, 4.5),
    ])
    rendered = cloud_engine._format_undiarized_from_words(alt, low_confidence_threshold=0.6)
    assert "\n\n" in rendered
    assert "Second." in rendered.split("\n\n")[1]


def test_format_undiarized_from_words_returns_empty_when_no_words():
    assert cloud_engine._format_undiarized_from_words(SimpleNamespace(words=None), 0.6) == ""
    assert cloud_engine._format_undiarized_from_words(SimpleNamespace(), 0.6) == ""


# ---------------------------------------------------------------------------
# Timestamped paragraph extractor honours the threshold
# ---------------------------------------------------------------------------

def test_extract_paragraphs_uses_word_loop_when_threshold_set():
    """The paragraph-sentence shortcut path has no per-word data, so
    the extractor must fall through to the word loop whenever the
    caller asked for confidence wrapping."""
    alt = SimpleNamespace(
        # paragraphs object is *present* but the threshold should still
        # force us into the words-loop path.
        paragraphs=SimpleNamespace(paragraphs=[
            SimpleNamespace(
                speaker=0, start=0.0, end=2.0,
                sentences=[SimpleNamespace(text="Smart-formatted text.")],
            ),
        ]),
        words=[
            _word("smart", 0.4, 0.0, 0.5, speaker=0),
            _word("formatted", 0.95, 0.6, 1.0, speaker=0),
        ],
    )
    paragraphs = cloud_engine._extract_deepgram_paragraphs(
        alt, diarize=True, low_confidence_threshold=0.6
    )
    assert len(paragraphs) == 1
    # The word-loop output is what landed in the paragraph, not the
    # smart-formatted sentence — proving the threshold redirected us.
    assert "~~smart~~" in paragraphs[0]["text"]
    assert "~~formatted~~" not in paragraphs[0]["text"]
    assert paragraphs[0]["speaker"] == 0


def test_extract_paragraphs_keeps_paragraph_path_when_no_threshold():
    """Threshold None must preserve the existing behaviour — the
    smart-formatted text comes from sentences, not from the word loop."""
    alt = SimpleNamespace(
        paragraphs=SimpleNamespace(paragraphs=[
            SimpleNamespace(
                speaker=0, start=0.0, end=2.0,
                sentences=[SimpleNamespace(text="Sentence one.")],
            ),
        ]),
        words=[_word("sentence", 0.4)],  # would be wrapped if we were in word loop
    )
    paragraphs = cloud_engine._extract_deepgram_paragraphs(alt, diarize=True)
    assert paragraphs[0]["text"] == "Sentence one."
    assert "~~" not in paragraphs[0]["text"]


# ---------------------------------------------------------------------------
# Preview render + reading stats
# ---------------------------------------------------------------------------

def test_preview_renders_low_confidence_with_amber_background():
    rendered = text_processor.render_transcript_html("Hello ~~maybe~~ world.")
    assert "background: #fef3c7" in rendered  # the amber colour we chose
    assert ">maybe<" in rendered  # text preserved between marker tags
    # Tilde marker tokens themselves must not appear in the output.
    assert "~~" not in rendered


def test_preview_low_confidence_composes_with_filler():
    """A word that is BOTH low-confidence and a filler (rare but
    possible if a Deepgram-uncertain ``um`` is later flagged by the
    filler pass) must render with both styles. Verify the inner italic
    is nested inside the amber span."""
    rendered = text_processor.render_transcript_html("She said ~~_um_~~ yeah.")
    assert "background: #fef3c7" in rendered
    assert "<em" in rendered
    # The amber span must wrap the italic, not the other way around —
    # otherwise an export that strips italics would also strip the
    # confidence marker.
    em_idx = rendered.index("<em")
    span_idx = rendered.index("background: #fef3c7")
    assert span_idx < em_idx


def test_reading_stats_strips_low_confidence_markers():
    plain = "Hello world this is fine"
    wrapped = "Hello ~~world~~ this is ~~fine~~"
    assert (
        text_processor.get_reading_stats(plain)["word_count"]
        == text_processor.get_reading_stats(wrapped)["word_count"]
    )


# ---------------------------------------------------------------------------
# Exporter: parser + DOCX run styling + PDF strip
# ---------------------------------------------------------------------------

def test_parse_text_segments_distinguishes_kinds():
    segments = exporter._parse_text_segments(
        "plain _filler_ middle ~~suspect~~ end"
    )
    kinds = [k for _, k in segments]
    assert kinds == [None, "filler", None, "low_confidence", None]


def test_strip_inline_markers_removes_both_marker_types():
    assert exporter._strip_inline_markers("~~suspect~~ _filler_") == "suspect filler"


def test_export_docx_with_low_confidence_succeeds():
    """End-to-end: a transcript with low-confidence markers exports to
    a valid DOCX and contains a run with the yellow highlight set."""
    text = "**Speaker 0:**\nThis is ~~probably~~ accurate."
    blob = exporter.export_docx(text, title="Test")
    assert blob.startswith(b"PK")  # zip header → valid DOCX

    # Open the doc and verify the wrapped word landed in a yellow run.
    import io
    from docx import Document
    from docx.enum.text import WD_COLOR_INDEX

    doc = Document(io.BytesIO(blob))
    runs = [r for p in doc.paragraphs for r in p.runs]
    yellow_runs = [r for r in runs if r.font.highlight_color == WD_COLOR_INDEX.YELLOW]
    assert any(r.text == "probably" for r in yellow_runs), (
        "expected the wrapped word to render as a yellow-highlighted run"
    )


def test_export_pdf_with_low_confidence_runs_strip_helper(monkeypatch):
    """PDF body text can't carry per-word styling without restructuring
    the layout, so the export drops the markers cleanly. We verify the
    behaviour by intercepting :func:`_strip_inline_markers` — checking
    the bytes directly is unreliable because fpdf2 zlib-compresses the
    content stream, and the ``~~`` byte pair occasionally appears in
    the compressed output by chance.
    """
    text = "**Speaker 0:**\nThis is ~~probably~~ accurate."

    received: list[str] = []
    original = exporter._strip_inline_markers

    def _spy(arg):
        received.append(arg)
        return original(arg)

    monkeypatch.setattr(exporter, "_strip_inline_markers", _spy)

    blob = exporter.export_pdf(text, title="Test")
    assert blob.startswith(b"%PDF")

    # The PDF path must route the content (which contains ``~~``)
    # through the strip helper before drawing it. The output of the
    # helper has no markers, so what fpdf2 actually rasterises is
    # marker-free regardless of how the bytes compress.
    assert any("~~probably~~" in arg for arg in received), (
        "PDF export must route low-confidence-wrapped content through "
        "_strip_inline_markers — it did not"
    )
    assert all("~~" not in original(arg) for arg in received)
