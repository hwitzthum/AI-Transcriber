"""Tests for the timestamped-transcript pipeline.

Covers the four pieces that have to agree for [HH:MM:SS] markers to
land in the right places:

* :func:`audio_processor.compute_chunk_offsets` — per-chunk start times.
* :func:`cloud_engine._format_hms` — formatting.
* :func:`cloud_engine._extract_whisper_paragraphs` — Whisper segments
  → paragraph dicts at >=1.5 s pauses.
* :func:`cloud_engine._extract_deepgram_paragraphs` — Deepgram
  paragraphs and the word-loop fallback.
* :func:`cloud_engine._assemble_with_timestamps` — chunk-offset
  translation, time-overlap dedup, and final rendering.
* The renderer / stats / exporter parser updates that were needed to
  keep the rest of the app functioning when markers are present.
"""

import os
import sys
from types import SimpleNamespace

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from transcriber import audio_processor, cloud_engine, exporter, text_processor


# ---------------------------------------------------------------------------
# Format + offset helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "00:00:00"),
        (5.4, "00:00:05"),
        (61.0, "00:01:01"),
        (3599.9, "00:59:59"),
        (3600.0, "01:00:00"),
        (3661.0, "01:01:01"),
        (-1.0, "00:00:00"),  # negative clamped — defensive against flaky timing
    ],
)
def test_format_hms(seconds, expected):
    assert cloud_engine._format_hms(seconds) == expected


def test_compute_chunk_offsets_single_chunk():
    """A file under the size cap produces exactly one chunk at offset 0.

    Without this, the timestamped path would attempt index 0 in a
    zero-length list and fail on the very common single-chunk case.
    """
    offsets = audio_processor.compute_chunk_offsets(
        duration_seconds=120.0,
        file_size_bytes=5 * 1024 * 1024,  # well under the 24 MB cap
        max_bytes=24 * 1024 * 1024,
    )
    assert offsets == [0.0]


def test_compute_chunk_offsets_multi_chunk_step_matches_planner():
    """The offset between consecutive chunks must equal the chunker's
    own step, otherwise inserted timestamps drift relative to the
    actual chunk boundaries."""
    duration = 7200.0  # 2 hours
    max_bytes = 24 * 1024 * 1024  # forces splitting
    file_size = 200 * 1024 * 1024  # large enough to chunk

    chunk_dur, step, num_chunks = audio_processor._plan_ffmpeg_chunks(duration, max_bytes)

    offsets = audio_processor.compute_chunk_offsets(
        duration_seconds=duration,
        file_size_bytes=file_size,
        max_bytes=max_bytes,
    )

    assert len(offsets) == num_chunks
    assert offsets[0] == 0.0
    for i in range(1, len(offsets)):
        # ~1e-9 tolerance is plenty — both sides are derived from the
        # same float division, but floating-point drift can still nibble
        # the last bit.
        assert abs((offsets[i] - offsets[i - 1]) - step) < 1e-6


# ---------------------------------------------------------------------------
# Whisper segment → paragraph extraction
# ---------------------------------------------------------------------------

def _whisper_response(segments):
    """Build a mock verbose_json response with the given segments."""
    return SimpleNamespace(
        segments=[SimpleNamespace(**s) for s in segments],
    )


def test_whisper_paragraphs_split_on_long_pause():
    """A gap of ≥1.5 s between segments must produce a new paragraph.

    Mirrors the Deepgram pause-paragraph rule so timestamped output
    looks consistent across providers."""
    response = _whisper_response([
        {"start": 0.0, "end": 5.0, "text": " First sentence."},
        {"start": 5.5, "end": 8.0, "text": " Still close."},
        # 2 s gap → new paragraph.
        {"start": 10.0, "end": 12.0, "text": " After the pause."},
    ])
    paragraphs = cloud_engine._extract_whisper_paragraphs(response)
    assert len(paragraphs) == 2
    assert paragraphs[0]["text"].endswith("Still close.")
    assert paragraphs[0]["start_sec"] == 0.0
    assert paragraphs[0]["end_sec"] == 8.0
    assert paragraphs[1]["text"] == "After the pause."
    assert paragraphs[1]["start_sec"] == 10.0
    # Whisper has no diarization → speaker is always None.
    assert all(p["speaker"] is None for p in paragraphs)


def test_whisper_paragraphs_empty_segments_returns_empty():
    assert cloud_engine._extract_whisper_paragraphs(SimpleNamespace(segments=[])) == []
    assert cloud_engine._extract_whisper_paragraphs(SimpleNamespace()) == []


# ---------------------------------------------------------------------------
# Deepgram paragraph extraction
# ---------------------------------------------------------------------------

def _deepgram_alternative_with_paragraphs(paragraphs):
    """Build a mock alternative carrying Deepgram's paragraphs structure."""
    return SimpleNamespace(
        paragraphs=SimpleNamespace(
            paragraphs=[
                SimpleNamespace(
                    speaker=p.get("speaker"),
                    start=p["start"],
                    end=p["end"],
                    sentences=[
                        SimpleNamespace(text=s) for s in p["sentences"]
                    ],
                )
                for p in paragraphs
            ]
        ),
        words=None,
    )


def test_deepgram_paragraphs_diarized_keeps_speakers():
    alt = _deepgram_alternative_with_paragraphs([
        {"speaker": 0, "start": 0.0, "end": 5.0, "sentences": ["Hello.", "How are you?"]},
        {"speaker": 1, "start": 5.5, "end": 9.0, "sentences": ["I am well."]},
    ])
    out = cloud_engine._extract_deepgram_paragraphs(alt, diarize=True)
    assert [p["speaker"] for p in out] == [0, 1]
    assert out[0]["text"] == "Hello. How are you?"
    assert out[0]["start_sec"] == 0.0
    assert out[1]["start_sec"] == 5.5


def test_deepgram_paragraphs_undiarized_drops_speaker():
    """When diarization is off, the speaker field on Deepgram's
    paragraph isn't meaningful — surfacing it would render
    ``**Speaker 0:**`` in non-diarized output."""
    alt = _deepgram_alternative_with_paragraphs([
        {"speaker": 0, "start": 0.0, "end": 5.0, "sentences": ["One."]},
    ])
    out = cloud_engine._extract_deepgram_paragraphs(alt, diarize=False)
    assert all(p["speaker"] is None for p in out)


def test_deepgram_word_loop_fallback_splits_on_speaker_change():
    """When ``alternative.paragraphs`` is missing, fall back to the
    word loop. Speaker change must produce a new paragraph block."""
    words = [
        SimpleNamespace(speaker=0, start=0.0, end=1.0, punctuated_word="Hello.", word="hello"),
        SimpleNamespace(speaker=0, start=1.1, end=2.0, punctuated_word="There.", word="there"),
        SimpleNamespace(speaker=1, start=2.2, end=3.0, punctuated_word="Hi.", word="hi"),
    ]
    alt = SimpleNamespace(paragraphs=None, words=words)
    out = cloud_engine._extract_deepgram_paragraphs(alt, diarize=True)
    assert len(out) == 2
    assert out[0]["speaker"] == 0
    assert out[1]["speaker"] == 1


# ---------------------------------------------------------------------------
# Assembly: offsets + dedup
# ---------------------------------------------------------------------------

def test_assemble_applies_chunk_offsets():
    """Chunks report timestamps relative to their own start; the
    assembler must add the chunk offset so the rendered marker is in
    absolute file time."""
    results = [
        {"paragraphs": [{"text": "Chunk-zero opener.", "start_sec": 5.0, "end_sec": 10.0, "speaker": None}]},
        {"paragraphs": [{"text": "Chunk-one opener.", "start_sec": 0.0, "end_sec": 4.0, "speaker": None}]},
    ]
    rendered = cloud_engine._assemble_with_timestamps(
        results,
        chunk_offsets=[0.0, 600.0],  # second chunk starts 10 min in
        indices=[0, 1],
    )
    assert "[00:00:05]" in rendered
    assert "[00:10:00]" in rendered  # 600 + 0 → 10:00
    assert "Chunk-zero opener." in rendered
    assert "Chunk-one opener." in rendered


def test_assemble_drops_time_overlapping_paragraphs():
    """Adjacent chunks share a 5 s overlap. A paragraph in chunk N+1
    whose absolute start falls inside chunk N's coverage is the
    overlap repeating itself — drop it, otherwise the transcript
    duplicates whatever was said at the seam."""
    results = [
        {"paragraphs": [
            {"text": "Earlier in the seam.", "start_sec": 0.0, "end_sec": 100.0, "speaker": None},
        ]},
        {"paragraphs": [
            # Absolute start = 95 + 0 = 95 → still inside chunk 0's range → drop.
            {"text": "Earlier in the seam.", "start_sec": 0.0, "end_sec": 5.0, "speaker": None},
            # Absolute start = 95 + 10 = 105 → past chunk 0's end → keep.
            {"text": "After the seam.", "start_sec": 10.0, "end_sec": 20.0, "speaker": None},
        ]},
    ]
    rendered = cloud_engine._assemble_with_timestamps(
        results,
        chunk_offsets=[0.0, 95.0],
        indices=[0, 1],
    )
    assert rendered.count("Earlier in the seam.") == 1
    assert "After the seam." in rendered


def test_assemble_renders_speaker_marker_when_present():
    results = [
        {"paragraphs": [
            {"text": "Hello.", "start_sec": 0.0, "end_sec": 1.0, "speaker": 0},
            {"text": "Goodbye.", "start_sec": 1.5, "end_sec": 2.5, "speaker": 1},
        ]},
    ]
    rendered = cloud_engine._assemble_with_timestamps(
        results,
        chunk_offsets=[0.0],
        indices=[0],
    )
    assert "[00:00:00] **Speaker 0:**\nHello." in rendered
    assert "[00:00:01] **Speaker 1:**\nGoodbye." in rendered


def test_assemble_renders_no_speaker_when_absent():
    results = [
        {"paragraphs": [{"text": "Just text.", "start_sec": 0.0, "end_sec": 1.0, "speaker": None}]},
    ]
    rendered = cloud_engine._assemble_with_timestamps(
        results,
        chunk_offsets=[0.0],
        indices=[0],
    )
    assert rendered.startswith("[00:00:00]\nJust text.")
    assert "**Speaker" not in rendered


# ---------------------------------------------------------------------------
# Renderer + stats keep working with markers present
# ---------------------------------------------------------------------------

def test_render_html_styles_timestamp_markers():
    rendered = text_processor.render_transcript_html(
        "[00:01:23] **Speaker 0:**\nHello there."
    )
    assert "<span" in rendered and "[00:01:23]" in rendered
    # Speaker label still bolded by the existing pass.
    assert "<strong>Speaker 0:</strong>" in rendered


def test_reading_stats_strips_timestamps():
    """Otherwise the timestamp tokens inflate word and char counts
    once per paragraph."""
    plain = "Hello world. This is a sentence."
    timestamped = "[00:00:00] **Speaker 0:**\n" + plain
    s_plain = text_processor.get_reading_stats(plain)
    s_ts = text_processor.get_reading_stats(timestamped)
    assert s_ts["word_count"] == s_plain["word_count"]


# ---------------------------------------------------------------------------
# Exporter parser keeps working in all four shapes
# ---------------------------------------------------------------------------

def test_parse_speaker_block_plain_no_timestamp():
    ts, label, content = exporter._parse_speaker_block(
        "Just a paragraph.\nWith two lines."
    )
    assert ts is None
    assert label is None
    assert content == "Just a paragraph.\nWith two lines."


def test_parse_speaker_block_speaker_no_timestamp():
    ts, label, content = exporter._parse_speaker_block(
        "**Speaker 0:**\nHello."
    )
    assert ts is None
    assert label == "Speaker 0:"
    assert content == "Hello."


def test_parse_speaker_block_speaker_with_timestamp():
    ts, label, content = exporter._parse_speaker_block(
        "[00:01:23] **Speaker 0:**\nHello."
    )
    assert ts == "00:01:23"
    assert label == "Speaker 0:"
    assert content == "Hello."


def test_parse_speaker_block_timestamp_only():
    ts, label, content = exporter._parse_speaker_block(
        "[00:02:00]\nNon-diarized content."
    )
    assert ts == "00:02:00"
    assert label is None
    assert content == "Non-diarized content."


# ---------------------------------------------------------------------------
# DOCX/PDF still round-trip when timestamps are present
# ---------------------------------------------------------------------------

def test_export_docx_with_timestamps_produces_bytes():
    text = (
        "[00:00:00] **Speaker 0:**\nHello, this is the start.\n\n"
        "[00:00:30] **Speaker 1:**\nReply at thirty seconds."
    )
    out = exporter.export_docx(text, title="Test")
    assert out.startswith(b"PK")  # DOCX is a zip


def test_export_pdf_with_timestamps_produces_bytes():
    text = (
        "[00:00:00]\nFirst paragraph, no speaker.\n\n"
        "[00:00:20] **Speaker 0:**\nNow with diarization."
    )
    out = exporter.export_pdf(text, title="Test")
    assert out.startswith(b"%PDF")
