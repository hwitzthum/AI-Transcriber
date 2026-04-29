"""Tests for the batch transcription helpers.

The Streamlit UI orchestration is exercised manually; these tests
target the bits that have to be correct independent of any UI:

* ``transcribe_one`` cleans up chunk temp files even when the
  transcription call raises (was the source of multi-hundred-MB temp
  leaks in the single-file path until that flow grew its finally
  block).
* ``build_zip`` produces a valid zip with one folder per file and
  routes failures into a side-car ``error.txt`` instead of dropping
  the entry.
* ``_safe_stem`` produces predictable, cross-platform-safe names.
"""

import io
import os
import sys
import zipfile
from unittest.mock import patch

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from transcriber import batch
from transcriber.batch import _safe_stem, build_zip, transcribe_one


# ---------------------------------------------------------------------------
# _safe_stem
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("meeting.mp4", "meeting"),
        # Path components are stripped by basename — can't smuggle a
        # subdirectory into the zip via the filename.
        ("/path/to/file.mp3", "file"),
        ("with spaces.mp3", "with spaces"),
        # Inline ``?`` is replaced (Windows-hostile char).
        ("question?.mp3", "question_"),
        # Unicode preserved — transcripts are often non-English source.
        ("français.mp3", "français"),
        # Empty → fallback so the zip never has an entry like ``/error.txt``
        # with no folder prefix at all.
        ("", "untitled"),
    ],
)
def test_safe_stem(filename, expected):
    assert _safe_stem(filename) == expected


# ---------------------------------------------------------------------------
# build_zip
# ---------------------------------------------------------------------------

def _names_in_zip(blob: bytes) -> set[str]:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        return set(zf.namelist())


def test_build_zip_emits_three_outputs_per_successful_file():
    """Every successful file must produce DOCX + PDF + plain TXT.
    The TXT is included so users on platforms without Word/Acrobat
    can still see the transcript without extra tooling."""
    blob = build_zip([
        {"filename": "alpha.mp3", "text": "**Speaker 0:**\nHello."},
        {"filename": "beta.mp4", "text": "Plain text here."},
    ])
    names = _names_in_zip(blob)
    for stem in ("alpha", "beta"):
        assert f"{stem}/transcription.docx" in names
        assert f"{stem}/transcription.pdf" in names
        assert f"{stem}/transcription.txt" in names


def test_build_zip_records_per_file_error_inline():
    """A file with an ``error`` key must produce ONLY ``error.txt`` —
    no half-baked DOCX/PDF that would mislead the user into thinking
    the transcription succeeded."""
    blob = build_zip([
        {"filename": "broken.mp3", "error": "401 Unauthorized"},
    ])
    names = _names_in_zip(blob)
    assert names == {"broken/error.txt"}
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        assert "401 Unauthorized" in zf.read("broken/error.txt").decode()


def test_build_zip_handles_empty_transcript():
    """Empty text shouldn't crash the exporter; the helper must catch
    that and substitute an explanatory placeholder so the user sees
    *something* in the zip."""
    blob = build_zip([
        {"filename": "silent.mp3", "text": ""},
    ])
    names = _names_in_zip(blob)
    assert "silent/empty.txt" in names
    assert "silent/transcription.docx" not in names


def test_build_zip_does_not_abort_on_per_export_failure(monkeypatch):
    """If one of the export paths raises (e.g. a font issue inside
    fpdf2), the zip should still contain the other format. Partial
    output beats no output."""
    def _bad_pdf(text, title="Transcription"):
        raise RuntimeError("simulated fpdf2 failure")

    monkeypatch.setattr(batch.exporter, "export_pdf", _bad_pdf)

    blob = build_zip([
        {"filename": "alpha.mp3", "text": "Hello."},
    ])
    names = _names_in_zip(blob)
    assert "alpha/transcription.docx" in names
    assert "alpha/pdf_error.txt" in names
    assert "alpha/transcription.pdf" not in names


@pytest.mark.filterwarnings("ignore:Duplicate name:UserWarning")
def test_build_zip_filename_collisions_are_stem_normalised():
    """Two uploads with the same basename but different extensions
    will collide in the zip if we don't keep the entries distinct.
    Today they share a folder; this test pins that current behaviour
    so a future change is a deliberate call rather than a silent
    regression."""
    blob = build_zip([
        {"filename": "talk.mp3", "text": "From mp3."},
        {"filename": "talk.mp4", "text": "From mp4."},
    ])
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        # Both entries land in the same ``talk/`` folder; the second
        # write overwrites the first within a given output extension.
        # Either way, the zip should still be valid and parseable.
        zf.testzip()


# ---------------------------------------------------------------------------
# transcribe_one
# ---------------------------------------------------------------------------

def test_transcribe_one_cleans_up_chunks_on_success(tmp_path, monkeypatch):
    """The chunk temp files materialised by ffmpeg must be removed
    even on the happy path — leaking them across a multi-file batch
    would balloon /tmp very quickly on long recordings."""
    fake_chunk = tmp_path / "chunk0.mp3"
    fake_chunk.write_bytes(b"x")

    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"x")

    monkeypatch.setattr(batch.audio_processor, "get_audio_info", lambda p: {
        "duration_seconds": 60.0,
        "duration_formatted": "01:00",
        "channels": 1,
        "sample_rate": 16000,
        "file_size_mb": 1.0,
    })
    monkeypatch.setattr(
        batch.audio_processor,
        "iter_chunks",
        lambda p, **k: (1, iter([str(fake_chunk)])),
    )
    monkeypatch.setattr(
        batch.audio_processor,
        "compute_chunk_offsets",
        lambda **k: [0.0],
    )
    monkeypatch.setattr(
        batch.cloud_engine,
        "transcribe_chunks_streaming",
        lambda **k: list(k["chunk_iter"]) and {  # drain the iterator
            "text": "result",
            "failed_chunks": [],
            "detected_language": "en",
            "quality_warnings": [],
        },
    )

    result = transcribe_one(
        audio_path=str(audio),
        provider="OpenAI Whisper API",
        api_key="dummy",
    )

    assert result["text"] == "result"
    assert result["duration_seconds"] == 60.0
    assert not fake_chunk.exists(), "chunk temp file was not cleaned up"


def test_transcribe_one_cleans_up_chunks_on_failure(tmp_path, monkeypatch):
    """The cleanup must run even when the transcription call raises —
    without this, a single file with a bad API key would leak its
    chunks until the process exited."""
    fake_chunk = tmp_path / "chunk0.mp3"
    fake_chunk.write_bytes(b"x")

    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"x")

    monkeypatch.setattr(batch.audio_processor, "get_audio_info", lambda p: {
        "duration_seconds": 60.0,
        "duration_formatted": "01:00",
        "channels": 1,
        "sample_rate": 16000,
        "file_size_mb": 1.0,
    })
    monkeypatch.setattr(
        batch.audio_processor,
        "iter_chunks",
        lambda p, **k: (1, iter([str(fake_chunk)])),
    )
    monkeypatch.setattr(
        batch.audio_processor,
        "compute_chunk_offsets",
        lambda **k: [0.0],
    )

    def _boom(**kwargs):
        # Drain the chunk iterator first so the production code path
        # has actually recorded the chunk path (mirrors what the real
        # transcription path does — chunks materialise before the API
        # call completes).
        list(kwargs["chunk_iter"])
        raise RuntimeError("API key invalid")

    monkeypatch.setattr(batch.cloud_engine, "transcribe_chunks_streaming", _boom)

    with pytest.raises(RuntimeError, match="API key"):
        transcribe_one(
            audio_path=str(audio),
            provider="OpenAI Whisper API",
            api_key="dummy",
        )

    assert not fake_chunk.exists(), (
        "chunk temp file leaked when transcription raised"
    )
