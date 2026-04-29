"""Tests for the URL audio downloader.

The real yt-dlp invocation is mocked out — the network and ffmpeg
post-processor are exercised by yt-dlp's own test suite. These tests
just verify the wrapper's contract: returns a path, surfaces typed
errors, cleans up its temp dir on failure.
"""

import os
import sys
from unittest.mock import patch, MagicMock

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from transcriber import url_source
from transcriber.url_source import (
    URLDownloadError,
    cleanup_url_download,
    download_audio_from_url,
    looks_like_url,
)


# ---------------------------------------------------------------------------
# looks_like_url
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://www.youtube.com/watch?v=abc", True),
        ("http://example.com/episode.mp3", True),
        ("HTTPS://example.com/foo", True),
        ("  https://example.com  ", True),
        # No scheme → reject
        ("example.com/foo", False),
        ("/Users/me/audio.mp3", False),
        # Common typos / non-URL inputs
        ("youtube.com", False),
        ("", False),
        (None, False),
    ],
)
def test_looks_like_url(value, expected):
    assert looks_like_url(value) == expected


# ---------------------------------------------------------------------------
# download_audio_from_url
# ---------------------------------------------------------------------------

def test_invalid_url_raises_typed_error():
    """Garbage input must fail fast with our own exception type, not
    yt-dlp's. The UI matches on type so it can render a friendly error."""
    with pytest.raises(URLDownloadError):
        download_audio_from_url("not-a-url")


def _make_fake_ydl(out_path: str, info_id: str = "video123"):
    """Build a fake YoutubeDL context manager that 'downloads' to ``out_path``.

    Materialises the file on disk inside ``extract_info`` so the
    post-condition check in :func:`download_audio_from_url` (mp3 path
    exists) passes. ``prepare_filename`` returns a non-mp3 extension so
    we exercise the same suffix swap the production code performs.
    """
    instance = MagicMock()
    instance.__enter__.return_value = instance

    def _extract_info(url, download=True):
        # The post-processor renames to .mp3, so write the mp3 directly.
        with open(out_path, "wb") as fh:
            fh.write(b"fake-mp3-bytes")
        return {"id": info_id, "ext": "webm"}

    def _prepare_filename(info):
        # Return the *pre*-postprocess path so the suffix swap kicks in.
        return out_path.replace(".mp3", ".webm")

    instance.extract_info.side_effect = _extract_info
    instance.prepare_filename.side_effect = _prepare_filename
    return instance


def test_successful_download_returns_mp3_path(tmp_path, monkeypatch):
    """Happy path: yt-dlp materialises the file, wrapper returns the mp3 path."""
    captured = {}

    def fake_mkdtemp(prefix=""):
        d = tmp_path / f"{prefix}out"
        d.mkdir()
        captured["dir"] = str(d)
        return str(d)

    monkeypatch.setattr(url_source.tempfile, "mkdtemp", fake_mkdtemp)

    expected_mp3 = None

    def fake_ytdl(opts):
        nonlocal expected_mp3
        # Compute where the post-processor will land its output.
        expected_mp3 = os.path.join(captured["dir"], "video123.mp3")
        return _make_fake_ydl(expected_mp3)

    monkeypatch.setattr(url_source, "YoutubeDL", fake_ytdl)

    result = download_audio_from_url("https://example.com/x")
    assert result == expected_mp3
    assert os.path.exists(result)


def test_download_error_cleans_up_temp_dir(tmp_path, monkeypatch):
    """If yt-dlp raises, the temp dir we created must be removed — leaving
    it behind would leak across the session lifetime on every failed URL."""
    from yt_dlp.utils import DownloadError

    created_dirs = []

    def fake_mkdtemp(prefix=""):
        d = tmp_path / f"{prefix}out{len(created_dirs)}"
        d.mkdir()
        created_dirs.append(str(d))
        return str(d)

    monkeypatch.setattr(url_source.tempfile, "mkdtemp", fake_mkdtemp)

    instance = MagicMock()
    instance.__enter__.return_value = instance
    instance.extract_info.side_effect = DownloadError("HTTP Error 403")

    monkeypatch.setattr(url_source, "YoutubeDL", lambda opts: instance)

    with pytest.raises(URLDownloadError):
        download_audio_from_url("https://example.com/private")

    assert created_dirs, "fake_mkdtemp was never called"
    assert not os.path.exists(created_dirs[0]), (
        "temp dir leaked after download failure"
    )


def test_missing_postprocessor_output_raises(tmp_path, monkeypatch):
    """When yt-dlp 'succeeds' but produces no MP3 (e.g. ffmpeg post-processor
    silently failed), surface a clear error rather than handing the
    original container off to the transcription pipeline."""
    def fake_mkdtemp(prefix=""):
        d = tmp_path / "out"
        d.mkdir()
        return str(d)

    monkeypatch.setattr(url_source.tempfile, "mkdtemp", fake_mkdtemp)

    instance = MagicMock()
    instance.__enter__.return_value = instance
    instance.extract_info.return_value = {"id": "x", "ext": "webm"}
    instance.prepare_filename.return_value = str(tmp_path / "out" / "x.webm")
    # No file actually written.

    monkeypatch.setattr(url_source, "YoutubeDL", lambda opts: instance)

    with pytest.raises(URLDownloadError, match="no MP3 output"):
        download_audio_from_url("https://example.com/x")

    assert not os.path.exists(str(tmp_path / "out")), "temp dir should be cleaned"


# ---------------------------------------------------------------------------
# cleanup_url_download
# ---------------------------------------------------------------------------

def test_cleanup_removes_file_and_private_temp_dir(tmp_path):
    """The cleanup helper must remove both the file and the private temp
    dir we created in ``download_audio_from_url`` — otherwise multiple
    URL fetches in one session leak directories under /tmp."""
    fake_dir = tmp_path / "transcriber_url_abc"
    fake_dir.mkdir()
    fake_file = fake_dir / "video.mp3"
    fake_file.write_bytes(b"x")

    cleanup_url_download(str(fake_file))

    assert not fake_file.exists()
    assert not fake_dir.exists()


def test_cleanup_does_not_touch_unprefixed_dirs(tmp_path):
    """Defensive: if a caller passes a path outside our private temp
    convention, the helper must not rmtree the parent directory."""
    other_dir = tmp_path / "user_documents"
    other_dir.mkdir()
    other_file = other_dir / "audio.mp3"
    other_file.write_bytes(b"x")

    cleanup_url_download(str(other_file))

    assert not other_file.exists(), "the file itself should still be removed"
    assert other_dir.exists(), "non-private parent must be left alone"


def test_cleanup_tolerates_missing_paths(tmp_path):
    """Calling cleanup on a file that's already gone must be a no-op so
    a finally-block invocation never raises."""
    cleanup_url_download(str(tmp_path / "nonexistent.mp3"))
    cleanup_url_download("")  # explicit empty path
