"""Download audio from a remote URL (YouTube, podcast feed, direct link).

Wraps yt-dlp so the rest of the pipeline never has to know whether the
audio came from a local upload, a filesystem path, or a URL — by the
time we hand a path off to ``audio_processor`` it's a regular MP3 on
disk. yt-dlp itself shells out to ffmpeg, which the app already
requires at startup.
"""

import logging
import os
import shutil
import tempfile
from typing import Callable, Optional

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

logger = logging.getLogger(__name__)


# We hand-roll a small RuntimeError subclass so the UI layer can match
# on type rather than substring-matching yt-dlp's messages, which change
# between releases.
class URLDownloadError(RuntimeError):
    """Raised when audio cannot be downloaded from the given URL."""


# Heuristic URL check used before we hand the string off to yt-dlp.
# yt-dlp itself accepts a wide variety of URL forms (and even some
# non-URL search keywords), but for this app the input is always a
# direct link. A trivial scheme check is enough to catch typos —
# anything more elaborate would just duplicate yt-dlp's own validation.
def looks_like_url(value: str) -> bool:
    if not value:
        return False
    s = value.strip().lower()
    return s.startswith(("http://", "https://"))


def download_audio_from_url(
    url: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> str:
    """Download the audio track of a URL and return a path to a temp MP3.

    The returned file lives in a private temp directory; callers should
    delete the file (and ideally the parent directory) when done. yt-dlp
    invokes ffmpeg's ``FFmpegExtractAudio`` post-processor under the
    hood, so the output is always an MP3 regardless of the source
    container — matching the format the rest of the pipeline expects.

    ``progress_callback`` receives ``(fraction, message)`` updates while
    the download is in progress. It is invoked from yt-dlp's own progress
    thread, so it must be safe to call from a non-main thread (Streamlit
    placeholders accept this; pure-Python state mutation does too).
    """
    if not looks_like_url(url):
        raise URLDownloadError(
            f"Not a valid URL: {url!r}. Paste a full http(s) link."
        )

    out_dir = tempfile.mkdtemp(prefix="transcriber_url_")
    # Use the source's own ID (yt-dlp's stable identifier) for the
    # filename rather than the title — titles can include slashes,
    # control characters, or non-ASCII glyphs that confuse downstream
    # path handling. The ID is always filesystem-safe.
    out_template = os.path.join(out_dir, "%(id)s.%(ext)s")

    def _hook(d: dict) -> None:
        if not progress_callback:
            return
        status = d.get("status")
        if status == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            downloaded = d.get("downloaded_bytes") or 0
            if total:
                progress_callback(
                    min(downloaded / total, 1.0),
                    f"Downloading audio… {downloaded / 1024 / 1024:.1f} MB",
                )
        elif status == "finished":
            progress_callback(1.0, "Extracting audio…")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",
        }],
        "progress_hooks": [_hook],
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        # yt-dlp prints to stdout/stderr by default; routing through a
        # logger keeps the Streamlit UI clean.
        "logger": logger,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            base = ydl.prepare_filename(info)
    except DownloadError as exc:
        # yt-dlp embeds the offending URL plus the upstream HTTP error
        # in its message. That's useful, but we prefix our own line so
        # the UI's first-line render makes the failure obvious.
        shutil.rmtree(out_dir, ignore_errors=True)
        raise URLDownloadError(f"Could not download from URL: {exc}") from exc
    except Exception as exc:
        # Anything else (network errors, ffmpeg failure inside the
        # post-processor, etc.) should also clean up the temp dir
        # before re-raising as our typed error.
        shutil.rmtree(out_dir, ignore_errors=True)
        raise URLDownloadError(f"Could not download from URL: {exc}") from exc

    # FFmpegExtractAudio rewrites the extension to .mp3 in place; the
    # path returned by prepare_filename is the *pre*-postprocess name.
    mp3_path = os.path.splitext(base)[0] + ".mp3"

    if not os.path.exists(mp3_path):
        # Defensive: if the post-processor failed silently, prepare_filename's
        # original may still exist. Surface a clear error rather than handing
        # an unexpected container to the transcription pipeline.
        shutil.rmtree(out_dir, ignore_errors=True)
        raise URLDownloadError(
            f"yt-dlp produced no MP3 output for {url}. "
            "The source may be DRM-protected or unavailable."
        )

    return mp3_path


def cleanup_url_download(file_path: str) -> None:
    """Remove a downloaded file and the private temp dir it lives in.

    Tolerant of missing files / non-existent directories so the caller
    can safely invoke this in a finally block without extra guards.
    """
    if not file_path:
        return
    parent = os.path.dirname(file_path)
    if os.path.exists(file_path):
        try:
            os.unlink(file_path)
        except OSError as exc:
            logger.warning("Failed to remove URL temp file %s: %s", file_path, exc)
    # Only remove the parent if it's one of our own private dirs — guard
    # against the unlikely case where the caller passed a path outside
    # ``mkdtemp(prefix="transcriber_url_")``.
    if parent and os.path.basename(parent).startswith("transcriber_url_"):
        shutil.rmtree(parent, ignore_errors=True)
