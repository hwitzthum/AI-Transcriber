"""Audio processing: format detection, conversion, and chunking for large files.

ffmpeg/ffprobe are hard requirements — see README for installation. The
module checks for them at startup via :func:`require_ffmpeg` and refuses
to do real work if they are missing, with a clear install hint.
"""

import hashlib
import json
import logging
import math
import os
import subprocess
import tempfile
from pathlib import Path

# Module-level logger
logger = logging.getLogger(__name__)


# Maximum chunk size in bytes (~24 MB to stay under 25 MB API limits)
MAX_CHUNK_BYTES = 24 * 1024 * 1024

# Overlap in milliseconds to avoid cutting words at chunk boundaries.
# 5 seconds gives the transcript deduplication enough material to find the
# join point without excessive repetition.
OVERLAP_MS = 5000

SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma", ".aac",
    ".opus", ".webm", ".mp4", ".mov", ".avi", ".mkv",
}

# Sensitive filesystem locations the user-facing path input must never resolve
# into. The check is post-realpath, so symlink traversal can't bypass it.
# Why a deny-list rather than an allow-list: legitimate audio files live in
# many places on a personal machine (~/Documents, /Volumes/<drive>, /tmp).
# An allow-list would force users to configure roots; a deny-list blocks the
# obvious exfiltration targets while keeping the local-use ergonomics.
_DENIED_PATH_PREFIXES = (
    "/etc",
    "/private/etc",
    "/root",
    "/var/root",
    "/sys",
    "/proc",
    "/private/var/db",
    "/private/var/root",
)
_DENIED_PATH_SUBSTRINGS = (
    "/.ssh/",
    "/.aws/",
    "/.gnupg/",
    "/.config/gh/",
    "/.docker/",
    "/.kube/",
)


def validate_file(file_path: str) -> tuple[bool, str]:
    """Validate that a file exists and is a supported audio format.

    Resolves symlinks before checking, then rejects paths that point at
    sensitive system locations or user secret directories. This is a
    defense-in-depth measure for the "Or Enter Path" UI input — without it,
    a user-supplied path would be passed straight to ffprobe/ffmpeg and any
    file the process could read would be uploaded to the transcription API.
    """
    path = Path(file_path)
    if not path.exists():
        return False, f"File not found: {file_path}"
    if not path.is_file():
        return False, f"Not a file: {file_path}"

    # Canonicalize: follow symlinks and resolve "..", so a path like
    # "/Users/x/audio/../../../etc/passwd" cannot sneak past the deny-list.
    try:
        resolved = str(path.resolve(strict=True))
    except (OSError, RuntimeError) as exc:
        return False, f"Could not resolve path: {exc}"

    for prefix in _DENIED_PATH_PREFIXES:
        if resolved == prefix or resolved.startswith(prefix + "/"):
            return False, "Access denied: path resolves to a protected location."
    for needle in _DENIED_PATH_SUBSTRINGS:
        if needle in resolved:
            return False, "Access denied: path resolves to a protected location."

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False, (
            f"Unsupported format: {path.suffix}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return True, "OK"


def get_audio_info(file_path: str) -> dict:
    """Get audio file metadata using ffprobe (memory-efficient).

    Reads only the container/stream headers; never decodes the file. This
    is essential for multi-hour recordings where any in-RAM decoder would
    allocate gigabytes just to report duration and sample rate.
    """
    require_ffmpeg()
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    info = _get_audio_info_ffprobe(file_path)
    info["file_size_mb"] = file_size_mb
    return info


_FFPROBE_AVAILABLE: bool | None = None
_FFMPEG_AVAILABLE: bool | None = None


def require_ffmpeg() -> None:
    """Raise a RuntimeError with install hints if ffmpeg/ffprobe are missing.

    Both binaries are hard prerequisites for transcoding, chunking, and
    reading metadata. The earlier pydub fallback path was a partial
    illusion — pydub itself shells out to ffmpeg for every format this
    app supports — so a single, explicit gate produces a clearer error.
    """
    if not _ffmpeg_available() or not _ffprobe_available():
        raise RuntimeError(
            "ffmpeg and ffprobe are required but were not found on PATH. "
            "Install with `brew install ffmpeg` (macOS) or "
            "`sudo apt install ffmpeg` (Debian/Ubuntu)."
        )


def _ffprobe_available() -> bool:
    """Check if ffprobe is available on PATH (memoised — the subprocess
    spawn used to add ~30–50 ms per call, and it's called several times
    per upload)."""
    global _FFPROBE_AVAILABLE
    if _FFPROBE_AVAILABLE is None:
        try:
            subprocess.run(
                ["ffprobe", "-version"],
                capture_output=True,
                check=True,
            )
            _FFPROBE_AVAILABLE = True
        except (FileNotFoundError, subprocess.CalledProcessError):
            _FFPROBE_AVAILABLE = False
    return _FFPROBE_AVAILABLE


def _get_audio_info_ffprobe(file_path: str) -> dict:
    """Extract audio metadata using ffprobe (reads only headers, O(1) memory).

    Returns:
        dict with duration_seconds, duration_formatted, channels, sample_rate
    """
    # Use ffprobe to get stream info in JSON format
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            "-select_streams", "a:0",  # First audio stream only
            file_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    data = json.loads(result.stdout)

    # Extract duration from format (container-level) or stream
    duration_seconds = 0.0
    if "format" in data and "duration" in data["format"]:
        duration_seconds = float(data["format"]["duration"])
    elif data.get("streams") and "duration" in data["streams"][0]:
        duration_seconds = float(data["streams"][0]["duration"])

    # Extract audio stream properties
    channels = 2  # default
    sample_rate = 44100  # default

    if data.get("streams"):
        stream = data["streams"][0]
        channels = stream.get("channels", 2)
        sample_rate = int(stream.get("sample_rate", 44100))

    return {
        "duration_seconds": duration_seconds,
        "duration_formatted": _format_duration(duration_seconds),
        "channels": channels,
        "sample_rate": sample_rate,
    }


def compute_upload_hash(file_buffer) -> str:
    """Fingerprint an upload buffer without copying its bytes.

    Uses ``getbuffer()`` (a zero-copy memoryview) instead of ``getvalue()``
    (which materialises the entire upload into a fresh bytes object) so a
    multi-hundred-megabyte file doesn't allocate a fresh copy on every
    Streamlit rerun. Hashes head + tail + size — same fingerprint as
    before, the change is purely about memory.
    """
    hasher = hashlib.md5()
    buf = file_buffer.getbuffer()
    size = len(buf)
    # Slicing a memoryview is O(1); .update accepts buffer protocol directly.
    hasher.update(buf[:65536])
    hasher.update(buf[-65536:])
    hasher.update(str(size).encode())
    return hasher.hexdigest()


def needs_chunking(file_path: str, max_bytes: int = MAX_CHUNK_BYTES) -> bool:
    """Check if the file exceeds the chunk size limit for the chosen provider."""
    return os.path.getsize(file_path) > max_bytes


def chunk_audio(
    file_path: str,
    progress_callback=None,
    max_bytes: int = MAX_CHUNK_BYTES,
    duration_seconds: float | None = None,
) -> list[str]:
    """
    Prepare an audio/video file for upload to a transcription API.

    Files at or below ``max_bytes`` are returned as a single-element list
    (transcoded to MP3 by ffmpeg if not already MP3); larger files are
    sliced into ``max_bytes``-sized MP3 chunks streamed through ffmpeg
    without ever loading the source into RAM.

    Returns a list of file paths. Callers must clean up the returned files
    via :func:`cleanup_chunks` — note the original path is returned
    unchanged when the input is already an MP3 below the size threshold,
    in which case ``cleanup_chunks`` will skip it.

    ``duration_seconds`` is an optional optimisation: callers that already
    have the audio's duration (from a prior :func:`get_audio_info` call)
    can pass it in to skip a second ffprobe subprocess spawn inside the
    chunker. Omitting it preserves the prior behaviour.
    """
    require_ffmpeg()

    file_size = os.path.getsize(file_path)

    # File already fits in one upload — skip chunking, just transcode to MP3
    # if needed. This is the dominant path for Deepgram (500 MB ceiling).
    if file_size <= max_bytes:
        return [_ensure_mp3(file_path)]

    # Larger than the upload ceiling — slice into chunks via ffmpeg. The
    # streaming path keeps RAM usage proportional to one chunk regardless
    # of source size, so multi-hour recordings just work.
    return _chunk_with_ffmpeg(
        file_path,
        progress_callback,
        max_bytes=max_bytes,
        duration_seconds=duration_seconds,
    )


# ---------------------------------------------------------------------------
# ffmpeg-based streaming chunker (memory-efficient path for very large files)
# ---------------------------------------------------------------------------

def _ffmpeg_available() -> bool:
    """Return True if ffmpeg is available on PATH (memoised — see ffprobe)."""
    global _FFMPEG_AVAILABLE
    if _FFMPEG_AVAILABLE is None:
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            _FFMPEG_AVAILABLE = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            _FFMPEG_AVAILABLE = False
    return _FFMPEG_AVAILABLE


def _get_duration_seconds(file_path: str) -> float:
    """
    Use ffprobe to get audio duration without decoding the entire file.

    This is O(1) in memory even for very large files.
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        # ffprobe stderr can echo the resolved input path or, on some
        # builds, codec-search paths from the environment. Cap the snippet
        # before surfacing it to the UI and drop ``file_path`` from the
        # message — the path is already in the logger call (with full
        # context) so it isn't lost.
        stderr_snippet = (result.stderr or "").strip()[:200]
        logger.error("ffprobe failed for %s: %s", file_path, stderr_snippet)
        raise RuntimeError(
            f"ffprobe could not determine audio duration: {stderr_snippet}"
        )


def _chunk_with_ffmpeg(
    file_path: str,
    progress_callback=None,
    max_bytes: int = MAX_CHUNK_BYTES,
    duration_seconds: float | None = None,
) -> list[str]:
    """
    Slice a file into MP3 chunks using ffmpeg without loading it into RAM.

    ffmpeg reads and encodes the audio stream sequentially, so peak RAM usage
    is proportional to one chunk's worth of buffered audio — typically a few
    MB — rather than the full file size.

    Each chunk overlaps with the next by OVERLAP_MS milliseconds. The cloud
    engine's deduplication logic removes the repeated words after transcription.

    Callers that already know the duration (from a prior ``get_audio_info``
    call) can pass it in via ``duration_seconds`` to avoid a duplicate
    ffprobe subprocess spawn here.
    """
    total_seconds = (
        duration_seconds
        if duration_seconds is not None
        else _get_duration_seconds(file_path)
    )

    # Calculate chunk and step duration to stay under max_bytes at 128 kbps
    bitrate_kbps = 128
    bytes_per_second = (bitrate_kbps * 1000) / 8
    chunk_duration_sec = max_bytes / bytes_per_second
    overlap_sec = OVERLAP_MS / 1000.0
    step_sec = chunk_duration_sec - overlap_sec

    # Safety guard: chunk must be longer than overlap
    chunk_duration_sec = max(chunk_duration_sec, overlap_sec + 1.0)
    step_sec = chunk_duration_sec - overlap_sec

    num_chunks = max(1, math.ceil(total_seconds / step_sec))
    chunk_paths: list[str] = []

    logger.info(
        "ffmpeg chunking: %.0f s total, %.0f s chunks, %d chunks",
        total_seconds,
        chunk_duration_sec,
        num_chunks,
    )

    # Tail chunks shorter than this produce ffmpeg outputs of a few
    # milliseconds — the API transcribes them as silence and the quality
    # warning fires on every multi-chunk run. Half a second is well below
    # any meaningful speech segment.
    MIN_CHUNK_SEC = 0.5

    for i in range(num_chunks):
        start_sec = i * step_sec
        # Do not exceed file duration
        actual_duration_sec = min(chunk_duration_sec, total_seconds - start_sec)
        if actual_duration_sec < MIN_CHUNK_SEC:
            break

        with tempfile.NamedTemporaryFile(
            suffix=f"_chunk_{i:03d}.mp3",
            delete=False,
            dir=tempfile.gettempdir(),
        ) as tmp:
            out_path = tmp.name

        # Track the temp file BEFORE running ffmpeg so the caller's
        # ``cleanup_chunks`` finally handler can remove it even when
        # this chunk is the one that fails. Previously the append
        # happened only on success, leaking the failing chunk's file.
        chunk_paths.append(out_path)

        cmd = [
            "ffmpeg",
            "-y",                        # overwrite output without asking
            "-ss", str(start_sec),       # seek to start (fast, keyframe-accurate)
            "-i", file_path,             # input
            "-t", str(actual_duration_sec),  # duration
            "-vn",                       # no video stream
            "-acodec", "libmp3lame",
            "-ab", "128k",
            "-ar", "16000",              # 16 kHz is sufficient for speech recognition
            "-ac", "1",                  # mono — halves file size with no quality loss
            out_path,
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=300,
            )
        except subprocess.CalledProcessError as exc:
            stderr_snippet = (exc.stderr or b"").decode("utf-8", errors="replace")[-500:]
            raise RuntimeError(
                f"ffmpeg failed on chunk {i + 1}: {stderr_snippet}"
            ) from exc

        if progress_callback:
            progress_callback(i + 1, num_chunks, f"Splitting audio: chunk {i + 1}/{num_chunks}")

    return chunk_paths


# ---------------------------------------------------------------------------
# Single-file transcoding helper
# ---------------------------------------------------------------------------

def _ensure_mp3(file_path: str) -> str:
    """Transcode any supported audio/video file to MP3 via ffmpeg.

    Returns the original path unchanged when the input is already MP3,
    avoiding a needless re-encode of files that providers can ingest
    directly. Otherwise produces a temporary 128 kbps mono 16 kHz MP3
    optimised for speech transcription, drops any video stream with
    ``-vn``, and returns the output path.
    """
    if file_path.lower().endswith(".mp3"):
        return file_path

    require_ffmpeg()

    with tempfile.NamedTemporaryFile(
        suffix=".mp3", delete=False, dir=tempfile.gettempdir()
    ) as tmp:
        out_path = tmp.name

    cmd = [
        "ffmpeg",
        "-y",
        "-i", file_path,
        "-vn",                      # drop any video stream
        "-acodec", "libmp3lame",
        "-ab", "128k",
        "-ar", "16000",             # 16 kHz mono is plenty for speech recognition
        "-ac", "1",
        out_path,
    ]
    # 5 minutes is the same ceiling used by the chunking path. Without
    # any timeout, a corrupted input or a stalled filesystem could hang
    # the Streamlit thread indefinitely with no diagnostic.
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        if os.path.exists(out_path):
            try:
                os.unlink(out_path)
            except OSError:
                pass
        raise RuntimeError(
            "ffmpeg transcode timed out after 300 s — the input file may be corrupted."
        ) from exc

    if result.returncode != 0:
        if os.path.exists(out_path):
            try:
                os.unlink(out_path)
            except OSError:
                pass
        raise RuntimeError(
            f"ffmpeg failed to transcode {file_path!r}: {result.stderr.strip()[:500]}"
        )
    return out_path


def cleanup_chunks(chunk_paths: list[str], original_path: str):
    """Remove temporary chunk files, skipping the original input file."""
    for path in chunk_paths:
        if path != original_path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError as exc:
                logger.warning("Failed to cleanup chunk file %s: %s", path, exc)


def _format_duration(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"