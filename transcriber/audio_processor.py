"""Audio processing: format detection, conversion, and chunking for large files."""

import logging
import os
import subprocess
import tempfile
import math
from pathlib import Path
from pydub import AudioSegment

# Module-level logger
logger = logging.getLogger(__name__)


# Maximum chunk size in bytes (~24 MB to stay under 25 MB API limits)
MAX_CHUNK_BYTES = 24 * 1024 * 1024

# Overlap in milliseconds to avoid cutting words at chunk boundaries.
# 5 seconds gives the transcript deduplication enough material to find the
# join point without excessive repetition.
OVERLAP_MS = 5000

# Files larger than this threshold (in MB) trigger the ffmpeg-direct chunking
# path to avoid loading the entire file into RAM with pydub.
# At 128 kbps MP3, 200 MB is roughly 3+ hours of audio.
_LARGE_FILE_THRESHOLD_MB = 200

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

    Uses ffprobe to read only the container/stream headers without decoding
    the entire file into RAM. This is essential for multi-hour recordings
    where pydub would allocate gigabytes of memory just to read metadata.

    Falls back to pydub if ffprobe is not available.
    """
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    # Try ffprobe first (O(1) memory, reads only headers)
    if _ffprobe_available():
        try:
            info = _get_audio_info_ffprobe(file_path)
            info["file_size_mb"] = file_size_mb
            return info
        except Exception as e:
            logger.warning("ffprobe failed, falling back to pydub: %s", e)

    # Fallback: pydub (loads entire file into RAM)
    audio = AudioSegment.from_file(file_path)
    return {
        "duration_seconds": len(audio) / 1000.0,
        "duration_formatted": _format_duration(len(audio) / 1000.0),
        "channels": audio.channels,
        "sample_rate": audio.frame_rate,
        "file_size_mb": file_size_mb,
    }


def _ffprobe_available() -> bool:
    """Check if ffprobe is available on PATH."""
    try:
        subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _get_audio_info_ffprobe(file_path: str) -> dict:
    """Extract audio metadata using ffprobe (reads only headers, O(1) memory).

    Returns:
        dict with duration_seconds, duration_formatted, channels, sample_rate
    """
    import json

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


def needs_chunking(file_path: str, max_bytes: int = MAX_CHUNK_BYTES) -> bool:
    """Check if the file exceeds the chunk size limit for the chosen provider."""
    return os.path.getsize(file_path) > max_bytes


def chunk_audio(
    file_path: str,
    progress_callback=None,
    max_bytes: int = MAX_CHUNK_BYTES,
) -> list[str]:
    """
    Split a large audio file into smaller chunks suitable for API upload.

    The ``max_bytes`` parameter lets each provider use its own upload ceiling:
    OpenAI/Groq cap at ~24 MB while Deepgram accepts a single chunk up to
    500 MB. Files at or below the limit are returned as a single-element list
    (after being transcoded to MP3 if needed) so the API receives one upload
    instead of many small chunks needlessly stitched back together.

    For very large files (> _LARGE_FILE_THRESHOLD_MB) the function uses
    ffmpeg directly to slice audio without loading the entire file into
    Python memory. For smaller files pydub is used.

    Returns a list of temporary MP3 file paths. The caller is responsible for
    cleaning them up via cleanup_chunks().
    """
    file_size = os.path.getsize(file_path)
    file_size_mb = file_size / (1024 * 1024)
    is_video = Path(file_path).suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    # --- Path 0: File already fits in one upload — skip chunking entirely. ---
    # This is the dominant path for Deepgram and saves the overlap/dedup
    # round-trip on files that the provider would happily ingest whole.
    if file_size <= max_bytes and not is_video:
        return [_ensure_mp3(file_path)]

    # --- Path 1: Very large files — stream directly through ffmpeg ---
    # This avoids loading gigabytes of audio into RAM and is essential for
    # multi-hour German/French conference recordings.
    if file_size_mb > _LARGE_FILE_THRESHOLD_MB and _ffmpeg_available():
        logger.info(
            "File is %.0f MB (> %d MB threshold). Using ffmpeg streaming path.",
            file_size_mb,
            _LARGE_FILE_THRESHOLD_MB,
        )
        return _chunk_with_ffmpeg(file_path, progress_callback, max_bytes=max_bytes)

    # --- Path 2: Medium-large video — extract audio first ---
    # For videos between 50 MB and the large threshold, extract audio track
    # before chunking so pydub doesn't decode video frames into RAM.
    temp_audio_path = None
    processed_path = file_path

    if is_video and file_size_mb > 50:
        try:
            temp_audio_path = _extract_audio_from_video(file_path)
            processed_path = temp_audio_path
            file_size = os.path.getsize(processed_path)
        except Exception as exc:
            logger.warning("Could not extract audio from video, falling back: %s", exc)

    # If file fits in one chunk (after possible audio extraction), skip splitting
    if file_size <= max_bytes:
        if temp_audio_path:
            return [temp_audio_path]
        return [_ensure_mp3(file_path)]

    # --- Path 3: Standard pydub chunking for smaller files ---
    audio = AudioSegment.from_file(processed_path)
    total_duration_ms = len(audio)

    # Estimate chunk duration to stay under max_bytes at 128 kbps
    bitrate_kbps = 128
    bytes_per_ms = (bitrate_kbps * 1000) / 8 / 1000
    chunk_duration_ms = int(max_bytes / bytes_per_ms)

    # Safety: chunk duration must be larger than overlap
    chunk_duration_ms = max(chunk_duration_ms, OVERLAP_MS + 1000)

    step_ms = chunk_duration_ms - OVERLAP_MS
    num_chunks = max(1, math.ceil(total_duration_ms / step_ms))
    chunk_paths = []

    for i in range(num_chunks):
        start_ms = i * step_ms
        end_ms = min(start_ms + chunk_duration_ms, total_duration_ms)

        chunk = audio[start_ms:end_ms]

        with tempfile.NamedTemporaryFile(
            suffix=f"_chunk_{i:03d}.mp3",
            delete=False,
            dir=tempfile.gettempdir(),
        ) as tmp:
            chunk.export(tmp.name, format="mp3", bitrate="128k")
            chunk_paths.append(tmp.name)

        if progress_callback:
            progress_callback(i + 1, num_chunks, f"Splitting audio: chunk {i + 1}/{num_chunks}")

    # Clean up the intermediate audio extract if we created one
    if temp_audio_path and os.path.exists(temp_audio_path):
        try:
            os.unlink(temp_audio_path)
        except OSError as exc:
            logger.warning("Failed to cleanup temp audio file %s: %s", temp_audio_path, exc)

    return chunk_paths


# ---------------------------------------------------------------------------
# ffmpeg-based streaming chunker (memory-efficient path for very large files)
# ---------------------------------------------------------------------------

def _ffmpeg_available() -> bool:
    """Return True if ffmpeg is available on PATH."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


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
        raise RuntimeError(
            f"ffprobe could not determine duration of {file_path}: {result.stderr.strip()}"
        )


def _chunk_with_ffmpeg(
    file_path: str,
    progress_callback=None,
    max_bytes: int = MAX_CHUNK_BYTES,
) -> list[str]:
    """
    Slice a file into MP3 chunks using ffmpeg without loading it into RAM.

    ffmpeg reads and encodes the audio stream sequentially, so peak RAM usage
    is proportional to one chunk's worth of buffered audio — typically a few
    MB — rather than the full file size.

    Each chunk overlaps with the next by OVERLAP_MS milliseconds. The cloud
    engine's deduplication logic removes the repeated words after transcription.
    """
    total_seconds = _get_duration_seconds(file_path)

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

    for i in range(num_chunks):
        start_sec = i * step_sec
        # Do not exceed file duration
        actual_duration_sec = min(chunk_duration_sec, total_seconds - start_sec)
        if actual_duration_sec <= 0:
            break

        with tempfile.NamedTemporaryFile(
            suffix=f"_chunk_{i:03d}.mp3",
            delete=False,
            dir=tempfile.gettempdir(),
        ) as tmp:
            out_path = tmp.name

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

        chunk_paths.append(out_path)

        if progress_callback:
            progress_callback(i + 1, num_chunks, f"Splitting audio: chunk {i + 1}/{num_chunks}")

    return chunk_paths


# ---------------------------------------------------------------------------
# pydub helpers
# ---------------------------------------------------------------------------

def _extract_audio_from_video(video_path: str) -> str:
    """Extract audio track from video to a temporary MP3 file using pydub/ffmpeg."""
    audio = AudioSegment.from_file(video_path, format=Path(video_path).suffix[1:])

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, dir=tempfile.gettempdir()) as tmp:
        audio.export(tmp.name, format="mp3", bitrate="128k")
        return tmp.name


def _ensure_mp3(file_path: str) -> str:
    """Convert file to MP3 if it is not already. Returns path to the MP3 file."""
    if file_path.lower().endswith(".mp3"):
        return file_path

    audio = AudioSegment.from_file(file_path)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, dir=tempfile.gettempdir()) as tmp:
        audio.export(tmp.name, format="mp3", bitrate="128k")
        return tmp.name


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