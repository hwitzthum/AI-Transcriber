"""Cloud transcription engine using OpenAI Whisper API, Groq API, or Deepgram."""

import logging
import re
import time
from openai import OpenAI
from groq import Groq
from deepgram import DeepgramClient
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom retry predicate: only retry on transient/retriable errors
# ---------------------------------------------------------------------------

# HTTP status codes that are retriable (transient server errors)
_RETRIABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class RetriableAPIError(Exception):
    """Raised when an API call fails with a retriable error (429, 5xx)."""
    pass


class NonRetriableAPIError(Exception):
    """Raised when an API call fails with a non-retriable error (400, 401, 403, etc.)."""
    pass


# Patterns for things that look like an API key or bearer token. Provider
# SDK exceptions sometimes embed the failing request in their string repr,
# which can include the Authorization header — so any exception we surface
# to the UI must run through this redactor first.
_SECRET_PATTERNS = (
    re.compile(r"Bearer\s+\S+", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}\b"),
    re.compile(r"\b(?:gsk|dg|ds|tk)_[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r'(?i)(?:api[_-]?key|authorization|token)["\']?\s*[:=]\s*["\']?([A-Za-z0-9_\-\.]{16,})'),
)


def redact_secrets(message: str) -> str:
    """Strip anything that looks like a credential from a string.

    Provider SDK exceptions occasionally embed the outgoing Authorization
    header in their repr (we've observed this on malformed-key 401s from
    the OpenAI client). Anything routed to ``st.error``/``st.warning`` —
    and therefore visible on a user's screen — must run through here so
    that a screenshot of an error doesn't leak the user's key.
    """
    if not message:
        return message
    redacted = message
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _is_retriable_error(exc: BaseException) -> bool:
    """
    Determine if an exception is retriable based on HTTP status code.

    Only retries on:
    - 429 Too Many Requests (rate limit)
    - 500 Internal Server Error
    - 502 Bad Gateway
    - 503 Service Unavailable
    - 504 Gateway Timeout
    - Connection/timeout errors

    Does NOT retry on:
    - 400 Bad Request (invalid audio, oversized file)
    - 401 Unauthorized (invalid API key)
    - 403 Forbidden (access denied)
    - 404 Not Found
    """
    exc_str_raw = str(exc)
    exc_str = exc_str_raw.lower()

    # Non-retriable status codes are checked FIRST, with word boundaries, so
    # that an embedded substring like "400" inside a larger number (e.g.
    # "14002") never falsely classifies as retriable. Use \b to anchor on
    # whole numeric tokens.
    _NON_RETRIABLE_CODES = (400, 401, 403, 404)
    for code in _NON_RETRIABLE_CODES:
        if re.search(rf"\b{code}\b", exc_str_raw):
            return False

    # Retriable status codes — also word-boundary anchored to avoid embedded matches.
    for code in _RETRIABLE_STATUS_CODES:
        if re.search(rf"\b{code}\b", exc_str_raw):
            return True

    # Check for common retriable error patterns
    retriable_patterns = [
        "rate limit",
        "too many requests",
        "service unavailable",
        "temporarily unavailable",
        "connection reset",
        "connection refused",
        "connection timeout",
        "read timeout",
        "socket timeout",
        "network unreachable",
        "temporary failure",
        "server error",
        "internal server error",
        "bad gateway",
        "gateway timeout",
    ]

    if any(pattern in exc_str for pattern in retriable_patterns):
        return True

    # Check for non-retriable text patterns (the numeric codes are already handled above).
    non_retriable_patterns = [
        "unauthorized",
        "invalid api key",
        "authentication",
        "bad request",
        "invalid audio",
        "unsupported format",
        "file too large",
        "forbidden",
        "access denied",
        "not found",
    ]

    if any(pattern in exc_str for pattern in non_retriable_patterns):
        return False

    # For RetriableAPIError, always retry
    if isinstance(exc, RetriableAPIError):
        return True

    # For NonRetriableAPIError, never retry
    if isinstance(exc, NonRetriableAPIError):
        return False

    # Default: retry on unknown errors (conservative approach for network issues)
    # but only if it looks like a connection/transport issue
    if "timeout" in exc_str or "connection" in exc_str:
        return True

    # Don't retry on other unknown errors
    return False


# OpenAI and Groq cap upload size at ~25 MB; Deepgram accepts up to ~2 GB
# but we cap our single-chunk path at 500 MB so a network blip during a
# very long upload only forces *one* chunk worth of re-transmission rather
# than the whole file. 500 MB is roughly 9 hours of 128 kbps mono speech,
# which covers nearly every real-world meeting in a single chunk.
_OPENAI_GROQ_MAX_CHUNK_BYTES = 24 * 1024 * 1024
_DEEPGRAM_MAX_CHUNK_BYTES = 500 * 1024 * 1024

# Cloud provider configurations
PROVIDERS = {
    "OpenAI Whisper API": {
        "description": "High accuracy, $0.006/min",
        "model": "whisper-1",
        "max_chunk_bytes": _OPENAI_GROQ_MAX_CHUNK_BYTES,
    },
    "Groq (whisper-large-v3-turbo)": {
        "description": "Very fast, free tier available",
        "model": "whisper-large-v3-turbo",
        "max_chunk_bytes": _OPENAI_GROQ_MAX_CHUNK_BYTES,
    },
    "Deepgram Nova-2": {
        "description": "Fast, supports **Speaker Diarization**",
        "model": "nova-2",
        "diarization": True,
        "max_chunk_bytes": _DEEPGRAM_MAX_CHUNK_BYTES,
    },
    "Deepgram Nova-3 (Multilingual)": {
        "description": "Best for **multilingual** audio, supports diarization",
        "model": "nova-3",
        "diarization": True,
        "multilingual": True,
        "max_chunk_bytes": _DEEPGRAM_MAX_CHUNK_BYTES,
    },
}


def get_max_chunk_bytes(provider: str) -> int:
    """Return the upload size ceiling (in bytes) for a given provider.

    Falls back to the conservative OpenAI/Groq limit for unknown providers
    so callers never get an over-large chunk by accident.
    """
    config = PROVIDERS.get(provider)
    if not config:
        return _OPENAI_GROQ_MAX_CHUNK_BYTES
    return int(config.get("max_chunk_bytes", _OPENAI_GROQ_MAX_CHUNK_BYTES))

# Minimum non-trivial word count to consider a chunk transcript valid.
# Chunks that produce fewer words than this are flagged as potentially garbage.
_MIN_WORDS_PER_CHUNK = 3

# How many words from the end of the previous chunk to compare against when
# deduplicating overlap regions. Larger window = safer but slower.
_DEDUP_WINDOW_WORDS = 40


def transcribe_chunks(
    chunk_paths: list[str],
    provider: str,
    api_key: str,
    language: str | None = None,
    progress_callback=None,
    diarize: bool = False,
    chunk_overlap_ms: int = 5000,
) -> dict:
    """
    Transcribe audio chunks using a cloud API.

    Implements graceful degradation: if a chunk fails after all retries, it is
    marked as failed and skipped rather than aborting the entire transcription.
    Overlapping regions between consecutive chunks are deduplicated to avoid
    repeated sentences at chunk join points.

    Args:
        chunk_paths: List of paths to audio chunk files.
        provider: Cloud provider name (key from PROVIDERS dict).
        api_key: API key for the chosen provider.
        language: Language code or None for auto-detect.
        progress_callback: Callable(current, total, message) for progress updates.
        diarize: Whether to enable speaker diarization (Deepgram only).
        chunk_overlap_ms: Overlap duration in milliseconds used during chunking.
            Used to estimate how aggressively to dedup chunk boundaries.

    Returns:
        dict with keys:
            "text": str - The combined, deduplicated transcript.
            "failed_chunks": list[int] - Zero-based indices of chunks that failed.
            "detected_language": str | None - Language code reported by the API
                (populated only when auto-detect is used and the provider
                supports it; None otherwise).
            "quality_warnings": list[str] - Human-readable quality warnings.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {list(PROVIDERS.keys())}")

    config = PROVIDERS[provider]
    transcripts: list[str] = []
    failed_chunks: list[int] = []
    detected_language: str | None = None
    quality_warnings: list[str] = []
    total = len(chunk_paths)

    # Create API client once for reuse across all chunks
    client = _create_client(provider, api_key)

    # Track timing for ETA estimation
    start_time = time.monotonic()
    chunk_times: list[float] = []

    for i, chunk_path in enumerate(chunk_paths):
        chunk_start = time.monotonic()

        # Build a progress message that includes ETA when we have timing data
        eta_str = _estimate_eta(i, total, chunk_times)
        progress_msg = f"Transcribing chunk {i + 1}/{total} via {provider}...{eta_str}"
        if progress_callback:
            progress_callback(i, total, progress_msg)

        try:
            result = _transcribe_single(chunk_path, provider, config, client, language, diarize)
            text = result["text"]
            chunk_lang = result.get("detected_language")

            # Capture detected language from the first chunk that reports one
            if chunk_lang and detected_language is None:
                detected_language = chunk_lang

            text = text.strip() if text else ""

            # Quality check: warn on suspiciously short or empty chunks
            if not text:
                logger.warning("Chunk %d/%d produced empty transcript", i + 1, total)
                quality_warnings.append(f"Chunk {i + 1} produced no text (may be silence or noise).")
            elif len(text.split()) < _MIN_WORDS_PER_CHUNK:
                logger.warning("Chunk %d/%d produced very short transcript: %r", i + 1, total, text)
                quality_warnings.append(
                    f"Chunk {i + 1} produced very little text ({len(text.split())} word(s)): {text!r}"
                )

            if text:
                transcripts.append(text)

        except Exception as exc:
            # Graceful degradation: log the failure, record the index, and continue
            logger.error(
                "Chunk %d/%d failed after all retries: %s", i + 1, total, exc, exc_info=True
            )
            failed_chunks.append(i)
            quality_warnings.append(
                f"Chunk {i + 1} failed and was skipped: {redact_secrets(str(exc))}"
            )

        chunk_elapsed = time.monotonic() - chunk_start
        chunk_times.append(chunk_elapsed)

    if progress_callback:
        progress_callback(total, total, "Transcription complete!")

    # Deduplicate overlapping regions between consecutive chunk transcripts
    merged = _deduplicate_overlap(transcripts)

    # Validate the overall transcript
    if not merged.strip():
        quality_warnings.append(
            "The final transcript is empty. "
            "Check that the file contains audible speech and that the correct language is selected."
        )
    elif _looks_like_garbage(merged):
        quality_warnings.append(
            "The transcript may contain recognition errors. "
            "This sometimes happens when the wrong language is selected. "
            f"Detected language: {detected_language or 'unknown'}."
        )

    return {
        "text": merged,
        "failed_chunks": failed_chunks,
        "detected_language": detected_language,
        "quality_warnings": quality_warnings,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _estimate_eta(completed: int, total: int, chunk_times: list[float]) -> str:
    """
    Return a human-readable ETA string like ' (ETA ~2m 30s)' or '' if unknown.

    Uses a rolling average of the last 5 chunk durations to smooth estimates.
    """
    if completed == 0 or not chunk_times:
        return ""

    window = chunk_times[-5:]
    avg_seconds = sum(window) / len(window)
    remaining_chunks = total - completed
    eta_seconds = avg_seconds * remaining_chunks

    if eta_seconds < 60:
        return f" (ETA ~{int(eta_seconds)}s)"
    minutes, seconds = divmod(int(eta_seconds), 60)
    return f" (ETA ~{minutes}m {seconds}s)"


def _deduplicate_overlap(transcripts: list[str]) -> str:
    """
    Merge a list of chunk transcripts, removing duplicated text at boundaries.

    When audio chunks overlap by a few seconds the same words appear at the end
    of chunk N and the beginning of chunk N+1. This function detects and removes
    those duplicates by comparing a sliding word window.

    Strategy:
      1. For each consecutive pair (prev, curr), take the last
         _DEDUP_WINDOW_WORDS words of prev as a "tail".
      2. Search for the longest suffix of that tail that appears as a prefix
         of curr (at least 4 words must match to avoid false positives).
      3. Strip that prefix from curr before appending.

    This is purely string-based and intentionally conservative: it only removes
    an overlap when there is a high-confidence match, to avoid accidentally
    deleting legitimate repeated phrases.
    """
    if not transcripts:
        return ""
    if len(transcripts) == 1:
        return transcripts[0]

    merged_words: list[str] = transcripts[0].split()

    for curr_text in transcripts[1:]:
        curr_words = curr_text.split()
        if not curr_words:
            continue

        # Compare against a window of the accumulated text so far
        tail_words = merged_words[-_DEDUP_WINDOW_WORDS:]

        overlap_len = _find_overlap_length(tail_words, curr_words)

        if overlap_len > 0:
            logger.debug(
                "Deduplicating %d overlapping words at chunk boundary", overlap_len
            )
            curr_words = curr_words[overlap_len:]

        if curr_words:
            merged_words.extend(curr_words)

    return " ".join(merged_words)


def _find_overlap_length(tail_words: list[str], curr_words: list[str]) -> int:
    """
    Find the length of the longest suffix of tail_words that equals a prefix
    of curr_words. Returns 0 if no sufficiently long match is found.

    Minimum overlap to act on is 4 words (avoids false positives from common
    short phrases like "und die" / "et le").
    """
    min_overlap = 4
    max_possible = min(len(tail_words), len(curr_words))

    # Normalise for comparison: lowercase, strip punctuation from word edges
    def _norm(w: str) -> str:
        return re.sub(r"^[^\w]+|[^\w]+$", "", w.lower())

    tail_norm = [_norm(w) for w in tail_words]
    curr_norm = [_norm(w) for w in curr_words]

    # Try the longest possible overlap first (greedy)
    for length in range(max_possible, min_overlap - 1, -1):
        # suffix of tail, prefix of curr
        if tail_norm[-length:] == curr_norm[:length]:
            return length

    return 0


def _looks_like_garbage(text: str) -> bool:
    """
    Heuristic check for transcription garbage (wrong language detection).

    Returns True when the text is suspiciously repetitive or contains a very
    high ratio of non-alphabetic characters — both are symptoms of the model
    hallucinating on non-English audio when no language hint is provided.
    """
    if not text or len(text) < 50:
        return False

    words = text.split()
    if not words:
        return False

    # Check for extreme word repetition: if the top word accounts for more than
    # 30% of all words, something is likely wrong.
    from collections import Counter
    counts = Counter(w.lower().strip(".,!?;:\"'") for w in words)
    most_common_word, most_common_count = counts.most_common(1)[0]
    if most_common_count / len(words) > 0.30:
        return True

    # Check ratio of alphabetic characters
    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars / len(text) < 0.50:
        return True

    return False


def _create_client(provider: str, api_key: str):
    """Create an API client for the given provider."""
    if "OpenAI" in provider:
        return OpenAI(api_key=api_key)
    elif "Groq" in provider:
        return Groq(api_key=api_key)
    elif "Deepgram" in provider:
        return DeepgramClient(api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _transcribe_single(
    file_path: str,
    provider: str,
    config: dict,
    client,
    language: str | None,
    diarize: bool = False,
) -> dict:
    """
    Transcribe a single audio file using the specified cloud provider.

    Returns:
        dict with keys:
            "text": str
            "detected_language": str | None
    """
    if "OpenAI" in provider:
        return _transcribe_openai(file_path, config["model"], client, language)
    elif "Groq" in provider:
        return _transcribe_groq(file_path, config["model"], client, language)
    elif "Deepgram" in provider:
        is_multilingual = config.get("multilingual", False)
        return _transcribe_deepgram(file_path, config["model"], client, language, diarize, is_multilingual)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=_is_retriable_error,
    reraise=True,
)
def _transcribe_openai(
    file_path: str,
    model: str,
    client: OpenAI,
    language: str | None,
) -> dict:
    """Transcribe using OpenAI Whisper API with smart retry logic.

    Only retries on transient errors (429, 5xx, connection issues).
    Does NOT retry on 400/401/403 errors to fail fast on invalid keys or bad requests.
    """
    with open(file_path, "rb") as audio_file:
        kwargs = {
            "model": model,
            "file": audio_file,
            # Use verbose_json to get the detected language field back
            "response_format": "verbose_json",
        }
        if language:
            kwargs["language"] = language

        response = client.audio.transcriptions.create(**kwargs)

    text = response.text if hasattr(response, "text") else str(response)
    detected_language = getattr(response, "language", None)

    return {"text": text, "detected_language": detected_language}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=_is_retriable_error,
    reraise=True,
)
def _transcribe_groq(
    file_path: str,
    model: str,
    client: Groq,
    language: str | None,
) -> dict:
    """Transcribe using Groq API with smart retry logic.

    Only retries on transient errors (429, 5xx, connection issues).
    Does NOT retry on 400/401/403 errors to fail fast on invalid keys or bad requests.
    """
    with open(file_path, "rb") as audio_file:
        kwargs = {
            "model": model,
            "file": audio_file,
            "response_format": "verbose_json",
        }
        if language:
            kwargs["language"] = language

        response = client.audio.transcriptions.create(**kwargs)

    text = response.text if hasattr(response, "text") else str(response)
    detected_language = getattr(response, "language", None)

    return {"text": text, "detected_language": detected_language}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=_is_retriable_error,
    reraise=True,
)
def _transcribe_deepgram(
    file_path: str,
    model: str,
    client: DeepgramClient,
    language: str | None,
    diarize: bool = False,
    is_multilingual: bool = False,
) -> dict:
    """Transcribe using Deepgram API with smart retry and diarization support.

    Only retries on transient errors (429, 5xx, connection issues).
    Does NOT retry on 400/401/403 errors to fail fast on invalid keys or bad requests.

    When diarization is enabled, this function preserves punctuation and smart formatting
    by using word-level speaker assignments to find speaker change boundaries, then
    extracting the corresponding text from the formatted transcript.
    """
    # Build keyword arguments for v5+ SDK
    kwargs = {
        "model": model,
        "smart_format": True,
        "punctuate": True,
        "diarize": diarize,
        "paragraphs": True,
    }

    # Handle language settings
    if is_multilingual:
        # Nova-3 multilingual mode: "multi" handles code-switching automatically
        kwargs["language"] = "multi"
    elif language and language != "auto":
        # Honour the user's forced language but still ask Deepgram to report
        # what it detected — without this, ``metadata.detected_language``
        # comes back ``None`` and the UI's mismatch warning can never fire,
        # so a wrong forced-language selection silently produces garbled
        # output. The provider docs allow combining the two: forced language
        # wins for the actual transcription, detection is reported in metadata.
        kwargs["language"] = language
        kwargs["detect_language"] = True
    else:
        # Auto-detect: CRITICAL - without this Deepgram defaults to English
        # and transcribes German/French audio as garbled phonemes
        kwargs["detect_language"] = True

    # Stream the file handle directly to the SDK rather than slurping the
    # whole file into RAM with ``read()`` — for a 500 MB chunk that
    # eliminated up to half a gigabyte of unnecessary allocation per call.
    # The SDK accepts a ``BufferedReader`` for ``transcribe_file``.
    with open(file_path, "rb") as audio_file:
        # Bare ``except`` was previously wrapping every SDK exception in a
        # generic RuntimeError. That defeated the @retry decorator's
        # isinstance-based predicate and meant typed Deepgram errors
        # (NonRetriable, etc.) lost their type before retry inspection.
        # Letting the original exceptions propagate keeps the smart-retry
        # logic working as intended.
        response = client.listen.v1.media.transcribe_file(
            request=audio_file,
            **kwargs,
        )

    # Parse response with null safety checks
    if not response or not response.results:
        return {"text": "", "detected_language": None}

    channels = getattr(response.results, "channels", None)
    if not channels or len(channels) == 0:
        return {"text": "", "detected_language": None}

    channel = channels[0]
    if not channel:
        return {"text": "", "detected_language": None}

    alternatives = getattr(channel, "alternatives", None)
    if not alternatives or len(alternatives) == 0:
        return {"text": "", "detected_language": None}

    alternative = alternatives[0]
    if not alternative:
        return {"text": "", "detected_language": None}

    # Extract the detected language from metadata if available
    detected_language: str | None = None
    metadata = getattr(response, "metadata", None)
    if metadata:
        detected_language = getattr(metadata, "detected_language", None)

    # Get the fully formatted transcript (with punctuation, smart formatting)
    full_transcript = getattr(alternative, "transcript", None) or ""

    if diarize:
        # Prefer Deepgram's structured `paragraphs` output: it already
        # carries speaker IDs, smart-format punctuation, and
        # natural-pause paragraph boundaries, so we don't have to rebuild
        # any of that from raw word timestamps.
        text = _format_diarized_from_paragraphs(alternative)
        if text is None:
            # Fall back to walking words[] for older models or responses
            # that don't include a paragraphs object.
            text = _format_diarized_from_words(alternative)
        if not text:
            # Ultimate fallback: the unformatted full transcript.
            text = full_transcript
        return {"text": text, "detected_language": detected_language}
    else:
        return {"text": full_transcript, "detected_language": detected_language}


def _format_diarized_from_paragraphs(alternative) -> str | None:
    """Build a ``**Speaker N:**\\n<text>`` transcript from Deepgram's
    ``paragraphs`` structure.

    Returns None when the paragraphs object is missing or empty so the
    caller can fall back to the word-loop. Each paragraph in Deepgram's
    response carries a speaker, a start/end, and a list of sentences
    whose ``.text`` is already punctuated by smart_format — so we just
    join sentences within a paragraph and emit one speaker block per
    paragraph (matching the prior word-loop behaviour where pauses
    produced new blocks).
    """
    paragraphs_obj = getattr(alternative, "paragraphs", None)
    if not paragraphs_obj:
        return None
    paragraphs_list = getattr(paragraphs_obj, "paragraphs", None)
    if not paragraphs_list:
        return None

    parts: list[str] = []
    for para in paragraphs_list:
        speaker = getattr(para, "speaker", None)
        if speaker is None:
            continue
        sentences = getattr(para, "sentences", None) or []
        text = " ".join(
            (getattr(s, "text", "") or "").strip()
            for s in sentences
            if getattr(s, "text", "")
        ).strip()
        if not text:
            continue
        parts.append(f"**Speaker {speaker}:**\n{text}")

    if not parts:
        return None
    return "\n\n".join(parts)


def _format_diarized_from_words(alternative) -> str:
    """Word-loop fallback used when ``alternative.paragraphs`` is absent.

    Walks the per-word output, starting a new ``**Speaker N:**`` block on
    every speaker change or pause longer than 1.5 seconds. Returns an
    empty string when no words are present.
    """
    words = getattr(alternative, "words", None)
    if not words or len(words) == 0:
        return ""

    # Pause threshold in seconds - gaps longer than this trigger paragraph breaks
    PAUSE_THRESHOLD = 1.5

    transcript_parts: list[str] = []
    current_speaker = None
    segment_words: list[str] = []
    last_word_end: float | None = None

    for word in words:
        speaker = getattr(word, "speaker", None)
        word_start = getattr(word, "start", None)
        word_end = getattr(word, "end", None)
        # Use punctuated_word if available (has punctuation), else fall back to word
        word_text = getattr(word, "punctuated_word", None) or getattr(word, "word", "")

        # Detect pause: gap between last word end and this word start
        is_pause = False
        if last_word_end is not None and word_start is not None:
            gap = word_start - last_word_end
            if gap >= PAUSE_THRESHOLD:
                is_pause = True

        # Start new segment on speaker change OR significant pause
        if speaker != current_speaker or (is_pause and segment_words):
            # Flush previous segment
            if current_speaker is not None and segment_words:
                segment_text = " ".join(segment_words)
                transcript_parts.append(f"**Speaker {current_speaker}:**\n{segment_text}")

            # Only reset speaker if it actually changed
            if speaker != current_speaker:
                current_speaker = speaker
            segment_words = []

        if word_text:
            segment_words.append(word_text)

        # Track end time for pause detection
        if word_end is not None:
            last_word_end = word_end

    # Flush the last speaker segment
    if current_speaker is not None and segment_words:
        segment_text = " ".join(segment_words)
        transcript_parts.append(f"**Speaker {current_speaker}:**\n{segment_text}")

    return "\n\n".join(transcript_parts)