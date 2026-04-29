"""Batch transcription helpers.

The single-file UI flow lives in :mod:`app` to stay close to the
Streamlit widgets it drives. The pieces here are the bits the batch
path reuses across N files: a self-contained "transcribe one path"
call that mirrors the single-file pipeline exactly, and a zip builder
for the final download.

Keeping these out of ``app.py`` means the batch logic is testable
without spinning up Streamlit, and the single-file path stays
unchanged — the batch path is purely additive.
"""

import io
import os
import zipfile
from typing import Callable, Optional

from . import audio_processor, cloud_engine, exporter


def transcribe_one(
    audio_path: str,
    provider: str,
    api_key: str,
    language: Optional[str] = None,
    diarize: bool = False,
    include_timestamps: bool = False,
    low_confidence_threshold: Optional[float] = None,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """Run the chunk + transcribe pipeline for a single audio file.

    Mirrors the orchestration the single-file Streamlit path performs
    inline: probe metadata once, plan chunks, compute per-chunk offsets
    (needed for the timestamped path), stream chunks into the
    transcription pool, and clean up chunk temp files at the end —
    even when the call raises.

    Returns the dict produced by :func:`cloud_engine.transcribe_chunks_streaming`,
    augmented with ``duration_seconds`` so callers (the batch summary
    table in particular) can show per-file metadata without a second
    ffprobe spawn.
    """
    info = audio_processor.get_audio_info(audio_path)
    max_bytes = cloud_engine.get_max_chunk_bytes(provider)
    total_chunks, raw_iter = audio_processor.iter_chunks(
        audio_path,
        max_bytes=max_bytes,
        duration_seconds=info["duration_seconds"],
        progress_callback=progress_callback,
    )
    chunk_offsets = audio_processor.compute_chunk_offsets(
        duration_seconds=info["duration_seconds"],
        file_size_bytes=os.path.getsize(audio_path),
        max_bytes=max_bytes,
    )

    chunk_paths: list[str] = []

    def _collecting_iter():
        """Tee the chunk iterator into ``chunk_paths`` for cleanup —
        identical to the single-file flow's tee in app.py so a partial
        run doesn't leak the chunks ffmpeg already produced."""
        for path in raw_iter:
            chunk_paths.append(path)
            yield path

    try:
        result = cloud_engine.transcribe_chunks_streaming(
            chunk_iter=_collecting_iter(),
            total=total_chunks,
            provider=provider,
            api_key=api_key,
            language=language,
            progress_callback=progress_callback,
            diarize=diarize,
            include_timestamps=include_timestamps,
            chunk_offsets=chunk_offsets,
            low_confidence_threshold=low_confidence_threshold,
        )
    finally:
        if chunk_paths:
            audio_processor.cleanup_chunks(chunk_paths, audio_path)

    result["duration_seconds"] = info["duration_seconds"]
    return result


def build_zip(per_file: list[dict]) -> bytes:
    """Pack each per-file transcript into a single zip archive.

    Each entry in ``per_file`` is expected to look like::

        {
            "filename": "meeting.mp4",   # original audio filename
            "text": "...",                 # transcript text (may be "")
            "error": "...",                # set when transcription failed
        }

    Successful files get ``<stem>/transcription.docx``,
    ``<stem>/transcription.pdf``, and ``<stem>/transcription.txt``.
    Failed files get ``<stem>/error.txt`` instead so the user can see
    *why* a particular file is missing without comparing the zip
    contents to the input list. Per-export failures (e.g. a font issue
    on a single PDF) write a side-car ``*_error.txt`` rather than
    aborting the whole zip — partial output beats no output.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for entry in per_file:
            stem = _safe_stem(entry.get("filename", "untitled"))

            error = entry.get("error")
            if error:
                zf.writestr(f"{stem}/error.txt", str(error))
                continue

            text = entry.get("text") or ""
            if not text.strip():
                zf.writestr(f"{stem}/empty.txt", "Transcription produced no text.")
                continue

            zf.writestr(f"{stem}/transcription.txt", text)

            try:
                zf.writestr(
                    f"{stem}/transcription.docx",
                    exporter.export_docx(text, title=stem),
                )
            except Exception as exc:  # noqa: BLE001 — partial output beats none
                zf.writestr(f"{stem}/docx_error.txt", str(exc))

            try:
                zf.writestr(
                    f"{stem}/transcription.pdf",
                    exporter.export_pdf(text, title=stem),
                )
            except Exception as exc:  # noqa: BLE001 — partial output beats none
                zf.writestr(f"{stem}/pdf_error.txt", str(exc))

    return buf.getvalue()


def _safe_stem(filename: str) -> str:
    """Strip the extension and sanitise the result for use as a zip
    folder name.

    The user-supplied filename can contain anything the OS allows, but
    inside a zip we want predictable, cross-platform-safe names: no
    path separators, no leading/trailing whitespace, and a fallback
    for empty or all-special inputs (otherwise the zip would silently
    end up with two entries that share a prefix, masking each other).
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    # Replace path separators and a small set of characters known to
    # confuse Windows or shells. We err on the side of keeping accented
    # characters intact since transcripts often correspond to
    # non-English source recordings.
    bad = '/\\:*?"<>|\r\n\t'
    cleaned = "".join("_" if c in bad else c for c in stem).strip()
    return cleaned or "untitled"
