# 🎙️ Transcriber

Production-ready audio and video transcription powered by cloud AI. Optimised for **multilingual content** (German, French, Spanish, Italian, and 30+ more), with speaker diarization, timestamps, low-confidence highlighting, AI summaries, and batch processing — all from a single Streamlit app.

> **Transcripts are produced in the spoken language of the audio.** The app does not translate. Pick Auto-detect or the actual spoken language — never the language you want the output in.

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Configuration](#configuration)
- [Choosing a Provider](#choosing-a-provider)
- [User Guide](#user-guide)
  - [1. Provider and API key](#1-provider-and-api-key)
  - [2. Spoken language](#2-spoken-language)
  - [3. Optional features](#3-optional-features)
  - [4. Provide your audio](#4-provide-your-audio)
  - [5. Transcribe](#5-transcribe)
  - [6. Review and edit](#6-review-and-edit)
  - [7. Improve readability](#7-improve-readability)
  - [8. Rename speakers](#8-rename-speakers)
  - [9. Download](#9-download)
- [Batch mode](#batch-mode)
- [Privacy and cost](#privacy-and-cost)
- [Project structure](#project-structure)
- [Running tests](#running-tests)
- [Troubleshooting](#troubleshooting)

---

## Features

### Transcription
- **Four cloud providers** — OpenAI Whisper API, Groq (whisper-large-v3-turbo), Deepgram Nova-2, Deepgram Nova-3 (Multilingual)
- **30+ languages** — Arabic, Bulgarian, Catalan, Chinese (Mandarin), Czech, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Hungarian, Indonesian, Italian, Japanese, Korean, Malay, Norwegian, Persian, Polish, Portuguese, Romanian, Russian, Slovak, Spanish, Swedish, Thai, Turkish, Ukrainian, Vietnamese
- **Auto-detect or pick a specific language** — Nova-3 also handles **code-switching** within the same recording
- **Speaker diarization** (Deepgram only) — `**Speaker 0:**`, `**Speaker 1:**`, …
- **Three input modes** — drag-and-drop upload, local filesystem path, or URL paste (YouTube, podcast, any yt-dlp-supported source)
- **Large file support** — automatic chunking with memory-efficient streaming, up to the 2 GB upload cap
- **Audio + video formats** — MP3, WAV, M4A, FLAC, OGG, AAC, OPUS, WEBM, MP4, MOV, AVI, MKV, WMA

### Output quality
- **Provider-aware chunking** — each provider's actual upload ceiling is respected (Deepgram is much higher than OpenAI/Groq)
- **Boundary deduplication** — repeated text at chunk seams is collapsed automatically
- **Garbage detection** — warns when a chunk looks like hallucination or wrong-language output
- **Graceful degradation** — failed chunks are skipped (not fatal); missing indices are surfaced
- **Smart retries** — exponential backoff on rate limits and 5xx; immediate fail on bad keys (no wasted retries)
- **Language-mismatch warning** — if you forced a language but the API heard something else, the app warns *before* you spend hours editing garbled output
- **Secret redaction** — provider SDK exceptions are scrubbed for bearer tokens / API keys before they reach the screen

### Readability
- **Paragraph breaks** — auto-inserted at speaker changes and natural pauses (>1.5 s silence)
- **Speaker labels** — bold blue, on their own line, indented body for visual clarity
- **Speaker renaming** — replace `Speaker 0` with `Maria`, etc., across the whole transcript and all exports
- **Filler-word highlighting** — *um, uh, äh, euh, like, basically* (English/German/French) marked italic for review, stripped on PDF export
- **Search & highlight** — find any word or phrase in the preview pane
- **Timestamps** — optional `[HH:MM:SS]` markers at every paragraph and speaker change, embedded in DOCX/PDF
- **Low-confidence highlighting** (Deepgram only) — words below an adjustable threshold (default 0.6) get an amber highlight in the editor and DOCX, so manual review focuses where it matters

### AI post-processing
- **Executive summary** — 3–5 sentences in the source language
- **Key topics** — 3–7 short phrases ordered by importance
- **Action items** — every explicit commitment, with speaker attribution; empty list when none are present (the prompt explicitly forbids inventing them)
- **Two summary providers** — Groq (llama-3.3-70b) and OpenAI (gpt-4o-mini); the same key is reused when the transcription and summary providers share a family
- **Cache-aware** — edits invalidate the summary; a *Regenerate summary* button gives you control over re-spend
- **Embedded in exports** — DOCX and PDF include the summary block above the transcript

### Editor
- **Edit / Preview modes** — raw text editing or a formatted visual view
- **Reading stats** — word count, character count, reading-time estimate
- **Live ETA** — estimated time remaining for large files during transcription
- **Cached re-renders** — preview HTML, reading stats, and exports are cached on content; toggling a checkbox or typing in the search box doesn't re-do the work

### Export
- **DOCX** — bold blue speaker labels, indented body, italic gray fillers, embedded summary, optional yellow text-highlight on low-confidence words
- **PDF** — Unicode-safe (ü ö ä ß œ é è à) via Arial Unicode, colour-coded speaker labels, embedded summary, fillers stripped clean
- **TXT** (batch only) — alongside DOCX/PDF inside the bundled zip

### Batch transcription
- **Drop multiple files at once** — UI switches to a queue view automatically
- **Single zip download** — `<filename>/transcription.docx | .pdf | .txt` for each file
- **Per-file isolation** — a single failure produces an `error.txt` in that file's folder; the rest of the batch keeps running
- **Per-file AI summary** — when enabled, each file's docx/pdf carries its own summary

---

## Quick Start

```bash
# 1. Install ffmpeg (system binary; not a Python dep)
brew install ffmpeg            # macOS
# sudo apt install ffmpeg      # Debian/Ubuntu

# 2. Clone and sync
git clone <repo-url>
cd Transcriber
uv sync

# 3. Run
uv run streamlit run app.py
```

The app opens at <http://localhost:8501>. Paste an API key in the sidebar, drop a file in, click **Begin Transcription**.

---

## Prerequisites

| Requirement | Version | Install |
|---|---|---|
| Python | 3.11.7+ | [python.org](https://python.org) |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **ffmpeg + ffprobe** | any recent | `brew install ffmpeg` (macOS) · `sudo apt install ffmpeg` (Debian/Ubuntu) · [ffmpeg.org](https://ffmpeg.org) (Windows) |
| API key | — | One of: [OpenAI](https://platform.openai.com/api-keys) · [Groq](https://console.groq.com/keys) · [Deepgram](https://console.deepgram.com/) |

> **ffmpeg is required, not optional.** The app shells out to it for every audio operation (format conversion, chunking, metadata). If `ffmpeg` or `ffprobe` is not on `PATH`, the app refuses to start with a clear install hint.

---

## Setup

```bash
git clone <repo-url>
cd Transcriber

# uv creates the venv and installs deps from the lockfile
uv sync

# Run
uv run streamlit run app.py
```

To stop the server, press `Ctrl+C` in the terminal.

---

## Configuration

### Upload size

The default upload cap is **2 GB**, set in `.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 2000   # in MB
```

The app also enforces this at the application layer, so loosening the Streamlit setting alone is not enough — adjust both if you need to go higher.

### Provider chunk limits

Chunk sizes are derived per provider automatically:

| Provider | Per-chunk cap |
|---|---|
| OpenAI Whisper API | ~24 MB (provider-side limit is 25 MB; we leave headroom) |
| Groq | ~24 MB |
| Deepgram Nova-2 / Nova-3 | ~1.9 GB |

Files above the cap are split with ffmpeg, transcribed concurrently (3 workers), reassembled, and deduplicated at the boundaries.

### API keys

API keys are entered in the sidebar and live only in the Streamlit session — they are never written to disk, environment, or logs. A separate key field is shown per provider; switching providers does not lose the others.

---

## Choosing a Provider

| Provider | Best for | Speaker diarization | Multilingual / code-switching | Indicative cost |
|---|---|---|---|---|
| **OpenAI Whisper API** | General-purpose accuracy, single language | ✗ | ✓ (single language at a time) | ~$0.006 / minute |
| **Groq (whisper-large-v3-turbo)** | Speed; free tier for tinkering | ✗ | ✓ | Free tier available |
| **Deepgram Nova-2** | Interviews, meetings, anything with multiple speakers | ✓ | ✓ | Pay-as-you-go |
| **Deepgram Nova-3 (Multilingual)** | Recordings that **switch language mid-sentence** (e.g. German + French in one meeting) | ✓ | ✓✓ (true code-switching) | Pay-as-you-go |

> **Rule of thumb:** Multiple speakers → Deepgram. Mixed languages → Nova-3. Single speaker, single language, lowest cost → Groq. Single speaker, highest accuracy → OpenAI.

---

## User Guide

### 1. Provider and API key

Open the **sidebar** (left). Pick a provider, paste the matching key into the **API Key** field. Keys are session-only.

### 2. Spoken language

Under **Spoken Language**, pick **Auto-detect** (recommended) or a specific language from the list of 30+. For audio that mixes languages, switch the provider to **Deepgram Nova-3 (Multilingual)** and leave language on Auto-detect.

> **Pick the language *spoken* in the recording, not the language you want the transcript in.** The app does not translate.

### 3. Optional features

All optional toggles live in the sidebar:

#### Speaker diarization (Deepgram only)
Identify and label each speaker. Paragraph breaks are auto-inserted at speaker changes and at silence gaps >1.5 s.

```
**Speaker 0:**
Guten Morgen, willkommen zur Konferenz.

**Speaker 1:**
Merci beaucoup pour l'invitation.
```

#### Highlight low-confidence words (Deepgram only)
Wrap words at or below a confidence threshold (default 0.6, slider 0.3–0.95) in `~~word~~` markers. Rendered as an amber highlight in the preview and a yellow text-highlight in DOCX. PDF strips the markers cleanly. Use it as a guide for manual review — these are the words the model was least sure of.

> **Trade-off:** This mode bypasses Deepgram's smart-formatted `paragraphs` shortcut so it can read per-word confidence — niceties like *"two thousand twenty"* → *"2020"* are not applied.

#### Include timestamps
Insert `[HH:MM:SS]` markers at every paragraph and speaker change. Visible in the editor (muted gray) and embedded in DOCX/PDF. Works with all providers — Deepgram uses paragraph timing; OpenAI/Groq use segment timing from `verbose_json`.

#### AI summary
Tick **Generate after transcription** to add a single LLM call after a successful transcription. Pick **Groq (llama-3.3-70b)** (faster, free tier) or **OpenAI (gpt-4o-mini)** (slightly more accurate). When the summary provider's family matches the transcription provider's, the same API key is reused automatically.

You'll get:
- **Executive summary** — 3–5 sentences in the source language
- **Key topics** — 3–7 short phrases by importance
- **Action items** — every explicit commitment, with speaker attribution; empty when none

The summary appears above the editor and is embedded in DOCX/PDF. Editing the transcript invalidates the cached summary and shows a **Regenerate summary** button — you choose whether the edit warrants a fresh call.

### 4. Provide your audio

Three input modes:

- **Upload** — drag and drop one or more files. Up to 2 GB each.
- **Local path** — paste an absolute path (e.g. `/Users/you/meeting.mp4`).
- **URL** — paste a YouTube/podcast/direct media URL and click **Fetch URL**. yt-dlp downloads the audio into a temp file via ffmpeg, then the file behaves identically to an upload.

Once the source is resolved, audio info is displayed:

```
Duration · 1 h 23 m | File Size · 78 MB | Channels · 2 | Sample Rate · 44 100 Hz
```

If the file is bigger than the provider's per-chunk cap, a *"will be split into chunks"* notice is shown.

### 5. Transcribe

Click **Begin Transcription**. A progress bar shows current chunk and live ETA. When done:

- A **Detected language** notice appears if Auto-detect was used.
- A **language mismatch warning** appears if you forced a language but the API heard something else — re-run with Auto-detect or pick the matching language.
- Failed chunks (if any) are listed and skipped — the transcript continues around them.
- Quality warnings (very short output, suspected garbage) are surfaced.
- Balloons on success.

### 6. Review and edit

The transcript appears in the **Edit / Preview** panel.

- **Edit** — raw text editor; speaker labels follow `**Speaker X:**` on their own line.
- **Preview** — formatted visual view with a search box for find-and-highlight.

A live stats line under the panel shows word count, character count, and estimated reading time.

### 7. Improve readability

Expand **Readability options**, tick **Highlight filler words**, click **Apply Filler Highlighting**. Fillers (um, uh, äh, euh, like, basically) become *italic gray*. They are stripped cleanly on PDF export and remain italic in DOCX.

### 8. Rename speakers

If diarization is on, expand **Rename speakers**, type a real name for each generic label, click **Apply Speaker Names**. Updates propagate everywhere — editor, preview, exports.

```
Speaker 0 → Maria
Speaker 1 → Jean-Paul
```

### 9. Download

Click **Download · DOCX** or **Download · PDF**. Both formats include:

- Bold **blue** speaker labels on their own line, indented body
- Filler words in *italic gray* (DOCX) or removed (PDF)
- Optional embedded AI summary block at the top
- Full Unicode (ü ö ä ß œ é è à)
- Yellow text-highlight on low-confidence words (DOCX only)

---

## Batch mode

Drop **two or more files** into the uploader at once. The page switches to a queue view:

```
3 files queued:
- meeting-2026-04-12.mp4   · 213.4 MB
- interview-marie.m4a      · 47.8 MB
- podcast-ep-12.mp3        · 92.1 MB
```

Click **Begin Batch Transcription**. Each file goes through the same pipeline (chunking, diarization, timestamps, confidence highlighting, summary — whatever you've enabled). When all files are done, you get a single **ZIP** containing:

```
transcripts.zip
├── meeting-2026-04-12/
│   ├── transcription.docx
│   ├── transcription.pdf
│   └── transcription.txt
├── interview-marie/
│   ├── ...
└── podcast-ep-12/
    └── ...
```

Failed files get an `error.txt` instead of the three transcript files, so it's obvious which entry is which. The rest of the batch keeps running on a per-file failure — one bad upload doesn't kill the whole job.

---

## Privacy and cost

- **Audio** is uploaded directly to the chosen provider (OpenAI, Groq, Deepgram) for transcription, and to OpenAI/Groq for the optional AI summary. Review their data-retention policies before uploading sensitive material.
- **API keys** live only in the active Streamlit session — never persisted, never logged. Error messages are scrubbed for anything that looks like a bearer token before they reach the screen.
- **Temp files** (uploads, chunks, URL downloads) are written to the OS temp directory and cleaned up automatically — chunks immediately after transcription, uploads when superseded by a different file in the same session.
- **Cost transparency** — Auto-detect, language mismatch, and the *Regenerate summary* button are designed so you don't pay for transcription twice unintentionally. The AI summary is opt-in and cached against the transcript content.

---

## Project structure

```
Transcriber/
├── app.py                       # Streamlit entry point and UI
├── assets/
│   └── styles.css               # Editorial paper-and-ink stylesheet
├── transcriber/
│   ├── __init__.py
│   ├── audio_processor.py       # Format conversion, chunking, ffprobe metadata
│   ├── cloud_engine.py          # OpenAI / Groq / Deepgram, retries, dedup, garbage detection
│   ├── ai_summary.py            # LLM post-processing (summary + topics + actions)
│   ├── batch.py                 # Batch helpers and zip builder
│   ├── exporter.py              # DOCX and PDF export with speaker formatting
│   ├── language.py              # ISO-639-1 normalisation for detected languages
│   ├── text_processor.py        # Filler detection, speaker rename, search highlight
│   └── url_source.py            # yt-dlp wrapper for URL inputs
├── tests/
│   ├── test_all.py
│   ├── test_cloud_engine.py
│   ├── test_ai_summary.py
│   ├── test_batch.py
│   ├── test_confidence_highlight.py
│   ├── test_language.py
│   ├── test_redact_secrets.py
│   ├── test_text_processor.py
│   ├── test_timestamps.py
│   └── test_url_source.py
├── .streamlit/config.toml       # Upload cap (2 GB)
├── pyproject.toml
└── README.md
```

---

## Running tests

```bash
uv run pytest tests/
```

All tests are offline — no API keys required. Provider SDKs are mocked at the boundary; ffmpeg is exercised against tiny fixture files in `tests/`.

The pytest config promotes warnings to errors so a new deprecation can't quietly creep in (the deepgram-sdk → `websockets.legacy` warning is the only allow-listed exemption — it's mid-migration upstream).

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Startup error: *"ffmpeg and ffprobe are required"* | Install ffmpeg: `brew install ffmpeg` (macOS) or `sudo apt install ffmpeg` (Debian/Ubuntu) |
| Transcript came back in English but the audio is German/French | The app does not translate. Pick **Auto-detect** or the actual spoken language — never the language you want the output in |
| You forced a language and the transcript looks garbled | The app shows a warning when the API detected a different language than the one forced. Re-run with Auto-detect or the matching language |
| PDF shows `?` for ü / ä / ö / œ | Arial Unicode must be installed. On macOS it lives in `/System/Library/Fonts/Supplemental/`. Use the DOCX export as a fallback |
| Transcription fails immediately on click | Bad API key — invalid keys are rejected without retrying. Re-paste the key and verify it in the provider's console |
| *"Rate limit exceeded"* | Wait a moment and click again. Groq's free tier is the most rate-sensitive |
| Large file runs out of memory | The app streams files through ffmpeg without loading them into RAM. Make sure ffmpeg is on `PATH` and you have free disk space for chunks |
| Duplicate words at chunk boundaries | Handled automatically by boundary deduplication. If you still see duplicates, file an issue with the offending audio |
| AI summary failed | Check the summary-provider key (it can be different from the transcription key when the families don't match). The transcript itself is unaffected |
| Speaker labels missing on Deepgram | Make sure **Speaker Diarization** is ticked in the sidebar (Deepgram only) |
| URL fetch failed | Some platforms throttle yt-dlp. Re-try, switch to a direct media URL if available, or download the file locally first |
| Upload rejected as too large | The cap is 2 GB. Adjust `.streamlit/config.toml` *and* the app-layer limit in `app.py` if you need to raise it |
