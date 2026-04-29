# 🎙️ Transcriber

Production-ready audio and video transcription powered by cloud AI. Optimised for **multilingual content** including German and French, with formatted output and speaker identification.

---

## Features

### Transcription
- **Multiple cloud providers** — OpenAI Whisper API, Groq, Deepgram Nova-2, Deepgram Nova-3 (Multilingual)
- **Multilingual support** — Auto-detect or pick from 30+ languages (English, German, French, Spanish, Italian, Portuguese, Dutch, Polish, Russian, Japanese, Chinese, Korean, Arabic, Hindi, Turkish, and more); Nova-3 handles code-switching within the same recording
- **Speaker diarization** — Identifies individual speakers (Deepgram only)
- **Large file support** — Automatic chunking and memory-efficient streaming for files of any size
- **Video support** — Extracts audio from MP4, MOV, AVI, MKV, and more
- **URL input** — Paste a YouTube, podcast, or direct media URL; the audio is downloaded and transcribed without leaving the app
- **All audio formats** — MP3, WAV, M4A, FLAC, OGG, AAC, OPUS, WEBM, and more

### Output Quality
- **Chunk deduplication** — Removes repeated text at chunk boundaries
- **Garbage detection** — Warns when output looks like hallucination or wrong language
- **Graceful degradation** — Failed chunks are skipped, not fatal; gaps are flagged
- **Smart retry logic** — Retries on rate limits and server errors; fails immediately on bad API keys

### Readability
- **Paragraph breaks** — Automatically inserted at natural pauses (>1.5 s silence) and speaker changes
- **Speaker labels** — Formatted bold on their own line (`**Speaker 0:**`) for clear visual separation
- **Speaker renaming** — Replace generic "Speaker 0" labels with actual names (e.g., Maria, John)
- **Filler word highlighting** — Marks um, uh, äh, euh, like, basically in italic for easy review (English, German, French)
- **Search & highlight** — Find and highlight any word or phrase in the preview
- **Timestamps** — Optional `[HH:MM:SS]` markers at every paragraph and speaker change, rendered in the editor and embedded in DOCX/PDF exports
- **Confidence highlighting** — Optional amber highlight on words Deepgram was least sure of, so manual review focuses on the parts that actually need it (Deepgram only)

### Editor
- **Edit / Preview modes** — Toggle between raw text editing and a formatted visual preview
- **Reading time estimate** — Shows word count, character count, and estimated reading time
- **ETA progress** — Live estimated time remaining during transcription of large files

### Export
- **DOCX** — Formatted Word document with bold speaker labels (blue), indented content, and italic filler words
- **PDF** — Unicode-capable export (supports ü, ö, ä, œ, é, etc.) with colour-coded speaker labels

---

## Prerequisites

| Requirement | Version | Install |
|---|---|---|
| Python | 3.11+ | [python.org](https://python.org) |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **ffmpeg + ffprobe** | any recent | `brew install ffmpeg` (macOS) or `sudo apt install ffmpeg` (Debian/Ubuntu) |

> **ffmpeg is required, not optional.** The app uses ffmpeg for all audio
> transcoding, chunking, and metadata reading. If it isn't on `PATH`, the
> app exits at startup with a clear install hint.

---

## Setup

```bash
git clone <repo-url>
cd Transcriber

# Install dependencies (uv handles the venv automatically)
uv sync

# Run the app
uv run streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Tutorial

### 1. Choose a Provider and Enter Your API Key

Open the **sidebar** (left panel). Select a transcription provider:

| Provider | Best For | Cost |
|---|---|---|
| **OpenAI Whisper API** | General accuracy | ~$0.006/min |
| **Groq (whisper-large-v3-turbo)** | Speed, low cost | Free tier available |
| **Deepgram Nova-2** | Speaker identification | Pay-as-you-go |
| **Deepgram Nova-3 (Multilingual)** | Mixed German/French audio | Pay-as-you-go |

Paste your API key into the **API Key** field. Keys are session-only and never stored.

---

### 2. Set the Language

Under **Spoken Language**, choose:
- **Auto-detect** — the API identifies the language automatically
- **A specific language** — set explicitly for best accuracy on single-language files. The dropdown covers 30+ languages including Arabic, Chinese, Dutch, German, Hindi, Japanese, Korean, Polish, Portuguese, Russian, Spanish, Turkish, Ukrainian, and Vietnamese.

> **Tip:** For recordings that switch between German and French, select **Deepgram Nova-3 (Multilingual)** and leave language on Auto-detect. Nova-3 handles code-switching within the same sentence.

---

### 3b. Highlight Low-Confidence Words (Optional, Deepgram only)

Tick **Highlight low-confidence words** in the sidebar (Deepgram providers only). Deepgram returns a per-word confidence score; words at or below the threshold (default 0.6, adjustable via the slider) are wrapped in `~~word~~` markers and rendered with an amber highlight in the editor preview and a yellow text-highlight in the DOCX export. PDF export strips the markers cleanly. Use this as a guide for manual review — the highlighted words are the ones the model was least sure of.

This requires per-word data, so transcripts are built from Deepgram's word list (the smart-formatted `paragraphs` shortcut is bypassed). Smart-format niceties like "two thousand twenty" → "2020" are not applied in this mode.

---

### 3a. Enable Timestamps (Optional)

Tick **Include timestamps** in the sidebar to insert `[HH:MM:SS]` markers at every paragraph and speaker change. They are rendered in the editor (muted gray) and embedded in DOCX/PDF exports — useful for navigating long recordings or producing professional transcripts where every quote needs to be cited to a time.

Works with all providers: Deepgram uses paragraph-level timing from its response; OpenAI Whisper / Groq use segment-level timing from `verbose_json`.

---

### 3. Enable Speaker Diarization (Optional)

If your recording has **multiple speakers** (e.g., an interview or meeting), tick **Speaker Diarization** in the sidebar.

This is only available with Deepgram providers. When enabled, the transcript labels each speaker:

```
**Speaker 0:**
Guten Morgen, willkommen zur Konferenz.

**Speaker 1:**
Merci beaucoup pour l'invitation.
```

Paragraphs are automatically inserted at:
- **Speaker changes** — every time a different person speaks
- **Natural pauses** — silence gaps longer than 1.5 seconds

---

### 4. Provide Your Audio

Three options:
- **Drag and drop** a file into the upload area
- **Paste a file path** (e.g., `/Users/you/meeting.mp4`) into the path field
- **Paste a URL** (YouTube video, podcast episode, or direct media link) and click **Fetch URL**

For uploads and paths, audio info (duration, size, sample rate) is displayed immediately. For URLs, click **Fetch URL** first; the audio is downloaded into a temp file via yt-dlp + ffmpeg, then audio info appears. Large files will be split into chunks automatically — you will see a notice.

---

### 5. Transcribe

Click **Begin Transcription**.

A progress bar and live status show which chunk is being processed, plus an estimated time remaining (ETA) for large files.

When complete:
- A detected language notice appears if Auto-detect was used
- If you forced a specific language and the API detected a different one, a warning is shown so you can re-run with Auto-detect or pick the matching language — this prevents the "garbled output because the wrong language was forced" failure mode
- Any failed chunks or quality warnings are shown
- Balloons appear on success

> **Note:** Transcripts are produced in the spoken language of the audio. The app does not translate — choose Auto-detect or the actual spoken language, not your preferred output language.

---

### 6. Review and Edit the Transcript

The transcript appears in the **Edit / Preview** panel.

#### Edit Mode
The raw text editor — make any corrections directly. Speaker labels follow the format `**Speaker X:**` on their own line.

#### Preview Mode
A formatted visual view. Use the **Search** box to find and highlight specific words or names.

---

### 7. Improve Readability

Expand **Readability options**:

**Highlight filler words** — tick the checkbox and click **Apply Filler Highlighting**. Filler words (um, uh, äh, euh, like, basically, etc.) appear in *italic gray* so you can spot them at a glance. They are stripped cleanly on export.

---

### 8. Rename Speakers

If diarization is enabled, expand **Rename speakers**. Type the actual name for each speaker and click **Apply Speaker Names**:

```
Speaker 0 → Maria
Speaker 1 → Jean-Paul
```

The labels update throughout the entire transcript instantly. Exported documents use the custom names.

---

### 9. Download

Click **Download · DOCX** or **Download · PDF**.

Both formats include:
- Speaker names in **bold blue** on their own line
- Speaker text indented beneath the label
- Filler words in *italic gray* (DOCX) or removed (PDF)
- Full Unicode support — ü, ö, ä, ß, œ, é, è, à all render correctly

---

## Project Structure

```
Transcriber/
├── app.py                   # Streamlit entry point and UI
├── assets/
│   └── styles.css           # Editorial paper-and-ink stylesheet (loaded by app.py)
├── transcriber/
│   ├── __init__.py
│   ├── audio_processor.py   # Format conversion, chunking, ffprobe metadata
│   ├── cloud_engine.py      # OpenAI / Groq / Deepgram API, retry logic, deduplication
│   ├── exporter.py          # DOCX and PDF export with speaker formatting
│   ├── language.py          # ISO-639-1 normalization for detected languages
│   ├── text_processor.py    # Filler detection, speaker renaming, search highlight
│   └── url_source.py        # yt-dlp wrapper for downloading audio from URLs
├── tests/
│   ├── test_all.py            # Core test suite (audio, chunking, export)
│   ├── test_cloud_engine.py   # Cloud engine unit tests (retry, dedup, garbage detection)
│   ├── test_language.py       # Language code normalization
│   ├── test_redact_secrets.py # Verifies API keys are stripped from error messages
│   ├── test_text_processor.py
│   ├── test_url_source.py     # yt-dlp wrapper unit tests (mocked)
│   └── _make_test_video.py    # Manual fixture regen (not run by pytest)
├── pyproject.toml
└── README.md
```

---

## Running Tests

```bash
uv run pytest tests/
```

All tests are offline — no API keys required.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Transcript is in English even though audio is German | Make sure **Auto-detect** is selected and you are using Deepgram, not OpenAI/Groq |
| You forced a language but the transcript looks garbled | The app shows a warning when the API detected a different language than the one you forced — re-run with Auto-detect or pick the matching language |
| The transcript came back in English but the audio is German/French | The app does not translate — it transcribes in the spoken language. Choose Auto-detect or the actual language, not the language you want the output in |
| PDF shows `?` for ü/ä/ö/œ | Arial Unicode must be installed — on macOS it lives in `/System/Library/Fonts/Supplemental/` |
| Transcription fails immediately | Check your API key — invalid keys are rejected without retrying |
| Large file runs out of memory | The app streams files through ffmpeg without loading them into RAM; ensure ffmpeg is installed and on `PATH` |
| Startup error: "ffmpeg and ffprobe are required" | Install ffmpeg: `brew install ffmpeg` (macOS) or `sudo apt install ffmpeg` (Debian/Ubuntu) |
| Duplicate words at chunk boundaries | Handled automatically by the deduplication step |