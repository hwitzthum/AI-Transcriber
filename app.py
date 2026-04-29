"""
Transcriber — Audio-to-Text Transcription App
==============================================
A Streamlit app that transcribes audio files using either a local
mlx-whisper model (optimized for Apple Silicon) or cloud APIs
(OpenAI Whisper / Groq).

Run with: uv run streamlit run app.py
"""

import os
import tempfile
import logging
from pathlib import Path

import streamlit as st

from transcriber import audio_processor
from transcriber import cloud_engine
from transcriber import exporter
from transcriber import text_processor
from transcriber.language import MULTI_SENTINEL, normalize_language_to_iso


@st.cache_data(show_spinner=False)
def _cached_audio_info(file_path: str, mtime: float) -> dict:
    """Streamlit-cached wrapper around audio_processor.get_audio_info.

    The ``mtime`` argument participates in the cache key so the cache is
    invalidated when the on-disk file changes; it is otherwise unused.
    Without this, ffprobe spawns a subprocess on every Streamlit rerun
    (every checkbox click, search keystroke, etc.) — measurable lag on
    long editing sessions.
    """
    return audio_processor.get_audio_info(file_path)


@st.cache_data(show_spinner=False)
def _cached_needs_chunking(file_path: str, mtime: float, max_bytes: int) -> bool:
    """Streamlit-cached wrapper around audio_processor.needs_chunking."""
    return audio_processor.needs_chunking(file_path, max_bytes=max_bytes)


def _get_cached_upload_path(uploaded_file) -> str:
    """Get or create a cached temp file path for the uploaded file.

    Uses session state to cache the temp file path based on file content hash.
    This prevents writing duplicate temp files on every Streamlit re-run.
    The original filename is mixed into the cache key alongside the
    head/tail/size hash so two files of identical size whose first and
    last 64 KB happen to match (rare but possible — same template, same
    container) don't silently share a cached temp path.
    """
    file_hash = audio_processor.compute_upload_hash(uploaded_file)
    cache_key = f"_upload_cache_{uploaded_file.name}_{file_hash}"

    # Check if we already have a cached path for this exact file
    if cache_key in st.session_state:
        cached_path = st.session_state[cache_key]
        # Verify the cached file still exists
        if os.path.exists(cached_path):
            return cached_path

    # Clean up old cached uploads (only keep current file)
    old_keys = [k for k in st.session_state if k.startswith("_upload_cache_")]
    for old_key in old_keys:
        old_path = st.session_state.get(old_key)
        if old_path and os.path.exists(old_path) and old_key != cache_key:
            try:
                os.unlink(old_path)
            except OSError:
                pass
        if old_key != cache_key:
            del st.session_state[old_key]

    # Write new temp file
    suffix = os.path.splitext(uploaded_file.name)[1]
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()

    # Cache the path
    st.session_state[cache_key] = tmp.name

    return tmp.name


# Configure logging
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Cached export functions
@st.cache_data(show_spinner=False)
def _cached_export_docx(text: str, title: str) -> bytes:
    return exporter.export_docx(text, title=title)


@st.cache_data(show_spinner=False)
def _cached_export_pdf(text: str, title: str) -> bytes:
    return exporter.export_pdf(text, title=title)


# Preview-mode helpers run on every Streamlit rerun. The search box in the
# preview re-runs the whole script per keystroke, which used to re-execute
# the multi-pass regex in render_transcript_html and the markdown-stripping
# regex in get_reading_stats over the full transcript every time. Caching
# on the (text, query) tuple cuts that to one execution per unique input.
@st.cache_data(show_spinner=False)
def _cached_render_transcript_html(text: str, search_query: str) -> str:
    return text_processor.render_transcript_html(text, search_query=search_query)


@st.cache_data(show_spinner=False)
def _cached_reading_stats(text: str) -> dict:
    return text_processor.get_reading_stats(text)


# Hard cap on uploaded file size (in bytes). Mirrors the
# ``maxUploadSize = 2000`` (MB) value in .streamlit/config.toml — Streamlit
# enforces it at the protocol layer for the upload widget, but the gate
# here gives a clean app-level error if that setting is ever loosened or
# bypassed and avoids ever writing a bigger temp file to disk.
_MAX_UPLOAD_BYTES = 2000 * 1024 * 1024


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Transcriber",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Startup gate: ffmpeg/ffprobe are hard prerequisites. If they aren't on
# PATH, every audio operation in the app will fail later with an opaque
# error — surface it immediately as a clean Streamlit error instead so
# the user knows what to install.
try:
    audio_processor.require_ffmpeg()
except RuntimeError as _ffmpeg_err:
    st.error(f"❌ {_ffmpeg_err}")
    st.stop()

# ── Custom CSS — "Studio Noir" Design System ─────────────────────────────────

# CSS lives in assets/styles.css. Keeping ~700 lines of stylesheet
# inside this Python file made app.py impractical to navigate and
# excluded the styles from any CSS-aware tooling.
_STYLES_PATH = Path(__file__).parent / "assets" / "styles.css"
st.markdown(_STYLES_PATH.read_text(encoding="utf-8"), unsafe_allow_html=True)


# ── Session state ────────────────────────────────────────────────────────────

if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "is_transcribing" not in st.session_state:
    st.session_state.is_transcribing = False
if "audio_info" not in st.session_state:
    st.session_state.audio_info = None



with st.sidebar:
    # Sidebar branding
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
            <div style="
                width: 36px;
                height: 36px;
                background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 50%, #d97706 100%);
                border-radius: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.1rem;
                box-shadow: 0 0 20px rgba(251,191,36,0.2);
            ">🎙️</div>
            <span style="
                font-family: 'Playfair Display', Georgia, serif;
                font-size: 1.4rem;
                font-weight: 700;
                color: #fafafa;
            ">Transcriber</span>
        </div>
        <p style="
            font-family: 'Source Sans 3', sans-serif;
            font-size: 0.8rem;
            color: #71717a;
            margin: 0;
            letter-spacing: 0.02em;
        ">AI-Powered Audio Transcription</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Provider")
    cloud_provider = st.selectbox(
        "Transcription Service",
        list(cloud_engine.PROVIDERS.keys()),
        help="Choose a cloud transcription provider.",
        label_visibility="collapsed",
    )

    provider_config = cloud_engine.PROVIDERS[cloud_provider]
    st.caption(f"{provider_config['description']}")

    # API Key Input
    api_key_key = f"{cloud_provider}_key"  # Unique key per provider

    st.markdown(
        f'<p style="font-size: 0.8rem; color: #a1a1aa; margin-bottom: 6px;">'
        f'{cloud_provider.split()[0]} API Key</p>',
        unsafe_allow_html=True,
    )
    api_key = st.text_input(
        f"{cloud_provider.split()[0]} API Key",
        type="password",
        help="Your API key. Not stored — session only.",
        key=api_key_key,
        label_visibility="collapsed",
    )

    # Diarization Option (Deepgram only)
    enable_diarization = False
    if "Deepgram" in cloud_provider:
        st.markdown("")  # spacing
        enable_diarization = st.checkbox(
            "Speaker Diarization",
            value=True,
            help="Identify different speakers (Speaker 0, Speaker 1, etc.)",
        )
        if "Multilingual" in cloud_provider:
            st.markdown(
                '<p style="font-size: 0.75rem; color: #71717a; margin-top: 4px;">'
                '✦ Nova-3 handles mixed-language audio automatically</p>',
                unsafe_allow_html=True,
            )

    # Language selection
    st.divider()
    st.markdown("### Spoken language")

    # Standard languages list (simplified for cloud)
    LANGUAGES = {
        "Auto-detect (recommended)": None,
        "English": "en",
        "German": "de",
        "French": "fr",
        "Spanish": "es",
        "Italian": "it",
    }

    language_name = st.selectbox(
        "Spoken language in the audio",
        list(LANGUAGES.keys()),
        help=(
            "The language *spoken in your recording* — not the output language. "
            "The transcript is always produced in the original language; nothing is translated. "
            "If you pick the wrong language here, the transcript will be garbled. "
            "When in doubt, leave this on Auto-detect. "
            "For audio that mixes multiple languages, switch the provider above to "
            "Deepgram Nova-3 (Multilingual)."
        ),
        label_visibility="collapsed",
    )
    language_code = LANGUAGES[language_name]
    st.markdown(
        '<p style="font-size: 0.75rem; color: #71717a; margin-top: 4px;">'
        "Transcribed in the original language — never translated. "
        "For mixed-language audio, use Nova-3 (Multilingual)."
        "</p>",
        unsafe_allow_html=True,
    )

    # Footer
    st.divider()
    st.markdown("""
    <div style="margin-top: 1rem;">
        <p style="
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            color: #52525b;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 4px;
        ">System</p>
        <div style="
            display: flex;
            align-items: center;
            gap: 8px;
        ">
            <span style="
                width: 6px;
                height: 6px;
                background: #22c55e;
                border-radius: 50%;
                box-shadow: 0 0 8px #22c55e;
            "></span>
            <span style="
                font-family: 'Source Sans 3', sans-serif;
                font-size: 0.8rem;
                color: #a1a1aa;
            ">Cloud Mode Active</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Main area ────────────────────────────────────────────────────────────────

# Hero Section
st.markdown("""
<div class="studio-hero">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
        <div class="studio-logo">
            <div class="studio-icon">🎙️</div>
            <div>
                <h1 class="studio-title">Transcriber</h1>
                <p class="studio-subtitle" style="margin: 0;">Transform audio into accurate, readable text</p>
            </div>
        </div>
        <div class="studio-badge">
            <div class="studio-waveform">
                <span></span><span></span><span></span><span></span><span></span>
            </div>
            AI-Powered
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Input section ────────────────────────────────────────────────────────────

col_upload, col_path = st.columns(2)

with col_upload:
    st.markdown("""
    <div class="studio-section-header">
        <div class="studio-section-icon">📁</div>
        <p class="studio-section-title">Upload File</p>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop your audio or video file here",
        type=["mp3", "wav", "m4a", "flac", "ogg", "wma", "aac", "opus", "webm", "mp4", "mov", "avi", "mkv"],
        help="Supports MP3, WAV, M4A, FLAC, MP4, MOV, and more.",
        label_visibility="collapsed",
    )

with col_path:
    st.markdown("""
    <div class="studio-section-header">
        <div class="studio-section-icon">📂</div>
        <p class="studio-section-title">Or Enter Path</p>
    </div>
    """, unsafe_allow_html=True)
    file_path_input = st.text_input(
        "Absolute path to audio/video file",
        placeholder="/path/to/your/audio.mp3",
        help="Enter the full path to a local audio or video file.",
        label_visibility="collapsed",
    )


# ── Resolve the audio source ────────────────────────────────────────────────

audio_file_path = None
temp_upload_path = None

if uploaded_file is not None:
    # App-layer guard against oversize uploads. Belt-and-suspenders next
    # to the protocol-level Streamlit cap — if that cap is ever loosened
    # the user gets a clean error instead of an unbounded temp-file write.
    if uploaded_file.size and uploaded_file.size > _MAX_UPLOAD_BYTES:
        cap_mb = _MAX_UPLOAD_BYTES / (1024 * 1024)
        size_mb = uploaded_file.size / (1024 * 1024)
        st.error(
            f"❌ File is too large ({size_mb:.0f} MB). The current limit is "
            f"{cap_mb:.0f} MB."
        )
        st.stop()
    # Use cached temp file to avoid redundant writes on Streamlit re-runs
    audio_file_path = _get_cached_upload_path(uploaded_file)
    temp_upload_path = audio_file_path
    file_size_mb = uploaded_file.size / (1024 * 1024) if uploaded_file.size > 0 else 0
    st.success(f"✅ Loaded: **{uploaded_file.name}** ({file_size_mb:.1f} MB)")
elif file_path_input.strip():
    valid, msg = audio_processor.validate_file(file_path_input.strip())
    if valid:
        audio_file_path = file_path_input.strip()
        st.success(f"✅ File found: **{os.path.basename(audio_file_path)}**")
    else:
        st.error(f"❌ {msg}")


# ── Audio info & Transcribe button ──────────────────────────────────────────

if audio_file_path:
    try:
        # mtime participates in the cache key — file change → cache miss.
        file_mtime = os.path.getmtime(audio_file_path)
        info = _cached_audio_info(audio_file_path, file_mtime)
        st.session_state.audio_info = info

        # Display audio info
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("⏱️ Duration", info["duration_formatted"])
        col2.metric("📦 Size", f"{info['file_size_mb']:.1f} MB")
        col3.metric("🔊 Channels", info["channels"])
        col4.metric("📊 Sample Rate", f"{info['sample_rate']} Hz")

        # Use the chosen provider's upload limit so the "needs chunking" hint
        # only fires when this specific provider would actually need it.
        provider_max_chunk_bytes = cloud_engine.get_max_chunk_bytes(cloud_provider)
        needs_split = _cached_needs_chunking(
            audio_file_path,
            file_mtime,
            provider_max_chunk_bytes,
        )
        if needs_split:
            st.info("📎 This file is large and will be split into chunks for processing.")

    except Exception as e:
        st.error(f"❌ Could not read audio file: {e}")
        audio_file_path = None


# ── Transcription ───────────────────────────────────────────────────────────

if audio_file_path:
    st.divider()
    
    if st.button("🚀 Start Transcription", type="primary", use_container_width=True, disabled=st.session_state.get("is_transcribing", False)):
        
        if not api_key or not api_key.strip():
            st.error("❌ Please enter your API key in the sidebar.")
            st.stop()
        
        st.session_state.is_transcribing = True
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total, message):
            if total > 0:
                progress_bar.progress(current / total)
            status_text.markdown(f"⏳ **{message}**")
        
        # Initialised before the try so the finally block is safe even if
        # chunk_audio() itself raises before returning a list.
        chunk_paths: list[str] = []
        try:
            # Step 1: Chunk the audio (using provider-specific upload ceiling).
            status_text.markdown("⏳ **Preparing audio...**")
            chunk_paths = audio_processor.chunk_audio(
                audio_file_path,
                progress_callback=update_progress,
                max_bytes=cloud_engine.get_max_chunk_bytes(cloud_provider),
            )

            # Step 2: Transcribe (Cloud Only)

            # Pass diarize flag only if supported (Deepgram)
            diarize_flag = enable_diarization if "Deepgram" in cloud_provider else False

            result = cloud_engine.transcribe_chunks(
                chunk_paths,
                provider=cloud_provider,
                api_key=api_key.strip(),
                language=language_code,
                progress_callback=update_progress,
                diarize=diarize_flag,
            )

            transcript = result["text"]
            failed_chunks = result.get("failed_chunks", [])
            detected_language = result.get("detected_language")
            quality_warnings = result.get("quality_warnings", [])

            # Store result
            st.session_state.transcript = transcript
            st.session_state.is_transcribing = False

            progress_bar.progress(1.0)
            status_text.markdown("✅ **Transcription complete!**")

            # Show detected language if auto-detect was used,
            # or warn the user when their forced selection disagrees with what
            # the API actually heard (the classic "I picked German but the audio
            # is French → garbled output" trap).
            if detected_language:
                detected_iso = normalize_language_to_iso(detected_language)
                # Nova-3 multilingual returns "multi" — it handled multiple
                # languages on purpose, so a single-language mismatch warning
                # would always be a false positive there.
                is_multi = detected_iso == MULTI_SENTINEL
                # When the model returns a code we don't recognise, fall back
                # to comparing the raw string so the warning still fires
                # instead of silently disappearing.
                effective_detected = detected_iso or detected_language.lower()
                if not language_code:
                    if is_multi:
                        st.info("Detected language: **multiple (multilingual)**")
                    else:
                        label = (
                            detected_iso.upper() if detected_iso else detected_language.upper()
                        )
                        st.info(f"Detected language: **{label}**")
                elif not is_multi and effective_detected != language_code:
                    display = (
                        detected_iso.upper() if detected_iso else detected_language
                    )
                    st.warning(
                        f"⚠️ You selected **{language_name}** but the audio sounds like "
                        f"**{display}**. The transcript may be garbled. "
                        "Re-run with Auto-detect, or pick the matching language."
                    )

            # Warn about any chunks that failed (graceful degradation)
            if failed_chunks:
                st.warning(
                    f"{len(failed_chunks)} chunk(s) failed and were skipped "
                    f"(chunk indices: {failed_chunks}). "
                    "The transcript may have gaps. Check your API key and connection."
                )

            # Show quality warnings (empty transcript, garbage detection, etc.)
            for warning in quality_warnings:
                # Skip per-chunk failure messages already shown above
                if "failed and was skipped" in warning:
                    continue
                st.warning(warning)

            if transcript:
                st.balloons()

        except Exception as e:
            st.session_state.is_transcribing = False
            progress_bar.empty()

            # Provide actionable error messages.
            # ``error_msg`` is what the user sees on screen, so it goes
            # through the redactor — provider SDK exceptions can embed
            # the Authorization header verbatim. The branch matching uses
            # the un-redacted lowercase string so legitimate "401" /
            # "rate limit" / "timeout" markers still classify correctly.
            raw_error = str(e)
            error_msg_lower = raw_error.lower()
            error_msg = cloud_engine.redact_secrets(raw_error)
            if "401" in raw_error or "unauthorized" in error_msg_lower or "invalid api key" in error_msg_lower:
                st.error("Invalid API key. Please check your API key in the sidebar.")
            elif "429" in raw_error or "rate limit" in error_msg_lower:
                st.error("Rate limit exceeded. Please wait a moment and try again.")
            elif "timeout" in error_msg_lower or "connection" in error_msg_lower:
                st.error("Connection error. Please check your internet connection and try again.")
            else:
                st.error(f"Transcription failed: {error_msg}")

            logger.exception("Transcription error")
        finally:
            # Always clean up chunk temp files — leaving them behind on
            # transcription failure used to accumulate hundreds of MB across
            # repeated retry attempts.
            if chunk_paths:
                audio_processor.cleanup_chunks(chunk_paths, audio_file_path)

# Note: temp_upload_path is NOT cleaned up here — the user may re-run
# transcription on the same file, and writing the upload again is the
# expensive thing we're avoiding. _get_cached_upload_path() removes the
# previous upload's temp file the next time a different file arrives.


# ── Editor & Export ─────────────────────────────────────────────────────────

if st.session_state.transcript:
    st.divider()
    st.markdown("### ✏️ Edit Transcription")

    # ── Readability Options ──────────────────────────────────────────────────
    with st.expander("📖 Readability Options", expanded=False):
        st.info(
            "**Paragraph breaks** are automatically inserted at natural pauses "
            "(>1.5 seconds of silence) when using Deepgram with speaker diarization."
        )

        enable_filler_highlight = st.checkbox(
            "Highlight filler words",
            value=False,
            help="Mark filler words (um, uh, äh, euh, like, basically) in italic for easy review",
        )

        if enable_filler_highlight:
            if st.button("Apply Filler Highlighting", type="secondary"):
                st.session_state.transcript = text_processor.highlight_filler_words(
                    st.session_state.transcript
                )
                st.rerun()

    # ── Speaker Renaming ─────────────────────────────────────────────────────
    speakers = text_processor.extract_speakers(st.session_state.transcript)
    if speakers:
        with st.expander("👥 Rename Speakers", expanded=False):
            st.caption("Replace generic speaker labels with actual names")

            # Initialize speaker names in session state
            if "speaker_names" not in st.session_state:
                st.session_state.speaker_names = {}

            cols = st.columns(min(len(speakers), 3))
            name_map = {}
            for i, speaker in enumerate(speakers):
                col_idx = i % 3
                with cols[col_idx]:
                    default_val = st.session_state.speaker_names.get(speaker, "")
                    new_name = st.text_input(
                        speaker,
                        value=default_val,
                        placeholder=f"e.g., Maria",
                        key=f"rename_{speaker}",
                    )
                    if new_name.strip():
                        name_map[speaker] = new_name.strip()

            if st.button("Apply Speaker Names", type="secondary"):
                if name_map:
                    st.session_state.speaker_names.update(name_map)
                    st.session_state.transcript = text_processor.rename_speakers(
                        st.session_state.transcript, name_map
                    )
                    st.rerun()

    # ── View Mode Toggle ─────────────────────────────────────────────────────
    col_mode, col_search = st.columns([1, 2])
    with col_mode:
        view_mode = st.radio(
            "View mode",
            ["Edit", "Preview"],
            horizontal=True,
            label_visibility="collapsed",
            help="Edit: Raw text editor | Preview: Formatted view",
        )

    # Search box (only in Preview mode)
    search_query = ""
    if view_mode == "Preview":
        with col_search:
            search_query = st.text_input(
                "🔍 Search",
                placeholder="Search transcript...",
                label_visibility="collapsed",
            )

    if view_mode == "Edit":
        st.caption("Edit the raw text below. Speaker labels use **Speaker X:** format.")
        edited_text = st.text_area(
            "Transcribed text",
            value=st.session_state.transcript,
            height=400,
            label_visibility="collapsed",
        )
        # Update session state with edits
        st.session_state.transcript = edited_text
    else:
        st.caption("Formatted preview. Switch to Edit mode to make changes.")

        display_html = _cached_render_transcript_html(
            st.session_state.transcript,
            search_query,
        )

        # Render formatted preview
        st.markdown(
            f'<div class="studio-preview">'
            f'{display_html.replace(chr(10), "<br>")}'
            f'</div>',
            unsafe_allow_html=True,
        )
        edited_text = st.session_state.transcript

    # Reading stats
    stats = _cached_reading_stats(edited_text)
    reading_time = stats["reading_time_minutes"]
    if reading_time < 1:
        time_str = f"{int(reading_time * 60)}s read"
    else:
        time_str = f"{reading_time:.1f}m read"
    st.caption(f"📝 {stats['word_count']:,} words · {stats['char_count']:,} characters · ⏱️ {time_str}")

    # ── Export Section ───────────────────────────────────────────────────────
    st.markdown("### 📥 Download")

    col_docx, col_pdf = st.columns(2)

    with col_docx:
        if edited_text.strip():
            docx_bytes = _cached_export_docx(edited_text, "Transcription")
            st.download_button(
                label="📄 Download DOCX",
                data=docx_bytes,
                file_name="transcription.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )

    with col_pdf:
        if edited_text.strip():
            try:
                pdf_bytes = _cached_export_pdf(edited_text, "Transcription")
                st.download_button(
                    label="📕 Download PDF",
                    data=pdf_bytes,
                    file_name="transcription.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"PDF export issue: {e}. Try DOCX instead.")
