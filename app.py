"""
Transcriber — Audio-to-Text Transcription App
==============================================
A Streamlit app that transcribes audio files using either a local
mlx-whisper model (optimized for Apple Silicon) or cloud APIs
(OpenAI Whisper / Groq).

Run with: uv run streamlit run app.py
"""

import hashlib
import os
import tempfile
import logging
import streamlit as st

from transcriber import audio_processor
from transcriber import cloud_engine
from transcriber import exporter
from transcriber import text_processor


def _compute_file_hash(file_buffer) -> str:
    """Compute a fast hash of the file buffer to detect changes.

    Uses first 64KB + last 64KB + file size for speed on large files.
    """
    hasher = hashlib.md5()
    data = file_buffer.getvalue()

    # Hash first 64KB
    hasher.update(data[:65536])
    # Hash last 64KB
    hasher.update(data[-65536:])
    # Include file size to differentiate files with same head/tail
    hasher.update(str(len(data)).encode())

    return hasher.hexdigest()


def _get_cached_upload_path(uploaded_file) -> str:
    """Get or create a cached temp file path for the uploaded file.

    Uses session state to cache the temp file path based on file content hash.
    This prevents writing duplicate temp files on every Streamlit re-run.
    """
    file_hash = _compute_file_hash(uploaded_file)
    cache_key = f"_upload_cache_{file_hash}"

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


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Transcriber",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — "Studio Noir" Design System ─────────────────────────────────

st.markdown("""
<style>
    /* ═══════════════════════════════════════════════════════════════════════
       STUDIO NOIR — Production-Grade Design System
       A sophisticated audio studio aesthetic with warm amber accents
       ═══════════════════════════════════════════════════════════════════════ */

    /* ── Google Fonts Import ─────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&family=Source+Sans+3:wght@300;400;500;600&display=swap');

    /* ── CSS Variables ───────────────────────────────────────────────────── */
    :root {
        /* Core palette */
        --noir-bg: #0a0a0c;
        --noir-surface: #141418;
        --noir-elevated: #1c1c22;
        --noir-border: #2a2a32;
        --noir-border-subtle: #1e1e24;

        /* Warm amber accent (audio waveform inspired) */
        --amber-50: #fffbeb;
        --amber-100: #fef3c7;
        --amber-200: #fde68a;
        --amber-300: #fcd34d;
        --amber-400: #fbbf24;
        --amber-500: #f59e0b;
        --amber-600: #d97706;

        /* Text hierarchy */
        --text-primary: #fafafa;
        --text-secondary: #a1a1aa;
        --text-muted: #71717a;

        /* Accent gradient */
        --gradient-amber: linear-gradient(135deg, #fbbf24 0%, #f59e0b 50%, #d97706 100%);
        --gradient-amber-soft: linear-gradient(135deg, rgba(251,191,36,0.15) 0%, rgba(217,119,6,0.08) 100%);

        /* Typography */
        --font-display: 'Playfair Display', Georgia, serif;
        --font-body: 'Source Sans 3', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-mono: 'JetBrains Mono', 'SF Mono', Consolas, monospace;

        /* Spacing scale */
        --space-xs: 0.25rem;
        --space-sm: 0.5rem;
        --space-md: 1rem;
        --space-lg: 1.5rem;
        --space-xl: 2rem;
        --space-2xl: 3rem;

        /* Shadows */
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.4);
        --shadow-md: 0 4px 12px rgba(0,0,0,0.5);
        --shadow-lg: 0 8px 32px rgba(0,0,0,0.6);
        --shadow-glow: 0 0 20px rgba(251,191,36,0.15);

        /* Transitions */
        --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
        --transition-smooth: 300ms cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ── Base Styles ─────────────────────────────────────────────────────── */
    .stApp {
        background: var(--noir-bg);
        color: var(--text-primary);
        font-family: var(--font-body);
    }

    /* Main container */
    .main .block-container {
        padding: var(--space-xl) var(--space-xl) var(--space-2xl);
        max-width: 1200px;
    }

    /* ── Typography ──────────────────────────────────────────────────────── */
    h1, h2, h3 {
        font-family: var(--font-display);
        letter-spacing: -0.02em;
    }

    h1 {
        font-size: 2.75rem !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        margin-bottom: 0 !important;
        position: relative;
        display: inline-block;
    }

    h2 {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }

    h3 {
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }

    h4 {
        font-family: var(--font-body);
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: var(--space-md) !important;
    }

    p, span, label, .stMarkdown {
        font-family: var(--font-body);
        color: var(--text-secondary);
        line-height: 1.6;
    }

    /* Custom subtitle class */
    .studio-subtitle {
        font-family: var(--font-body);
        font-size: 1.125rem;
        font-weight: 300;
        color: var(--text-muted);
        margin-top: -0.25rem;
        margin-bottom: var(--space-xl);
        letter-spacing: 0.02em;
    }

    /* ── Hero Section ────────────────────────────────────────────────────── */
    .studio-hero {
        position: relative;
        padding: var(--space-lg) 0 var(--space-xl);
        margin-bottom: var(--space-lg);
    }

    .studio-hero::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100px;
        right: -100px;
        height: 100%;
        background: radial-gradient(ellipse 80% 50% at 50% -20%, rgba(251,191,36,0.08) 0%, transparent 70%);
        pointer-events: none;
    }

    .studio-logo {
        display: inline-flex;
        align-items: center;
        gap: var(--space-md);
    }

    .studio-icon {
        width: 48px;
        height: 48px;
        background: var(--gradient-amber);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        box-shadow: var(--shadow-glow);
    }

    .studio-title {
        font-family: var(--font-display);
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
    }

    .studio-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(251,191,36,0.1);
        border: 1px solid rgba(251,191,36,0.2);
        color: var(--amber-400);
        font-family: var(--font-mono);
        font-size: 0.7rem;
        font-weight: 500;
        padding: 4px 10px;
        border-radius: 100px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* ── Cards & Surfaces ────────────────────────────────────────────────── */
    .studio-card {
        background: var(--noir-surface);
        border: 1px solid var(--noir-border);
        border-radius: 16px;
        padding: var(--space-lg);
        transition: all var(--transition-smooth);
    }

    .studio-card:hover {
        border-color: var(--noir-border);
        box-shadow: var(--shadow-md);
    }

    .studio-card-elevated {
        background: var(--noir-elevated);
        border: 1px solid var(--noir-border);
        border-radius: 16px;
        padding: var(--space-lg);
        box-shadow: var(--shadow-md);
    }

    /* Glass morphism card */
    .studio-glass {
        background: rgba(28,28,34,0.8);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: var(--space-lg);
    }

    /* ── Upload Zone ─────────────────────────────────────────────────────── */
    .studio-upload-zone {
        background: linear-gradient(135deg, var(--noir-surface) 0%, var(--noir-elevated) 100%);
        border: 2px dashed var(--noir-border);
        border-radius: 20px;
        padding: var(--space-xl);
        text-align: center;
        transition: all var(--transition-smooth);
        position: relative;
        overflow: hidden;
    }

    .studio-upload-zone::before {
        content: '';
        position: absolute;
        inset: 0;
        background: var(--gradient-amber-soft);
        opacity: 0;
        transition: opacity var(--transition-smooth);
    }

    .studio-upload-zone:hover {
        border-color: var(--amber-500);
    }

    .studio-upload-zone:hover::before {
        opacity: 1;
    }

    /* ── Metrics Display ─────────────────────────────────────────────────── */
    .studio-metric {
        background: var(--noir-surface);
        border: 1px solid var(--noir-border-subtle);
        border-radius: 12px;
        padding: var(--space-md) var(--space-lg);
        text-align: center;
    }

    .studio-metric-value {
        font-family: var(--font-mono);
        font-size: 1.25rem;
        font-weight: 500;
        color: var(--text-primary);
        margin-bottom: 2px;
    }

    .studio-metric-label {
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* ── Streamlit Metric Override ───────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: var(--noir-surface);
        border: 1px solid var(--noir-border-subtle);
        border-radius: 12px;
        padding: var(--space-md);
    }

    [data-testid="stMetricLabel"] {
        font-family: var(--font-body);
        font-size: 0.75rem !important;
        font-weight: 500;
        color: var(--text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    [data-testid="stMetricValue"] {
        font-family: var(--font-mono) !important;
        font-size: 1.25rem !important;
        font-weight: 500;
        color: var(--text-primary) !important;
    }

    /* ── Progress Bar ────────────────────────────────────────────────────── */
    .stProgress > div > div {
        background: var(--gradient-amber) !important;
        border-radius: 100px;
        box-shadow: var(--shadow-glow);
    }

    .stProgress > div {
        background: var(--noir-surface) !important;
        border-radius: 100px;
    }

    /* ── Sidebar ─────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: var(--noir-surface);
        border-right: 1px solid var(--noir-border-subtle);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: var(--space-xl);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-family: var(--font-display) !important;
        color: var(--text-primary) !important;
    }

    [data-testid="stSidebar"] h2 {
        font-size: 1.25rem !important;
        margin-bottom: var(--space-lg) !important;
    }

    [data-testid="stSidebar"] h3 {
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted) !important;
        margin-top: var(--space-lg) !important;
        margin-bottom: var(--space-sm) !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: var(--text-secondary) !important;
    }

    [data-testid="stSidebar"] .stCaption {
        color: var(--text-muted) !important;
        font-size: 0.8rem;
    }

    /* Sidebar section divider */
    [data-testid="stSidebar"] hr {
        border: none;
        height: 1px;
        background: var(--noir-border);
        margin: var(--space-lg) 0;
    }

    /* ── Input Fields ────────────────────────────────────────────────────── */
    .stTextInput > div > div > input,
    .stTextArea textarea {
        font-family: var(--font-mono) !important;
        font-size: 0.9rem !important;
        color: var(--text-primary) !important;
        background: var(--noir-bg) !important;
        border: 1px solid var(--noir-border) !important;
        border-radius: 10px !important;
        padding: var(--space-md) !important;
        transition: all var(--transition-fast);
    }

    .stTextInput > div > div > input:focus,
    .stTextArea textarea:focus {
        border-color: var(--amber-500) !important;
        box-shadow: 0 0 0 2px rgba(251,191,36,0.15) !important;
    }

    .stTextInput > div > div > input::placeholder,
    .stTextArea textarea::placeholder {
        color: var(--text-muted) !important;
    }

    /* Select boxes */
    .stSelectbox > div > div {
        background: var(--noir-bg) !important;
        border: 1px solid var(--noir-border) !important;
        border-radius: 10px !important;
    }

    .stSelectbox > div > div > div {
        color: var(--text-primary) !important;
    }

    /* ── Buttons ─────────────────────────────────────────────────────────── */
    .stButton > button {
        font-family: var(--font-body);
        font-weight: 600;
        letter-spacing: 0.02em;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        transition: all var(--transition-fast);
    }

    /* Primary button */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {
        background: var(--gradient-amber) !important;
        color: var(--noir-bg) !important;
        border: none !important;
        box-shadow: var(--shadow-sm), var(--shadow-glow);
    }

    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-md), 0 0 30px rgba(251,191,36,0.25);
    }

    /* Secondary button */
    .stButton > button[kind="secondary"],
    .stButton > button[data-testid="baseButton-secondary"] {
        background: transparent !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--noir-border) !important;
    }

    .stButton > button[kind="secondary"]:hover,
    .stButton > button[data-testid="baseButton-secondary"]:hover {
        background: var(--noir-elevated) !important;
        border-color: var(--amber-500) !important;
        color: var(--amber-400) !important;
    }

    /* Download buttons */
    .stDownloadButton > button {
        font-family: var(--font-body) !important;
        font-weight: 600 !important;
        background: var(--noir-surface) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--noir-border) !important;
        border-radius: 10px !important;
        transition: all var(--transition-fast);
    }

    .stDownloadButton > button:hover {
        background: var(--noir-elevated) !important;
        border-color: var(--amber-500) !important;
        color: var(--amber-400) !important;
        box-shadow: var(--shadow-glow);
    }

    /* ── File Uploader ───────────────────────────────────────────────────── */
    [data-testid="stFileUploader"] {
        background: var(--noir-surface);
        border: 2px dashed var(--noir-border);
        border-radius: 16px;
        padding: var(--space-lg);
        transition: all var(--transition-smooth);
    }

    [data-testid="stFileUploader"]:hover {
        border-color: var(--amber-500);
        background: linear-gradient(135deg, var(--noir-surface) 0%, rgba(251,191,36,0.03) 100%);
    }

    [data-testid="stFileUploader"] section {
        padding: var(--space-md);
    }

    [data-testid="stFileUploader"] button {
        background: var(--gradient-amber) !important;
        color: var(--noir-bg) !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
    }

    /* ── Expander ────────────────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        font-family: var(--font-body) !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        background: var(--noir-surface) !important;
        border: 1px solid var(--noir-border) !important;
        border-radius: 12px !important;
        padding: var(--space-md) var(--space-lg) !important;
    }

    .streamlit-expanderContent {
        background: var(--noir-surface) !important;
        border: 1px solid var(--noir-border) !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
        padding: var(--space-lg) !important;
    }

    /* ── Checkbox & Radio ────────────────────────────────────────────────── */
    .stCheckbox > label,
    .stRadio > label {
        color: var(--text-secondary) !important;
    }

    .stCheckbox > label > span:first-child,
    .stRadio > label > div:first-child {
        border-color: var(--noir-border) !important;
    }

    .stRadio > div {
        gap: var(--space-md);
    }

    .stRadio > div > label {
        background: var(--noir-surface) !important;
        border: 1px solid var(--noir-border) !important;
        border-radius: 8px !important;
        padding: var(--space-sm) var(--space-md) !important;
        transition: all var(--transition-fast);
    }

    .stRadio > div > label:hover {
        border-color: var(--amber-500) !important;
    }

    .stRadio > div > label[data-checked="true"] {
        background: rgba(251,191,36,0.1) !important;
        border-color: var(--amber-500) !important;
        color: var(--amber-400) !important;
    }

    /* ── Alerts ──────────────────────────────────────────────────────────── */
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
    }

    [data-testid="stNotification"] {
        background: var(--noir-surface) !important;
        border-left: 3px solid var(--amber-500) !important;
        border-radius: 8px !important;
    }

    /* Success alert */
    .stSuccess {
        background: rgba(34,197,94,0.1) !important;
        border-left: 3px solid #22c55e !important;
    }

    /* Info alert */
    .stInfo {
        background: rgba(59,130,246,0.1) !important;
        border-left: 3px solid #3b82f6 !important;
    }

    /* Warning alert */
    .stWarning {
        background: rgba(251,191,36,0.1) !important;
        border-left: 3px solid var(--amber-500) !important;
    }

    /* Error alert */
    .stError {
        background: rgba(239,68,68,0.1) !important;
        border-left: 3px solid #ef4444 !important;
    }

    /* ── Divider ─────────────────────────────────────────────────────────── */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent 0%, var(--noir-border) 20%, var(--noir-border) 80%, transparent 100%) !important;
        margin: var(--space-xl) 0 !important;
    }

    /* ── Editor Preview Box ──────────────────────────────────────────────── */
    .studio-preview {
        background: var(--noir-bg);
        border: 1px solid var(--noir-border);
        border-radius: 12px;
        padding: var(--space-lg);
        max-height: 450px;
        overflow-y: auto;
        line-height: 1.9;
        font-family: var(--font-body);
        font-size: 1rem;
        color: var(--text-secondary);
    }

    .studio-preview strong {
        color: var(--amber-400);
        font-weight: 600;
    }

    .studio-preview mark {
        background: rgba(251,191,36,0.3);
        color: var(--text-primary);
        padding: 2px 4px;
        border-radius: 4px;
    }

    /* ── Reading Stats ───────────────────────────────────────────────────── */
    .studio-stats {
        display: inline-flex;
        align-items: center;
        gap: var(--space-lg);
        font-family: var(--font-mono);
        font-size: 0.8rem;
        color: var(--text-muted);
        padding: var(--space-sm) 0;
    }

    .studio-stats span {
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* ── Section Headers ─────────────────────────────────────────────────── */
    .studio-section-header {
        display: flex;
        align-items: center;
        gap: var(--space-md);
        margin-bottom: var(--space-lg);
    }

    .studio-section-icon {
        width: 32px;
        height: 32px;
        background: rgba(251,191,36,0.1);
        border: 1px solid rgba(251,191,36,0.2);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
    }

    .studio-section-title {
        font-family: var(--font-display);
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
    }

    /* ── Caption Override ────────────────────────────────────────────────── */
    .stCaption {
        color: var(--text-muted) !important;
        font-size: 0.85rem !important;
    }

    /* ── Scrollbar Styling ───────────────────────────────────────────────── */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--noir-bg);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: var(--noir-border);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-muted);
    }

    /* ── Animation Keyframes ─────────────────────────────────────────────── */
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 15px rgba(251,191,36,0.2); }
        50% { box-shadow: 0 0 25px rgba(251,191,36,0.35); }
    }

    @keyframes fade-in {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .animate-fade-in {
        animation: fade-in 0.4s ease-out forwards;
    }

    /* ── Waveform Decoration ─────────────────────────────────────────────── */
    .studio-waveform {
        display: flex;
        align-items: center;
        gap: 3px;
        height: 24px;
    }

    .studio-waveform span {
        width: 3px;
        background: var(--amber-500);
        border-radius: 2px;
        animation: wave 1s ease-in-out infinite;
    }

    .studio-waveform span:nth-child(1) { height: 40%; animation-delay: 0s; }
    .studio-waveform span:nth-child(2) { height: 70%; animation-delay: 0.1s; }
    .studio-waveform span:nth-child(3) { height: 100%; animation-delay: 0.2s; }
    .studio-waveform span:nth-child(4) { height: 60%; animation-delay: 0.3s; }
    .studio-waveform span:nth-child(5) { height: 80%; animation-delay: 0.4s; }

    @keyframes wave {
        0%, 100% { transform: scaleY(1); }
        50% { transform: scaleY(0.5); }
    }
</style>
""", unsafe_allow_html=True)


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
    st.markdown("### Language")

    # Standard languages list (simplified for cloud)
    LANGUAGES = {
        "Auto-detect": None,
        "English": "en",
        "German": "de",
        "French": "fr",
        "Spanish": "es",
        "Italian": "it",
    }

    language_name = st.selectbox(
        "Audio language",
        list(LANGUAGES.keys()),
        help="Choose the language or let the AI detect it automatically.",
        label_visibility="collapsed",
    )
    language_code = LANGUAGES[language_name]

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
        info = audio_processor.get_audio_info(audio_file_path)
        st.session_state.audio_info = info
        
        # Display audio info
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("⏱️ Duration", info["duration_formatted"])
        col2.metric("📦 Size", f"{info['file_size_mb']:.1f} MB")
        col3.metric("🔊 Channels", info["channels"])
        col4.metric("📊 Sample Rate", f"{info['sample_rate']} Hz")
        
        needs_split = audio_processor.needs_chunking(audio_file_path)
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
        
        try:
            # Step 1: Chunk the audio
            status_text.markdown("⏳ **Preparing audio...**")
            chunk_paths = audio_processor.chunk_audio(audio_file_path, progress_callback=update_progress)

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

            # Cleanup temp chunks
            audio_processor.cleanup_chunks(chunk_paths, audio_file_path)

            progress_bar.progress(1.0)
            status_text.markdown("✅ **Transcription complete!**")

            # Show detected language if auto-detect was used
            if detected_language and not language_code:
                st.info(f"Detected language: **{detected_language.upper()}**")

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

            # Provide actionable error messages
            error_msg = str(e)
            if "401" in error_msg or "Unauthorized" in error_msg.lower() or "invalid api key" in error_msg.lower():
                st.error("Invalid API key. Please check your API key in the sidebar.")
            elif "429" in error_msg or "rate limit" in error_msg.lower():
                st.error("Rate limit exceeded. Please wait a moment and try again.")
            elif "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                st.error("Connection error. Please check your internet connection and try again.")
            else:
                st.error(f"Transcription failed: {error_msg}")

            logger.exception("Transcription error")

# Cleanup temp upload
if temp_upload_path and os.path.exists(temp_upload_path):
    # Don't delete yet — might still be needed for re-transcription
    pass


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

        # Prepare display text with search highlighting
        display_text = st.session_state.transcript

        # Apply search highlighting if query provided
        if search_query:
            display_text = text_processor.search_and_highlight(display_text, search_query)

        # Convert markdown bold to HTML bold, italic to HTML italic
        display_html = display_text.replace("**", "<strong>").replace("_", "<em>")
        # Fix paired tags (every other occurrence)
        import re
        display_html = re.sub(r'<strong>([^<]+)<strong>', r'<strong>\1</strong>', display_html)
        display_html = re.sub(r'<em>([^<]+)<em>', r'<em style="color: #9ca3af;">\1</em>', display_html)

        # Render formatted preview
        st.markdown(
            f'<div class="studio-preview">'
            f'{display_html.replace(chr(10), "<br>")}'
            f'</div>',
            unsafe_allow_html=True,
        )
        edited_text = st.session_state.transcript

    # Reading stats
    stats = text_processor.get_reading_stats(edited_text)
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
