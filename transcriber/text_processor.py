"""Text processing utilities for improving transcript readability."""

import html
import re
from typing import Optional


# Common filler words in multiple languages
FILLER_WORDS = {
    # English
    "um", "uh", "uhm", "umm", "er", "ah", "like", "basically",
    "actually", "literally", "honestly", "obviously", "anyway",
    # German
    "äh", "ähm", "öhm", "halt", "quasi", "sozusagen", "irgendwie",
    # French
    "euh", "heu", "ben", "bah", "genre", "quoi", "voilà",
}

# Regex pattern to match filler words (case-insensitive, whole words only)
_FILLER_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(w) for w in FILLER_WORDS) + r')\b',
    re.IGNORECASE
)

# Pattern to detect speaker labels
_SPEAKER_LABEL_PATTERN = re.compile(r'\*\*Speaker (\d+):\*\*')

# Inline ``[HH:MM:SS]`` markers prepended by the timestamped-transcript
# path. Anchored with word boundaries so it doesn't match arbitrary
# bracketed content inside the speaker text.
_TIMESTAMP_MARKER_PATTERN = re.compile(r'\[(\d{2}:\d{2}:\d{2})\]')


def highlight_filler_words(text: str) -> str:
    """
    Wrap filler words in italic markers for visual distinction.

    Filler words like "um", "uh", "like" are wrapped in *word* format
    which renders as italic in markdown and can be styled in exports.

    Args:
        text: The transcript text.

    Returns:
        Text with filler words wrapped in italic markers.
    """
    def _wrap_filler(match: re.Match) -> str:
        word = match.group(1)
        # Use a special marker that won't conflict with speaker labels
        return f"_{word}_"

    return _FILLER_PATTERN.sub(_wrap_filler, text)


def extract_speakers(text: str) -> list[str]:
    """
    Extract all unique speaker labels from the transcript.

    Args:
        text: The transcript text.

    Returns:
        List of speaker labels like ["Speaker 0", "Speaker 1"].
    """
    matches = _SPEAKER_LABEL_PATTERN.findall(text)
    unique_speakers = sorted(set(matches), key=int)
    return [f"Speaker {num}" for num in unique_speakers]


def rename_speakers(text: str, name_map: dict[str, str]) -> str:
    """
    Replace speaker labels with custom names.

    Args:
        text: The transcript text.
        name_map: Mapping from speaker label to custom name,
                  e.g., {"Speaker 0": "Maria", "Speaker 1": "John"}

    Returns:
        Text with renamed speakers.
    """
    result = text
    for old_name, new_name in name_map.items():
        if new_name and new_name.strip():
            # Replace in speaker label format
            old_pattern = f"**{old_name}:**"
            new_label = f"**{new_name}:**"
            result = result.replace(old_pattern, new_label)
    return result


def render_transcript_html(text: str, search_query: Optional[str] = None) -> str:
    """
    Convert a transcript (containing markdown-style **bold** and _italic_
    markers, plus optional `**Speaker N:**` labels) to safe HTML for preview.

    The raw transcript is HTML-escaped first so any markup in the underlying
    text (e.g. a poisoned API response containing <script>) is rendered as
    literal text. Search highlighting and markdown→HTML conversion run on
    the escaped text and inject only their own known-safe tags.

    Args:
        text: Transcript text (may contain ** and _ markers).
        search_query: Optional search query to highlight via <mark> tags.

    Returns:
        HTML-safe string ready for embedding in `unsafe_allow_html` markdown.
    """
    escaped = html.escape(text)

    if search_query and search_query.strip():
        escaped = search_and_highlight(escaped, search_query)

    # Bold: **text** → <strong>text</strong> (non-greedy, paired markers).
    escaped = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', escaped)
    # Italic: _text_ → <em>text</em>. Paired underscores only, no newline
    # crossing. Replaces the previous global `_`→`<em>` substitution which
    # broke any text containing an odd number of underscores (e.g. filenames).
    escaped = re.sub(
        r'_([^_\n]+?)_',
        r'<em style="color: #9ca3af;">\1</em>',
        escaped,
    )
    # Timestamps: [HH:MM:SS] → muted monospace badge. Visual weight is
    # deliberately light so the eye lands on speaker labels and prose
    # first; the timestamp is for navigation, not reading.
    escaped = _TIMESTAMP_MARKER_PATTERN.sub(
        r'<span style="color: #9ca3af; font-family: monospace; font-size: 0.85em;">[\1]</span>',
        escaped,
    )
    return escaped


def search_and_highlight(text: str, query: str) -> str:
    """
    Highlight search matches in the text using HTML mark tags.

    Args:
        text: The transcript text.
        query: The search query (case-insensitive).

    Returns:
        Text with matching words wrapped in <mark> tags.
    """
    if not query or not query.strip():
        return text

    # Escape special regex characters in query
    escaped_query = re.escape(query.strip())

    # Create pattern for case-insensitive word matching
    pattern = re.compile(f'({escaped_query})', re.IGNORECASE)

    # Replace matches with highlighted version
    return pattern.sub(r'<mark style="background: #fef08a; padding: 0 2px;">\1</mark>', text)


def get_reading_stats(text: str) -> dict:
    """
    Calculate reading statistics for the transcript.

    Args:
        text: The transcript text.

    Returns:
        Dictionary with word_count, char_count, reading_time_minutes.
    """
    # Remove markdown formatting and timestamp markers for an accurate count.
    # Timestamps are inserted by the timestamped-transcript path; without
    # this strip they'd inflate the word count by one per paragraph and
    # the character count by ten per paragraph.
    plain_text = _TIMESTAMP_MARKER_PATTERN.sub('', text)
    plain_text = re.sub(r'\*\*[^*]+\*\*', '', plain_text)  # Remove bold
    plain_text = re.sub(r'_[^_]+_', '', plain_text)  # Remove italic
    plain_text = plain_text.replace("**", "").replace("_", "")

    words = plain_text.split()
    word_count = len(words)
    char_count = len(plain_text)

    # Average reading speed: 200 words per minute
    reading_time_minutes = word_count / 200

    return {
        "word_count": word_count,
        "char_count": char_count,
        "reading_time_minutes": reading_time_minutes,
    }