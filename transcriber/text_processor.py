"""Text processing utilities for improving transcript readability."""

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


def add_paragraph_breaks(text: str, max_sentences_per_para: int = 4) -> str:
    """
    Add paragraph breaks to improve readability of long text blocks.

    For speaker-labeled content, breaks are added within each speaker's
    text if it exceeds the sentence limit. For plain text, breaks are
    added throughout.

    Args:
        text: The transcript text.
        max_sentences_per_para: Maximum sentences before inserting a break.

    Returns:
        Text with additional paragraph breaks for readability.
    """
    # Split into paragraphs (speaker blocks or regular paragraphs)
    paragraphs = text.split("\n\n")
    result_paragraphs = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Check if this is a speaker block
        match = _SPEAKER_LABEL_PATTERN.match(para)
        if match:
            # This is a speaker block: **Speaker X:**\nContent
            lines = para.split("\n", 1)
            speaker_label = lines[0]
            content = lines[1] if len(lines) > 1 else ""

            # Break content into readable paragraphs
            broken_content = _break_into_paragraphs(content, max_sentences_per_para)
            result_paragraphs.append(f"{speaker_label}\n{broken_content}")
        else:
            # Regular text block
            broken = _break_into_paragraphs(para, max_sentences_per_para)
            result_paragraphs.append(broken)

    return "\n\n".join(result_paragraphs)


def _break_into_paragraphs(text: str, max_sentences: int) -> str:
    """
    Break text into paragraphs based on sentence count.

    Uses sentence-ending punctuation to count sentences and inserts
    paragraph breaks when the limit is reached.
    """
    if not text.strip():
        return text

    # Split by sentence-ending punctuation while keeping the punctuation
    # This pattern splits on . ! ? followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    if len(sentences) <= max_sentences:
        return text

    # Group sentences into paragraphs
    paragraphs = []
    current_para = []

    for sentence in sentences:
        current_para.append(sentence)
        if len(current_para) >= max_sentences:
            paragraphs.append(" ".join(current_para))
            current_para = []

    # Add remaining sentences
    if current_para:
        paragraphs.append(" ".join(current_para))

    return "\n\n".join(paragraphs)


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
    # Remove markdown formatting for accurate count
    plain_text = re.sub(r'\*\*[^*]+\*\*', '', text)  # Remove bold
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