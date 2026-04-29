"""Export transcribed text to DOCX and PDF formats."""

import io
import datetime
import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple, List

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from fpdf import FPDF


# Regex to match filler words marked with underscores like _um_ or _uh_.
# Speaker labels are parsed by ``_parse_speaker_block`` using a broader
# pattern that handles renamed speakers too — there is no separate
# Speaker-only regex here on purpose.
_FILLER_PATTERN = re.compile(r"_([^_]+)_")


# Module-level font cache: font_path for the Unicode font
_cached_font_path: Optional[str] = None


def _find_unicode_font() -> Optional[str]:
    """
    Find a Unicode-capable font for PDF generation.

    Prioritizes fonts with good Unicode support (French œ, German ü, etc.)

    Returns:
        Full path to font file if found, None otherwise.
    """
    # Font candidates in priority order (best Unicode support first)
    font_candidates = []

    if sys.platform == "darwin":  # macOS
        font_candidates = [
            # Arial Unicode has excellent Unicode support
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
            # Standard Arial in Supplemental folder
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            # DejaVu if installed via Homebrew
            "/opt/homebrew/share/fonts/dejavu/DejaVuSans.ttf",
            "/usr/local/share/fonts/dejavu/DejaVuSans.ttf",
        ]
    elif sys.platform == "linux":
        font_candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        ]
    elif sys.platform == "win32":
        font_candidates = [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\ArialUni.ttf",
            r"C:\Windows\Fonts\calibri.ttf",
        ]

    # Check each candidate
    for font_path in font_candidates:
        if os.path.isfile(font_path):
            return font_path

    return None


def _create_pdf_with_unicode_font() -> Tuple[FPDF, str]:
    """
    Create an FPDF instance with a Unicode font registered.

    Returns:
        Tuple of (FPDF instance, font family name to use)
    """
    global _cached_font_path

    pdf = FPDF()

    # Try to find and register a Unicode font
    if _cached_font_path is None:
        _cached_font_path = _find_unicode_font()

    if _cached_font_path and os.path.isfile(_cached_font_path):
        try:
            pdf.add_font("UniFont", "", _cached_font_path, uni=True)
            return pdf, "UniFont"
        except Exception:
            pass

    # Fallback to Helvetica (limited charset)
    return pdf, "Helvetica"


def _parse_speaker_block(block: str) -> Tuple[Optional[str], str]:
    """
    Parse a text block to extract speaker label and content.

    Args:
        block: A text block that may start with **Speaker X:**

    Returns:
        Tuple of (speaker_label, content) where speaker_label is None if no speaker found.
        Speaker label is returned without the ** markers.
    """
    lines = block.strip().split("\n", 1)
    if not lines:
        return None, ""

    first_line = lines[0].strip()

    # Check if first line is a speaker label like **Speaker 0:** or **CustomName:**
    match = re.match(r"^\*\*(.+?):\*\*$", first_line)
    if match:
        speaker_label = f"{match.group(1)}:"
        content = lines[1].strip() if len(lines) > 1 else ""
        return speaker_label, content

    # No speaker label - return entire block as content
    return None, block.strip()


def _parse_text_segments(text: str) -> List[Tuple[str, bool]]:
    """
    Parse text into segments, identifying filler words marked with underscores.

    Args:
        text: Text that may contain _filler_ markers.

    Returns:
        List of tuples (text, is_filler) where is_filler indicates if the
        segment is a filler word that should be styled differently.
    """
    segments = []
    last_end = 0

    for match in _FILLER_PATTERN.finditer(text):
        # Add text before the filler
        if match.start() > last_end:
            segments.append((text[last_end:match.start()], False))
        # Add the filler word (without underscores)
        segments.append((match.group(1), True))
        last_end = match.end()

    # Add remaining text after last filler
    if last_end < len(text):
        segments.append((text[last_end:], False))

    return segments if segments else [(text, False)]


def _add_formatted_text_to_paragraph(para, text: str) -> None:
    """
    Add text to a DOCX paragraph with filler words styled as italic gray.

    Args:
        para: A python-docx paragraph object.
        text: Text that may contain _filler_ markers.
    """
    segments = _parse_text_segments(text)

    for segment_text, is_filler in segments:
        run = para.add_run(segment_text)
        if is_filler:
            run.italic = True
            run.font.color.rgb = RGBColor(0x9c, 0xa3, 0xaf)  # Gray color for fillers


def _strip_filler_markers(text: str) -> str:
    """
    Remove filler word underscore markers, keeping the words.

    Args:
        text: Text that may contain _filler_ markers.

    Returns:
        Text with markers removed (e.g., "_um_" -> "um").
    """
    return _FILLER_PATTERN.sub(r"\1", text)


def export_docx(text: str, title: str = "Transcription") -> bytes:
    """
    Export text to a DOCX document with formatted speaker labels.

    Speaker labels (e.g., **Speaker 0:**) are rendered bold on their own line,
    followed by the speaker's text.

    Args:
        text: The transcribed text content.
        title: Document title.

    Returns:
        DOCX file content as bytes.
    """
    doc = Document()

    # Set default font
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Calibri"
    font.size = Pt(11)
    font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    # Title
    heading = doc.add_heading(title, level=1)
    heading.alignment = WD_ALIGN_PARAGRAPH.LEFT

    # Metadata
    meta_para = doc.add_paragraph()
    meta_run = meta_para.add_run(
        f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    meta_run.font.size = Pt(9)
    meta_run.font.color.rgb = RGBColor(0x99, 0x99, 0x99)
    meta_run.font.italic = True

    # Separator
    doc.add_paragraph("─" * 60)

    # Content — split by double newlines into paragraphs
    paragraphs = text.split("\n\n")
    for para_text in paragraphs:
        para_text = para_text.strip()
        if not para_text:
            continue

        speaker_label, content = _parse_speaker_block(para_text)

        if speaker_label:
            # Add speaker label as bold paragraph
            speaker_para = doc.add_paragraph()
            speaker_run = speaker_para.add_run(speaker_label)
            speaker_run.bold = True
            speaker_run.font.size = Pt(11)
            speaker_run.font.color.rgb = RGBColor(0x1a, 0x56, 0xdb)  # Blue color for speaker
            speaker_para.paragraph_format.space_after = Pt(2)

            # Add content paragraph with filler word formatting
            if content:
                content_para = doc.add_paragraph()
                _add_formatted_text_to_paragraph(content_para, content)
                content_para.paragraph_format.space_after = Pt(12)
                content_para.paragraph_format.left_indent = Inches(0.25)
        else:
            # Regular paragraph with filler word formatting
            para = doc.add_paragraph()
            _add_formatted_text_to_paragraph(para, para_text)
            para.paragraph_format.space_after = Pt(8)

    # Save to bytes buffer
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


def export_pdf(text: str, title: str = "Transcription") -> bytes:
    """
    Export text to a PDF document with formatted speaker labels.

    Speaker labels (e.g., **Speaker 0:**) are rendered in bold blue on their own line,
    followed by the speaker's text with slight indentation.

    Args:
        text: The transcribed text content.
        title: Document title.

    Returns:
        PDF file content as bytes.

    Raises:
        ValueError: If text is empty.
    """
    # Handle empty text input
    if not text or not text.strip():
        raise ValueError("Cannot export empty text to PDF")

    # Create PDF with Unicode font registered on THIS instance
    pdf, body_font = _create_pdf_with_unicode_font()
    pdf.set_auto_page_break(auto=True, margin=25)
    pdf.add_page()

    # Title
    pdf.set_font(body_font, "", 18)
    pdf.set_text_color(33, 33, 33)
    pdf.cell(0, 15, title, new_x="LMARGIN", new_y="NEXT")

    # Metadata
    pdf.set_font(body_font, "", 9)
    pdf.set_text_color(153, 153, 153)
    pdf.cell(
        0, 8,
        f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        new_x="LMARGIN", new_y="NEXT",
    )

    # Separator line
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y() + 2, 200, pdf.get_y() + 2)
    pdf.ln(8)

    # Content - process paragraphs with speaker label formatting
    paragraphs = text.split("\n\n")
    for para_text in paragraphs:
        para_text = para_text.strip()
        if not para_text:
            continue

        speaker_label, content = _parse_speaker_block(para_text)

        if speaker_label:
            # Speaker label: blue, slightly larger
            pdf.set_font(body_font, "", 12)
            pdf.set_text_color(26, 86, 219)  # Blue color for speaker
            pdf.cell(0, 7, speaker_label, new_x="LMARGIN", new_y="NEXT")

            # Content: normal text, slightly indented (strip filler markers)
            if content:
                pdf.set_font(body_font, "", 11)
                pdf.set_text_color(51, 51, 51)
                pdf.set_x(pdf.l_margin + 5)  # Slight indent
                pdf.multi_cell(0, 6, _strip_filler_markers(content))
            pdf.ln(6)
        else:
            # Regular paragraph (strip filler markers)
            pdf.set_font(body_font, "", 11)
            pdf.set_text_color(51, 51, 51)
            pdf.multi_cell(0, 6, _strip_filler_markers(para_text))
            pdf.ln(4)

    # Output as bytes - handle bytes, bytearray, and string return types from fpdf2
    output = pdf.output()
    if isinstance(output, (bytes, bytearray)):
        return bytes(output)
    return output.encode("utf-8")