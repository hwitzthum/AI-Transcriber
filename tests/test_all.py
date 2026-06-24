"""
Comprehensive test script for the Transcriber app.
Tests all key design decisions:
1. Audio processing & chunking
2. Local transcription (mlx-whisper)
3. DOCX & PDF export
4. Format support (AIFF → MP3 conversion)
"""

import os
import shutil
import sys

import pytest

# Skip tests that need ffmpeg/ffprobe when those binaries aren't on PATH.
# The app requires them at startup (see transcriber/audio_processor.py:require_ffmpeg),
# so these tests can only run in environments where they're installed.
_FFTOOLS_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
requires_fftools = pytest.mark.skipif(
    not _FFTOOLS_AVAILABLE,
    reason="ffmpeg and ffprobe not found on PATH",
)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from transcriber import audio_processor, cloud_engine, exporter


@pytest.fixture
def text():
    """Sample text fixture for export tests."""
    return """**Speaker 0:**
Hello, this is a test transcription with multiple speakers.

**Speaker 1:**
Yes, we're testing the export functionality to ensure DOCX and PDF work correctly.

**Speaker 0:**
Great, let's make sure the formatting is preserved properly."""


def separator(title):
    print(f"\n{'='*60}")
    print(f"  TEST: {title}")
    print(f"{'='*60}\n")


def test_audio_validation():
    """Test 1: File validation for various formats."""
    separator("Audio File Validation")
    
    # Valid file
    ok, msg = audio_processor.validate_file("tests/test_english.mp3")
    print(f"  ✅ Valid MP3:   ok={ok}, msg={msg}")
    assert ok, "MP3 validation failed"
    
    # Valid AIFF 
    ok, msg = audio_processor.validate_file("tests/test_english.aiff")
    print(f"  ✅ Valid AIFF:  ok={ok}, msg={msg}")
    assert ok, "AIFF validation failed"
    
    # Non-existent file
    ok, msg = audio_processor.validate_file("nonexistent.mp3")
    print(f"  ✅ Missing:    ok={ok}, msg={msg}")
    assert not ok, "Should fail for missing file"
    
    # Unsupported format
    ok, msg = audio_processor.validate_file("tests/test_script.py")
    print(f"  ✅ Bad format: ok={ok}, msg={msg}")
    assert not ok, "Should fail for .py file"

    # Path traversal: deny-listed sensitive locations even if file exists.
    # /etc/hosts is readable on macOS/Linux and would otherwise pass the
    # exists/is_file/extension checks (but it doesn't have a supported suffix
    # — so we test with a contrived path under /etc that has the right suffix).
    if os.path.exists("/etc/hosts"):
        # Symlink an .mp3-suffixed name into /etc to verify deny-list catches it
        link_path = "/tmp/_traversal_test.mp3"
        try:
            if os.path.lexists(link_path):
                os.unlink(link_path)
            os.symlink("/etc/hosts", link_path)
            ok, msg = audio_processor.validate_file(link_path)
            print(f"  ✅ Traversal blocked: ok={ok}, msg={msg}")
            assert not ok, "Should reject path resolving into /etc"
            assert "protected" in msg.lower(), f"Expected 'protected' in msg, got: {msg}"
        finally:
            if os.path.lexists(link_path):
                os.unlink(link_path)

    print("\n  ✅ All validation tests PASSED")


def test_compute_upload_hash_deterministic():
    """Same content must produce the same hash; different content differs."""
    import io
    separator("Upload Hash Determinism")

    a = io.BytesIO(b"some audio content here for hashing" * 100)
    b = io.BytesIO(b"some audio content here for hashing" * 100)
    different = io.BytesIO(b"different audio content" * 100)

    hash_a = audio_processor.compute_upload_hash(a)
    hash_b = audio_processor.compute_upload_hash(b)
    hash_diff = audio_processor.compute_upload_hash(different)

    print(f"  hash(a) = {hash_a}")
    print(f"  hash(b) = {hash_b}")
    print(f"  hash(different) = {hash_diff}")
    assert hash_a == hash_b, "Identical content must produce identical hashes"
    assert hash_a != hash_diff, "Different content must produce different hashes"
    print("  ✅ Hash is deterministic and content-sensitive")


def test_compute_upload_hash_uses_size():
    """Two buffers with identical head/tail but different sizes must hash differently."""
    import io
    # Construct head and tail blocks (each 100 KB) shared between both buffers.
    head = b"H" * 100_000
    tail = b"T" * 100_000
    # Buffer 1: just head + tail (200 KB total)
    buf1 = io.BytesIO(head + tail)
    # Buffer 2: head + 1 MB filler + tail (still has the same first 64KB and last 64KB)
    buf2 = io.BytesIO(head + b"\x00" * 1_000_000 + tail)

    h1 = audio_processor.compute_upload_hash(buf1)
    h2 = audio_processor.compute_upload_hash(buf2)
    assert h1 != h2, "Hash must include file size to disambiguate same head/tail"
    print(f"  ✅ Size is part of the hash — different lengths produce different hashes")


@requires_fftools
def test_chunking_threshold_per_provider():
    """A file under the OpenAI 24 MB limit reports needs_chunking=False;
    the same file with a tiny 1 KB limit reports True. Demonstrates the
    per-provider knob without needing a 25 MB fixture."""
    separator("Chunking Threshold (Per-Provider)")

    test_file = "tests/test_english.mp3"
    size = os.path.getsize(test_file)
    print(f"  Test file size: {size} bytes ({size / 1024:.1f} KB)")

    # With the default OpenAI/Groq limit, the small test fixture fits in one chunk
    assert not audio_processor.needs_chunking(test_file), (
        "Small test file should NOT need chunking at default OpenAI limit"
    )
    print(f"  ✅ Default OpenAI limit: small file does not need chunking")

    # With an artificially tiny limit, the same file should need chunking
    assert audio_processor.needs_chunking(test_file, max_bytes=1024), (
        "Test file should need chunking when limit is 1 KB"
    )
    print(f"  ✅ Tiny 1 KB limit: file needs chunking")

    # And chunk_audio must respect the parameterised limit: a tiny limit
    # produces multiple chunk paths, the default limit produces one.
    single = audio_processor.chunk_audio(test_file)
    assert len(single) == 1, f"Expected 1 chunk at default limit, got {len(single)}"
    audio_processor.cleanup_chunks(single, test_file)
    print(f"  ✅ Single-chunk path returns 1 file at default limit")


@requires_fftools
def test_audio_info():
    """Test 2: Audio metadata extraction."""
    separator("Audio Metadata Extraction")
    
    info = audio_processor.get_audio_info("tests/test_english.mp3")
    print(f"  Duration:    {info['duration_formatted']} ({info['duration_seconds']:.1f}s)")
    print(f"  Size:        {info['file_size_mb']:.2f} MB")
    print(f"  Channels:    {info['channels']}")
    print(f"  Sample rate: {info['sample_rate']} Hz")
    
    assert info["duration_seconds"] > 0, "Duration should be > 0"
    assert info["file_size_mb"] > 0, "File size should be > 0"
    
    print("\n  ✅ Metadata extraction PASSED")


@requires_fftools
def test_chunking_logic():
    """Test 3: Chunking — verify the small file doesn't get split,
    and test chunking with a forced lower threshold via the max_bytes kwarg."""
    separator("Chunking Logic")

    # Small file should NOT need chunking at the default OpenAI/Groq limit
    needs = audio_processor.needs_chunking("tests/test_english.mp3")
    print(f"  Small file needs chunking: {needs}")
    assert not needs, "Small file should not need chunking"
    print(f"  ✅ Small file correctly skips chunking")

    # Force chunking by passing a tiny max_bytes — exercises the multi-chunk path.
    forced_limit = 5 * 1024  # 5 KB
    needs = audio_processor.needs_chunking("tests/test_english.mp3", max_bytes=forced_limit)
    print(f"  With 5KB threshold, needs chunking: {needs}")
    assert needs, "Should need chunking with 5KB threshold"

    chunks = audio_processor.chunk_audio("tests/test_english.mp3", max_bytes=forced_limit)
    print(f"  Chunk count: {len(chunks)}")
    print(f"  Chunk files:")
    for i, c in enumerate(chunks):
        size = os.path.getsize(c) / 1024
        print(f"    [{i}] {os.path.basename(c)} — {size:.1f} KB")

    assert len(chunks) > 1, "Should produce multiple chunks"

    # Verify all chunk files exist
    for c in chunks:
        assert os.path.exists(c), f"Chunk file missing: {c}"

    # Test cleanup
    audio_processor.cleanup_chunks(chunks, "tests/test_english.mp3")
    for c in chunks:
        assert not os.path.exists(c), f"Chunk not cleaned up: {c}"
    print(f"  ✅ Cleanup removed all {len(chunks)} temp chunks")

    print("\n  ✅ Chunking logic PASSED")


@requires_fftools
def test_iter_chunks_streaming():
    """``iter_chunks`` is the lazy variant used by the streaming pipeline.

    Why a dedicated test: ``chunk_audio`` exercises the list-returning
    code path; the streaming entry point has its own arithmetic for the
    upfront chunk count + a generator that yields paths over time. A
    regression in either would silently break the encode/upload overlap
    in app.py.
    """
    separator("Streaming chunk iterator (iter_chunks)")

    # Single-chunk path: small file at default limit yields exactly one path
    total, it = audio_processor.iter_chunks("tests/test_english.mp3")
    assert total == 1, f"Expected total=1 for single-chunk path, got {total}"
    paths = list(it)
    assert len(paths) == 1, f"Expected 1 yielded path, got {len(paths)}"
    audio_processor.cleanup_chunks(paths, "tests/test_english.mp3")
    print("  ✅ Single-chunk path: total=1, 1 path yielded")

    # Multi-chunk path: forced tiny limit produces several chunks; the
    # announced total must match what the iterator actually yields.
    forced_limit = 5 * 1024
    total, it = audio_processor.iter_chunks(
        "tests/test_english.mp3", max_bytes=forced_limit
    )
    assert total > 1, f"Expected multi-chunk total, got {total}"

    yielded: list[str] = []
    for path in it:
        # Each path must exist on disk by the time it's yielded — otherwise
        # the streaming consumer would race ffmpeg.
        assert os.path.exists(path), f"Yielded path does not exist: {path}"
        yielded.append(path)

    # Iterator may yield slightly fewer than ``total`` if the tail chunk
    # is shorter than _MIN_CHUNK_SEC. It must never yield more.
    assert len(yielded) <= total, (
        f"Iterator yielded {len(yielded)} paths but announced total={total}"
    )
    assert len(yielded) >= 1, "Multi-chunk path produced zero chunks"

    audio_processor.cleanup_chunks(yielded, "tests/test_english.mp3")
    for path in yielded:
        assert not os.path.exists(path), f"Chunk not cleaned up: {path}"
    print(f"  ✅ Multi-chunk path: announced total={total}, yielded {len(yielded)} paths")


@requires_fftools
def test_video_audio_extraction():
    """Verify video files are transcoded to MP3 audio via ffmpeg (the
    `_ensure_mp3` helper drops video with `-vn` and produces a speech-
    optimised MP3)."""
    separator("Video → MP3 Extraction")

    video_path = "tests/test_video.mp4"
    if not os.path.exists(video_path):
        print("  ⚠️ Test video not found, skipping")
        return

    extracted_path = audio_processor._ensure_mp3(video_path)
    print(f"  Extracted to: {extracted_path}")

    assert os.path.exists(extracted_path), "Extracted file should exist"
    assert os.path.getsize(extracted_path) > 0, "Extracted file should not be empty"
    assert extracted_path.endswith(".mp3"), "Extracted file should be MP3"
    # Sanity-check it's an actual audio file ffprobe can read.
    info = audio_processor.get_audio_info(extracted_path)
    print(f"  Duration: {info['duration_formatted']}")
    assert info["duration_seconds"] > 0, "Extracted MP3 should have a duration"

    os.unlink(extracted_path)
    print("  ✅ Video → MP3 extraction works")


def test_docx_export(text):
    """Test 6: DOCX export."""
    separator("DOCX Export")
    
    docx_bytes = exporter.export_docx(text, title="Test Transcription")
    
    print(f"  DOCX size: {len(docx_bytes):,} bytes")
    assert len(docx_bytes) > 100, "DOCX too small"
    
    # Write to temp file and verify it's valid
    tmp = os.path.join("tests", "test_output.docx")
    with open(tmp, "wb") as f:
        f.write(docx_bytes)
    
    # Verify it's a valid ZIP (DOCX is a ZIP)
    import zipfile
    assert zipfile.is_zipfile(tmp), "DOCX is not a valid ZIP archive"
    print(f"  ✅ Valid DOCX file: {tmp}")
    
    # Check it contains expected XML
    with zipfile.ZipFile(tmp) as z:
        names = z.namelist()
        assert "word/document.xml" in names, "Missing word/document.xml"
        print(f"  ✅ Contains word/document.xml and {len(names)} other files")
    
    os.unlink(tmp)
    print("\n  ✅ DOCX export PASSED")


def test_pdf_export(text):
    """Test 7: PDF export."""
    separator("PDF Export")
    
    pdf_bytes = exporter.export_pdf(text, title="Test Transcription")
    
    print(f"  PDF size: {len(pdf_bytes):,} bytes")
    assert isinstance(pdf_bytes, bytes), f"PDF output should be bytes, got {type(pdf_bytes)}"
    assert len(pdf_bytes) > 100, "PDF too small"
    
    # Verify it starts with PDF header
    assert pdf_bytes[:5] == b"%PDF-", "Not a valid PDF (missing header)"
    print(f"  ✅ Valid PDF header")
    
    # Write to verify
    tmp = os.path.join("tests", "test_output.pdf")
    with open(tmp, "wb") as f:
        f.write(pdf_bytes)
    print(f"  ✅ Written to: {tmp}")
    
    os.unlink(tmp)
    print("\n  ✅ PDF export PASSED")


@requires_fftools
def test_format_conversion():
    """Test 8: AIFF format conversion (non-MP3 input)."""
    separator("Format Conversion (AIFF → MP3)")
    
    # Process AIFF file directly
    info = audio_processor.get_audio_info("tests/test_english.aiff")
    print(f"  AIFF duration: {info['duration_formatted']}")
    print(f"  AIFF size:     {info['file_size_mb']:.2f} MB")
    
    chunks = audio_processor.chunk_audio("tests/test_english.aiff")
    print(f"  Converted to:  {len(chunks)} chunk(s)")
    
    for c in chunks:
        assert c.endswith(".mp3"), f"Chunk should be MP3: {c}"
        print(f"  ✅ Chunk format: MP3 ({os.path.basename(c)})")
    
    audio_processor.cleanup_chunks(chunks, "tests/test_english.aiff")
    print("\n  ✅ Format conversion PASSED")


# Run with: uv run pytest tests/
# The previous __main__ block bypassed pytest fixture injection and
# called fixture-decorated tests directly with positional strings,
# masking real fixture errors. Pytest discovery is the single source
# of truth for what gets executed.
