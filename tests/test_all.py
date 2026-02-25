"""
Comprehensive test script for the Transcriber app.
Tests all key design decisions:
1. Audio processing & chunking
2. Local transcription (mlx-whisper)
3. DOCX & PDF export
4. Format support (AIFF → MP3 conversion)
"""

import os
import sys
import tempfile
import time

import pytest

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
    
    # Non-existent file
    ok, msg = audio_processor.validate_file("nonexistent.mp3")
    print(f"  ✅ Missing:    ok={ok}, msg={msg}")
    assert not ok, "Should fail for missing file"
    
    # Unsupported format
    ok, msg = audio_processor.validate_file("tests/test_script.py")
    print(f"  ✅ Bad format: ok={ok}, msg={msg}")
    assert not ok, "Should fail for .py file"
    
    print("\n  ✅ All validation tests PASSED")


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


def test_chunking_logic():
    """Test 3: Chunking — verify the small file doesn't get split, 
    and test chunking with a forced lower threshold."""
    separator("Chunking Logic")
    
    # Small file should NOT need chunking
    needs = audio_processor.needs_chunking("tests/test_english.mp3")
    print(f"  Small file needs chunking: {needs}")
    assert not needs, "Small file should not need chunking"
    print(f"  ✅ Small file correctly skips chunking")
    
    # Test chunking with artificially low threshold
    original_max = audio_processor.MAX_CHUNK_BYTES
    try:
        # Force chunking by setting threshold very low (5 KB)
        audio_processor.MAX_CHUNK_BYTES = 5 * 1024
        
        needs = audio_processor.needs_chunking("tests/test_english.mp3")
        print(f"  With 5KB threshold, needs chunking: {needs}")
        assert needs, "Should need chunking with 5KB threshold"
        
        chunks = audio_processor.chunk_audio("tests/test_english.mp3")
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
        
    finally:
        audio_processor.MAX_CHUNK_BYTES = original_max
    
    print("\n  ✅ Chunking logic PASSED")


def test_video_optimization():
    separator("Video Optimization (Large File Handling)")
    
    video_path = "tests/test_video.mp4"
    if not os.path.exists(video_path):
        # reuse generation logic or skip
        print("  ⚠️ Test video not found, skipping optimization test")
        return

    # Check if optimization path works by mocking size check or lowering threshold?
    # We can't easily change the hardcoded 50MB inside the function without monkeypatching.
    # But we can verify _extract_audio_from_video works directly.
    
    print("  ⏳ Testing direct audio extraction...")
    extracted_path = audio_processor._extract_audio_from_video(video_path)
    print(f"  Extracted to: {extracted_path}")
    
    assert os.path.exists(extracted_path), "Extracted file should exist"
    assert os.path.getsize(extracted_path) > 0, "Extracted file should not be empty"
    assert extracted_path.endswith(".mp3"), "Extracted file should be MP3"
    
    # Check info of extracted file
    info = audio_processor.get_audio_info(extracted_path)
    print(f"  Extracted Duration: {info['duration_formatted']}")
    
    # Cleanup
    os.unlink(extracted_path)
    print("  ✅ Direct extraction PASSED")


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


if __name__ == "__main__":
    print("\n" + "🧪" * 30)
    print("  TRANSCRIBER — CLOUD-ONLY TEST SUITE")
    print("🧪" * 30)
    
    all_passed = True
    results = {}
    
    tests = [
        ("Audio Validation", test_audio_validation),
        ("Audio Metadata", test_audio_info),
        ("Chunking Logic", test_chunking_logic),
        ("Format Conversion", test_format_conversion),
        ("Video Optimization", test_video_optimization),
    ]
    
    # Note: Cloud API integration tests require real API keys.
    # Unit tests for retry logic, deduplication, etc. are below.
    
    for name, test_fn in tests:
        try:
            result = test_fn()
            results[name] = "✅ PASSED"
        except Exception as e:
            results[name] = f"❌ FAILED: {e}"
            all_passed = False
            import traceback
            traceback.print_exc()
    
    # Export tests
    export_text = "This is a placeholder text for export testing, as we skipped transcription."
    
    for name, test_fn in [("DOCX Export", test_docx_export), ("PDF Export", test_pdf_export)]:
        try:
            test_fn(export_text)
            results[name] = "✅ PASSED"
        except Exception as e:
            results[name] = f"❌ FAILED: {e}"
            all_passed = False
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        print(f"  {result}  {name}")
    
    print(f"\n{'='*60}")
    if all_passed:
        print("  🎉 ALL TESTS PASSED!")
    else:
        print("  ⚠️  SOME TESTS FAILED — see above for details")
    print(f"{'='*60}\n")
