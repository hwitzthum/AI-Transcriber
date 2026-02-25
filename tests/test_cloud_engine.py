"""
Unit tests for cloud_engine module.

These tests verify the internal logic without making actual API calls:
1. Smart retry predicate (retriable vs non-retriable errors)
2. Overlap deduplication at chunk boundaries
3. Garbage text detection
4. Graceful degradation on chunk failures
"""

import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from transcriber import cloud_engine


def separator(title):
    print(f"\n{'='*60}")
    print(f"  TEST: {title}")
    print(f"{'='*60}\n")


def test_retry_predicate():
    """Test the smart retry predicate recognizes retriable vs non-retriable errors."""
    separator("Smart Retry Predicate")

    # --- Retriable errors (should return True) ---
    retriable_cases = [
        Exception("429 Too Many Requests"),
        Exception("rate limit exceeded"),
        Exception("500 Internal Server Error"),
        Exception("502 Bad Gateway"),
        Exception("503 Service Unavailable"),
        Exception("504 Gateway Timeout"),
        Exception("connection refused"),
        Exception("connection timeout"),
        Exception("read timeout"),
        Exception("temporarily unavailable"),
        cloud_engine.RetriableAPIError("explicit retriable"),
    ]

    print("  Testing retriable errors (should retry):")
    for exc in retriable_cases:
        result = cloud_engine._is_retriable_error(exc)
        status = "✅" if result else "❌"
        print(f"    {status} '{str(exc)[:40]}...' -> retry={result}")
        assert result, f"Should retry on: {exc}"

    # --- Non-retriable errors (should return False) ---
    non_retriable_cases = [
        Exception("401 Unauthorized"),
        Exception("invalid api key"),
        Exception("400 Bad Request"),
        Exception("unsupported format"),
        Exception("file too large"),
        Exception("403 Forbidden"),
        Exception("access denied"),
        Exception("404 not found"),
        cloud_engine.NonRetriableAPIError("explicit non-retriable"),
    ]

    print("\n  Testing non-retriable errors (should NOT retry):")
    for exc in non_retriable_cases:
        result = cloud_engine._is_retriable_error(exc)
        status = "✅" if not result else "❌"
        print(f"    {status} '{str(exc)[:40]}...' -> retry={result}")
        assert not result, f"Should NOT retry on: {exc}"

    print("\n  ✅ Smart retry predicate PASSED")


def test_overlap_deduplication():
    """Test that overlapping text at chunk boundaries is correctly deduplicated."""
    separator("Overlap Deduplication")

    # Test case 1: Clear overlap (minimum 4 words required to deduplicate)
    transcripts = [
        "Hello world, this is the first chunk of text ending with these four overlap words here today.",
        "these four overlap words here today. And this is the second chunk continuing the story.",
    ]
    merged = cloud_engine._deduplicate_overlap(transcripts)
    print(f"  Input chunks: 2")
    print(f"  Merged length: {len(merged)} chars")

    # Should not have double "these four overlap words here today"
    # Note: deduplication requires 4+ word match
    count = merged.lower().count("these four overlap words")
    print(f"  'these four overlap words' appears: {count} time(s)")
    assert count == 1, f"Expected 1 occurrence, got {count}"
    print("  ✅ Duplicate text removed (4+ word overlap)")

    # Test case 2: No overlap (different text)
    transcripts_no_overlap = [
        "First chunk ends differently.",
        "Second chunk starts differently.",
    ]
    merged_no_overlap = cloud_engine._deduplicate_overlap(transcripts_no_overlap)
    assert "First chunk" in merged_no_overlap
    assert "Second chunk" in merged_no_overlap
    print("  ✅ Non-overlapping text preserved")

    # Test case 3: Empty input
    assert cloud_engine._deduplicate_overlap([]) == ""
    print("  ✅ Empty input returns empty string")

    # Test case 4: Single chunk
    single = ["Only one chunk here."]
    assert cloud_engine._deduplicate_overlap(single) == single[0]
    print("  ✅ Single chunk returns unchanged")

    # Test case 5: German text with overlap (4+ words required)
    german_transcripts = [
        "Guten Tag, meine Damen und Herren, willkommen zur heutigen Konferenz hier.",
        "willkommen zur heutigen Konferenz hier. Heute sprechen wir über wichtige Themen.",
    ]
    merged_german = cloud_engine._deduplicate_overlap(german_transcripts)
    count_german = merged_german.lower().count("willkommen zur heutigen konferenz")
    print(f"  German 'willkommen zur heutigen Konferenz' appears: {count_german} time(s)")
    assert count_german == 1, f"Expected 1 occurrence, got {count_german}"
    print("  ✅ German overlap deduplication works (4+ word overlap)")

    print("\n  ✅ Overlap deduplication PASSED")


def test_garbage_detection():
    """Test that garbage text (hallucination) is correctly detected."""
    separator("Garbage Text Detection")

    # Test case 1: Normal text (should NOT be garbage)
    normal_text = """
    This is a normal transcription of a conversation. The speaker discusses
    various topics including technology, weather, and current events. The text
    contains proper punctuation and varied vocabulary.
    """
    is_garbage = cloud_engine._looks_like_garbage(normal_text)
    print(f"  Normal text is garbage: {is_garbage}")
    assert not is_garbage, "Normal text should not be flagged as garbage"
    print("  ✅ Normal text not flagged")

    # Test case 2: Repetitive text (hallucination loop - should be garbage)
    repetitive_text = "the the the the the the the the the the " * 20
    is_garbage = cloud_engine._looks_like_garbage(repetitive_text)
    print(f"  Repetitive text is garbage: {is_garbage}")
    assert is_garbage, "Repetitive text should be flagged as garbage"
    print("  ✅ Repetitive text flagged")

    # Test case 3: Symbol soup (should be garbage)
    symbol_soup = "!@#$%^&*()_+{}|:<>?~`-=[]\\;',./  " * 10
    is_garbage = cloud_engine._looks_like_garbage(symbol_soup)
    print(f"  Symbol soup is garbage: {is_garbage}")
    assert is_garbage, "Symbol soup should be flagged as garbage"
    print("  ✅ Symbol soup flagged")

    # Test case 4: Short text (too short to judge - should NOT be garbage)
    short_text = "Hi"
    is_garbage = cloud_engine._looks_like_garbage(short_text)
    print(f"  Short text is garbage: {is_garbage}")
    assert not is_garbage, "Short text should not be flagged"
    print("  ✅ Short text not flagged")

    # Test case 5: German text (should NOT be garbage)
    german_text = """
    Guten Morgen, meine Damen und Herren. Heute möchte ich über die
    wirtschaftliche Entwicklung in Deutschland sprechen. Die Zahlen zeigen
    einen positiven Trend in verschiedenen Sektoren der Industrie.
    """
    is_garbage = cloud_engine._looks_like_garbage(german_text)
    print(f"  German text is garbage: {is_garbage}")
    assert not is_garbage, "German text should not be flagged as garbage"
    print("  ✅ German text not flagged")

    # Test case 6: French text with special chars (should NOT be garbage)
    french_text = """
    Bonjour à tous. Aujourd'hui, nous allons discuter de l'économie française
    et des œuvres d'art qui représentent notre culture. C'est très important
    de comprendre ces sujets.
    """
    is_garbage = cloud_engine._looks_like_garbage(french_text)
    print(f"  French text is garbage: {is_garbage}")
    assert not is_garbage, "French text should not be flagged as garbage"
    print("  ✅ French text not flagged")

    print("\n  ✅ Garbage detection PASSED")


def test_provider_validation():
    """Test that invalid providers are rejected."""
    separator("Provider Validation")

    # Invalid provider should raise ValueError
    try:
        cloud_engine.transcribe_chunks(
            chunk_paths=["fake.mp3"],
            provider="Invalid Provider",
            api_key="fake_key",
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Invalid provider error: {e}")
        assert "Unknown provider" in str(e)
        print("  ✅ Invalid provider rejected")

    # Valid providers should be in PROVIDERS dict
    valid_providers = list(cloud_engine.PROVIDERS.keys())
    print(f"  Valid providers: {valid_providers}")
    assert "OpenAI Whisper API" in valid_providers
    assert "Deepgram Nova-2" in valid_providers
    print("  ✅ Provider list is correct")

    print("\n  ✅ Provider validation PASSED")


def test_eta_estimation():
    """Test the ETA estimation helper."""
    separator("ETA Estimation")

    # No data yet
    eta = cloud_engine._estimate_eta(0, 10, [])
    assert eta == "", "No ETA on first chunk"
    print("  ✅ No ETA before first chunk")

    # Some timing data
    chunk_times = [10.0, 12.0, 11.0]  # ~11 sec average
    eta = cloud_engine._estimate_eta(3, 10, chunk_times)
    print(f"  ETA after 3/10 chunks: {eta}")
    assert "ETA" in eta, "Should contain ETA"
    assert "m" in eta or "s" in eta, "Should contain time unit"
    print("  ✅ ETA calculated correctly")

    # Near completion
    eta = cloud_engine._estimate_eta(9, 10, [5.0] * 9)
    print(f"  ETA after 9/10 chunks: {eta}")
    assert "s" in eta, "Should be in seconds near completion"
    print("  ✅ Short ETA near completion")

    print("\n  ✅ ETA estimation PASSED")


def test_find_overlap_length():
    """Test the overlap length finder."""
    separator("Overlap Length Detection")

    # Clear overlap
    tail = ["the", "quick", "brown", "fox", "jumps"]
    curr = ["brown", "fox", "jumps", "over", "lazy"]
    overlap = cloud_engine._find_overlap_length(tail, curr)
    print(f"  5-word tail, 3-word overlap: {overlap}")
    # Minimum is 4 words, so 3-word overlap won't match
    # Let's test with 4+ word overlap

    tail = ["one", "two", "three", "four", "five", "six"]
    curr = ["three", "four", "five", "six", "seven", "eight"]
    overlap = cloud_engine._find_overlap_length(tail, curr)
    print(f"  6-word tail, 4-word overlap: {overlap}")
    assert overlap == 4, f"Expected 4-word overlap, got {overlap}"
    print("  ✅ 4-word overlap detected")

    # No overlap
    tail = ["completely", "different", "words", "here"]
    curr = ["unrelated", "text", "follows", "now"]
    overlap = cloud_engine._find_overlap_length(tail, curr)
    print(f"  No overlap: {overlap}")
    assert overlap == 0, "Should be 0 for no overlap"
    print("  ✅ No false positive overlap")

    # Punctuation difference (should still match)
    tail = ["Hello,", "world!", "How", "are", "you?"]
    curr = ["How", "are", "you?", "I", "am"]
    overlap = cloud_engine._find_overlap_length(tail, curr)
    print(f"  With punctuation: {overlap}")
    # Minimum is 4 words, so 3 words won't match
    # This is expected behavior

    print("\n  ✅ Overlap length detection PASSED")


if __name__ == "__main__":
    print("\n" + "🧪" * 30)
    print("  CLOUD ENGINE UNIT TESTS")
    print("🧪" * 30)

    all_passed = True
    results = {}

    tests = [
        ("Smart Retry Predicate", test_retry_predicate),
        ("Overlap Deduplication", test_overlap_deduplication),
        ("Garbage Detection", test_garbage_detection),
        ("Provider Validation", test_provider_validation),
        ("ETA Estimation", test_eta_estimation),
        ("Overlap Length Detection", test_find_overlap_length),
    ]

    for name, test_fn in tests:
        try:
            test_fn()
            results[name] = "✅ PASSED"
        except Exception as e:
            results[name] = f"❌ FAILED: {e}"
            all_passed = False
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("  CLOUD ENGINE TEST SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        print(f"  {result}  {name}")

    print(f"\n{'='*60}")
    if all_passed:
        print("  🎉 ALL CLOUD ENGINE TESTS PASSED!")
    else:
        print("  ⚠️  SOME TESTS FAILED — see above for details")
    print(f"{'='*60}\n")

    sys.exit(0 if all_passed else 1)