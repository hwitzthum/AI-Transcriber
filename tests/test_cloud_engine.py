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

    # --- Substring false-positives (must NOT retry just because "400"/"429"
    # appears inside a larger token like "14002" or "stream id 4291"). ---
    embedded_substring_cases = [
        Exception("Server returned status 14002"),
        Exception("error in stream id 4291 but otherwise fine"),
    ]
    print("\n  Testing substring false-positives (should NOT retry):")
    for exc in embedded_substring_cases:
        result = cloud_engine._is_retriable_error(exc)
        status = "✅" if not result else "❌"
        print(f"    {status} '{str(exc)[:40]}...' -> retry={result}")
        assert not result, f"Should NOT retry on embedded-substring: {exc}"

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


class _Obj:
    """Lightweight stand-in for SDK response objects.

    Deepgram's response objects expose nested fields via attribute access,
    so a tiny namespace class is enough to exercise the formatters without
    importing the SDK or hitting the network.
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_format_diarized_from_paragraphs():
    """Paragraphs path: each paragraph becomes its own **Speaker N:** block,
    sentence text is joined with smart_format punctuation preserved."""
    separator("Deepgram Paragraphs Formatter")

    alt = _Obj(paragraphs=_Obj(paragraphs=[
        _Obj(speaker=0, sentences=[
            _Obj(text="Hello, how are you today?"),
            _Obj(text="I hope you are well."),
        ]),
        _Obj(speaker=1, sentences=[
            _Obj(text="I'm doing great, thanks!"),
        ]),
        _Obj(speaker=0, sentences=[
            _Obj(text="Glad to hear it."),
        ]),
    ]))
    result = cloud_engine._format_diarized_from_paragraphs(alt)
    print(f"  Output:\n{result}")
    assert "**Speaker 0:**" in result
    assert "**Speaker 1:**" in result
    # Smart-format punctuation must survive the join
    assert "Hello, how are you today?" in result
    assert "I'm doing great, thanks!" in result
    # Each paragraph should produce its own block — three blocks total
    assert result.count("**Speaker") == 3
    print("  ✅ Paragraphs formatter works")


def test_format_diarized_from_paragraphs_empty():
    """Missing or empty paragraphs returns None so the caller can fall back."""
    separator("Deepgram Paragraphs Empty / Missing")

    # No paragraphs attribute at all
    assert cloud_engine._format_diarized_from_paragraphs(_Obj()) is None
    # paragraphs object but empty list
    assert cloud_engine._format_diarized_from_paragraphs(
        _Obj(paragraphs=_Obj(paragraphs=[]))
    ) is None
    # paragraphs with no speaker
    assert cloud_engine._format_diarized_from_paragraphs(
        _Obj(paragraphs=_Obj(paragraphs=[_Obj(speaker=None, sentences=[])]))
    ) is None
    print("  ✅ Empty/missing paragraphs return None")


def test_format_diarized_from_words_fallback():
    """Word-loop fallback: speaker change starts a new block, long pauses also."""
    separator("Deepgram Words Fallback Formatter")

    alt = _Obj(words=[
        _Obj(speaker=0, start=0.0, end=0.5, punctuated_word="Hello,"),
        _Obj(speaker=0, start=0.6, end=1.0, punctuated_word="world."),
        # Speaker change → new block
        _Obj(speaker=1, start=1.5, end=2.0, punctuated_word="Hi"),
        _Obj(speaker=1, start=2.1, end=2.5, punctuated_word="there."),
        # 2-second pause → new block (pause > 1.5s threshold)
        _Obj(speaker=1, start=4.5, end=5.0, punctuated_word="Sorry,"),
        _Obj(speaker=1, start=5.1, end=5.5, punctuated_word="continuing."),
    ])
    result = cloud_engine._format_diarized_from_words(alt)
    print(f"  Output:\n{result}")
    assert result.count("**Speaker 0:**") == 1
    # Two blocks for speaker 1: one for "Hi there." and one after the pause
    assert result.count("**Speaker 1:**") == 2
    print("  ✅ Word-loop fallback works")


def test_format_diarized_from_words_empty():
    """Empty/missing words returns empty string."""
    assert cloud_engine._format_diarized_from_words(_Obj()) == ""
    assert cloud_engine._format_diarized_from_words(_Obj(words=[])) == ""
    print("  ✅ Empty words returns ''")


def test_provider_max_chunk_bytes():
    """Each provider exposes its own upload ceiling.

    OpenAI/Groq cap at ~25 MB; Deepgram allows up to 500 MB in one chunk
    so most real-world meetings can be uploaded without chunking at all.
    """
    separator("Per-Provider Max Chunk Bytes")

    openai_limit = cloud_engine.get_max_chunk_bytes("OpenAI Whisper API")
    groq_limit = cloud_engine.get_max_chunk_bytes("Groq (whisper-large-v3-turbo)")
    deepgram_limit = cloud_engine.get_max_chunk_bytes("Deepgram Nova-2")
    deepgram_multi = cloud_engine.get_max_chunk_bytes("Deepgram Nova-3 (Multilingual)")
    unknown_limit = cloud_engine.get_max_chunk_bytes("not-a-real-provider")

    print(f"  OpenAI:           {openai_limit / 1024 / 1024:.0f} MB")
    print(f"  Groq:             {groq_limit / 1024 / 1024:.0f} MB")
    print(f"  Deepgram Nova-2:  {deepgram_limit / 1024 / 1024:.0f} MB")
    print(f"  Deepgram Nova-3:  {deepgram_multi / 1024 / 1024:.0f} MB")
    print(f"  Unknown provider: {unknown_limit / 1024 / 1024:.0f} MB")

    assert openai_limit == 24 * 1024 * 1024, "OpenAI limit must be 24 MB"
    assert groq_limit == 24 * 1024 * 1024, "Groq limit must be 24 MB"
    assert deepgram_limit == 500 * 1024 * 1024, "Deepgram limit must be 500 MB"
    assert deepgram_multi == 500 * 1024 * 1024, "Deepgram multilingual must be 500 MB"
    # Unknown providers fall back to the conservative OpenAI/Groq ceiling
    # so we never accidentally upload an oversized chunk.
    assert unknown_limit == 24 * 1024 * 1024, "Unknown provider must default to safe limit"

    # Deepgram limit must be at least an order of magnitude larger than the
    # OpenAI/Groq limit — otherwise the per-provider strategy isn't earning
    # its complexity.
    assert deepgram_limit >= openai_limit * 10
    print("  ✅ Per-provider chunk limits configured correctly")


def test_parallel_transcribe_chunks_preserves_order(monkeypatch):
    """Parallel chunk transcription must reassemble results in chunk-index
    order regardless of which future completes first, and must surface
    detected_language from the lowest-index chunk that reports one.

    Why: with concurrency, the first future to *complete* is not the first
    chunk; if we naively appended results in completion order the merged
    transcript would be garbled and the detected-language picker would
    become non-deterministic across runs.
    """
    import time as _time

    separator("Parallel transcribe_chunks ordering")

    # Three chunks; sleep durations are deliberately reverse-ordered so
    # chunk index 2 finishes first if everything ran in parallel — the
    # exact thing we want to stress-test.
    fake_results = {
        "chunk_a.mp3": {"text": "alpha beta gamma delta epsilon", "detected_language": "en"},
        "chunk_b.mp3": {"text": "zeta eta theta iota kappa", "detected_language": "en"},
        "chunk_c.mp3": {"text": "lambda mu nu xi omicron", "detected_language": "en"},
    }
    sleeps = {"chunk_a.mp3": 0.10, "chunk_b.mp3": 0.05, "chunk_c.mp3": 0.01}

    def fake_transcribe_single(file_path, provider, config, client, language, diarize, low_confidence_threshold=None):
        _time.sleep(sleeps[file_path])
        return fake_results[file_path]

    monkeypatch.setattr(cloud_engine, "_transcribe_single", fake_transcribe_single)
    monkeypatch.setattr(cloud_engine, "_create_client", lambda provider, key: object())

    result = cloud_engine.transcribe_chunks(
        ["chunk_a.mp3", "chunk_b.mp3", "chunk_c.mp3"],
        provider="OpenAI Whisper API",
        api_key="fake",
        max_workers=3,
    )

    # The merged transcript must contain words in chunk-index order even
    # though chunk_c (index 2) completed first.
    text = result["text"]
    pos_a = text.find("alpha")
    pos_z = text.find("zeta")
    pos_l = text.find("lambda")
    print(f"  positions: alpha={pos_a}, zeta={pos_z}, lambda={pos_l}")
    assert pos_a >= 0 and pos_z > pos_a and pos_l > pos_z, (
        "Chunks must be assembled in index order, not completion order"
    )
    assert result["failed_chunks"] == []
    assert result["detected_language"] == "en"
    print("  ✅ Parallel results assembled in chunk-index order")


def test_streaming_overlaps_chunking_and_transcription(monkeypatch):
    """Streaming chunker + transcribe must overlap: while the producer is
    still yielding chunk N+1, chunk N must already be uploading.

    Why this matters: the whole point of transcribe_chunks_streaming is to
    cut end-to-end wall time on long files by running ffmpeg encode in
    parallel with API uploads. If the consumer naively materialised the
    iterator before submitting (or held a lock), this test's wall time
    would be (chunking_time + transcription_time) instead of
    ~max(chunking_time, transcription_time).
    """
    import time as _time

    separator("Streaming overlap of chunking + transcription")

    chunk_count = 4
    encode_secs = 0.20  # simulated ffmpeg encode per chunk
    upload_secs = 0.20  # simulated API call per chunk

    def fake_chunk_iter():
        for i in range(chunk_count):
            _time.sleep(encode_secs)
            yield f"chunk_{i}.mp3"

    def fake_transcribe_single(file_path, provider, config, client, language, diarize, low_confidence_threshold=None):
        _time.sleep(upload_secs)
        return {"text": f"text-from-{file_path}", "detected_language": "en"}

    monkeypatch.setattr(cloud_engine, "_transcribe_single", fake_transcribe_single)
    monkeypatch.setattr(cloud_engine, "_create_client", lambda provider, key: object())

    t0 = _time.monotonic()
    result = cloud_engine.transcribe_chunks_streaming(
        chunk_iter=fake_chunk_iter(),
        total=chunk_count,
        provider="OpenAI Whisper API",
        api_key="fake",
        max_workers=3,
    )
    elapsed = _time.monotonic() - t0

    sequential_baseline = chunk_count * (encode_secs + upload_secs)
    print(f"  elapsed: {elapsed:.2f}s, sequential baseline: {sequential_baseline:.2f}s")

    # If overlap works, elapsed should be well under the sequential
    # baseline (encoding + uploading every chunk back-to-back). Allow
    # generous slack for CI jitter; even on a slow machine the overlap
    # should easily beat 80% of the baseline.
    assert elapsed < sequential_baseline * 0.8, (
        f"Streaming pipeline did not overlap encoding with upload "
        f"(elapsed={elapsed:.2f}s vs baseline={sequential_baseline:.2f}s)"
    )

    # Result still in chunk-index order
    assert "text-from-chunk_0.mp3" in result["text"]
    assert "text-from-chunk_3.mp3" in result["text"]
    assert result["text"].index("chunk_0") < result["text"].index("chunk_3")
    print("  ✅ Encoding and uploading overlap correctly")


def test_parallel_transcribe_chunks_graceful_degradation(monkeypatch):
    """A failing chunk in the middle must not abort the whole job — the
    surviving chunks are merged and the failure index is reported."""
    separator("Parallel transcribe_chunks graceful degradation")

    def fake_transcribe_single(file_path, provider, config, client, language, diarize, low_confidence_threshold=None):
        if file_path == "boom.mp3":
            raise RuntimeError("simulated transient failure")
        return {"text": f"text from {file_path}", "detected_language": None}

    monkeypatch.setattr(cloud_engine, "_transcribe_single", fake_transcribe_single)
    monkeypatch.setattr(cloud_engine, "_create_client", lambda provider, key: object())

    result = cloud_engine.transcribe_chunks(
        ["ok1.mp3", "boom.mp3", "ok2.mp3"],
        provider="OpenAI Whisper API",
        api_key="fake",
        max_workers=3,
    )

    assert result["failed_chunks"] == [1], "Should report index 1 as failed"
    # Surviving chunks merged in order
    assert "text from ok1.mp3" in result["text"]
    assert "text from ok2.mp3" in result["text"]
    # Failure should surface as a quality warning
    assert any("Chunk 2 failed" in w for w in result["quality_warnings"])
    print("  ✅ Graceful degradation works under parallel execution")


# Run with: uv run pytest tests/
# The previous __main__ block omitted the Deepgram paragraphs/words
# formatter tests from its hand-curated list, so running this file
# directly produced incomplete coverage. Pytest discovery picks up
# every test_* function automatically.