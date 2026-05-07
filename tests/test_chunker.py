from brain_mcp.indexer.chunker import split_into_chunks, CHUNK_THRESHOLD_WORDS, MIN_CHUNK_WORDS


def _make_long_content(word_count: int, headings: list[str] | None = None) -> str:
    if headings is None:
        headings = ["## Section A", "## Section B", "## Section C"]
    words_per_section = word_count // (len(headings) + 1)
    filler = " ".join(["word"] * words_per_section)
    parts = [f"Intro paragraph.\n{filler}"]
    for h in headings:
        parts.append(f"\n\n{h}\n\n{filler}")
    return "\n".join(parts)


def test_short_note_no_chunks():
    content = "Short note with only a few words."
    result = split_into_chunks(content, "Short Note")
    assert result == []


def test_long_note_splits_at_headings():
    content = _make_long_content(600, ["## Architecture", "## Testing"])
    result = split_into_chunks(content, "My Note")
    assert len(result) >= 2
    headings = [c["heading"] for c in result]
    assert "Architecture" in headings or "My Note" in headings


def test_chunk_has_required_fields():
    content = _make_long_content(600)
    result = split_into_chunks(content, "Test Note")
    assert len(result) > 0
    chunk = result[0]
    assert "heading" in chunk
    assert "content" in chunk
    assert "content_hash" in chunk
    assert "word_count" in chunk
    assert "chunk_idx" in chunk


def test_chunk_idx_sequential():
    content = _make_long_content(800, ["## A", "## B", "## C"])
    result = split_into_chunks(content, "Note")
    indices = [c["chunk_idx"] for c in result]
    assert indices == list(range(len(indices)))


def test_no_headings_returns_empty():
    content = " ".join(["word"] * 600)
    result = split_into_chunks(content, "Plain Note")
    assert result == []


def test_small_first_chunk_merges_forward():
    content = "# Title\nShort intro.\n\n## Big Section\n\n" + " ".join(["word"] * 550)
    result = split_into_chunks(content, "Note")
    if result:
        assert result[0]["word_count"] >= MIN_CHUNK_WORDS


def test_small_chunk_merges_backward():
    big = " ".join(["word"] * 300)
    small = "tiny"
    content = f"## Section 1\n\n{big}\n\n## Section 2\n\n{big}\n\n## Section 3\n\n{small}"
    result = split_into_chunks(content, "Note")
    for chunk in result:
        assert chunk["word_count"] >= MIN_CHUNK_WORDS or len(result) == 1


def test_content_hash_deterministic():
    content = _make_long_content(600)
    r1 = split_into_chunks(content, "Note")
    r2 = split_into_chunks(content, "Note")
    assert [c["content_hash"] for c in r1] == [c["content_hash"] for c in r2]


def test_h3_headings_also_split():
    parts = ["Intro.\n" + " ".join(["word"] * 200)]
    parts.append("### Sub A\n\n" + " ".join(["word"] * 200))
    parts.append("### Sub B\n\n" + " ".join(["word"] * 200))
    content = "\n\n".join(parts)
    result = split_into_chunks(content, "Note")
    assert len(result) >= 2
