from brain_mcp.indexer.reranker import CrossEncoderReRanker
from brain_mcp.tools.retrieve import ReRanker


def test_reranker_implements_protocol():
    r = CrossEncoderReRanker()
    assert isinstance(r, ReRanker)


def test_rerank_empty_candidates():
    r = CrossEncoderReRanker()
    assert r.rerank("test", []) == []


def test_rerank_few_candidates_passthrough():
    r = CrossEncoderReRanker()
    candidates = [{"title": "A", "snippet": "aaa"}]
    assert r.rerank("query", candidates) == candidates
    candidates2 = [{"title": "A", "snippet": "aaa"}, {"title": "B", "snippet": "bbb"}]
    assert r.rerank("query", candidates2) == candidates2


def test_rerank_model_not_ready_passthrough():
    r = CrossEncoderReRanker()
    candidates = [
        {"title": "A", "snippet": "aaa"},
        {"title": "B", "snippet": "bbb"},
        {"title": "C", "snippet": "ccc"},
    ]
    result = r.rerank("query", candidates)
    assert result == candidates


def test_rerank_with_mock_model():
    """Test that reranking reorders candidates when model is available."""
    r = CrossEncoderReRanker()

    class FakeModel:
        def predict(self, pairs, batch_size=32, show_progress_bar=False):
            return [len(p[1]) for p in pairs]

    r._model = FakeModel()
    r._ready.set()

    candidates = [
        {"title": "Short", "snippet": "ab"},
        {"title": "Medium", "snippet": "abcdefgh"},
        {"title": "Long", "snippet": "abcdefghijklmnop"},
    ]
    result = r.rerank("query", candidates)
    assert result[0]["title"] == "Long"
    assert result[1]["title"] == "Medium"
    assert result[2]["title"] == "Short"


def test_rerank_preserves_beyond_top_n():
    """Candidates beyond RERANK_TOP_N are appended unchanged."""
    from brain_mcp.indexer.reranker import RERANK_TOP_N

    r = CrossEncoderReRanker()

    class FakeModel:
        def predict(self, pairs, batch_size=32, show_progress_bar=False):
            return [1.0] * len(pairs)

    r._model = FakeModel()
    r._ready.set()

    candidates = [{"title": f"Note {i}", "snippet": f"content {i}"} for i in range(RERANK_TOP_N + 5)]
    result = r.rerank("query", candidates)
    assert len(result) == RERANK_TOP_N + 5
    assert result[RERANK_TOP_N:] == candidates[RERANK_TOP_N:]
