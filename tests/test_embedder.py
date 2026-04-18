import numpy as np
from brain_mcp.indexer.embedder import EmbeddingBackend, SentenceTransformerBackend


def test_mock_embedder_implements_protocol(mock_embedder):
    assert isinstance(mock_embedder.dimension, int)
    assert mock_embedder.dimension == 384


def test_mock_embedder_returns_correct_shape(mock_embedder):
    result = mock_embedder.embed(["hello", "world"])
    assert result.shape == (2, 384)
    assert result.dtype == np.float32


def test_mock_embedder_normalized(mock_embedder):
    result = mock_embedder.embed(["test"])
    norm = np.linalg.norm(result[0])
    assert abs(norm - 1.0) < 1e-5


def test_mock_embedder_deterministic(mock_embedder):
    r1 = mock_embedder.embed(["hello"])
    r2 = mock_embedder.embed(["hello"])
    np.testing.assert_array_equal(r1, r2)


def test_mock_embedder_different_texts_differ(mock_embedder):
    r1 = mock_embedder.embed(["hello"])
    r2 = mock_embedder.embed(["world"])
    assert not np.allclose(r1, r2)


def test_sentence_transformer_backend_has_protocol_methods():
    assert hasattr(SentenceTransformerBackend, "embed")
    assert hasattr(SentenceTransformerBackend, "dimension")
