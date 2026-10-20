"""
Tests for SentenceTransformerEmbedding.encode_batch().

Verifies:
- encode_batch returns the same vectors as N individual encode() calls
- encode_batch is faster than N individual encode() calls
"""

import time

import pytest

from th2rag.rag.embeddings.sentence_transformer import SentenceTransformerEmbedding


SAMPLE_TEXTS = [
    "The organization identifies and manages cybersecurity risk.",
    "Asset management includes hardware and software inventory.",
    "Access control policies restrict unauthorized access to systems.",
    "Backups of information are conducted, maintained, and tested.",
    "Recovery planning ensures restoration of systems after an incident.",
    "Anomalies and events are detected and their potential impact assessed.",
    "Protective technology is managed to ensure security and resilience.",
    "The organization communicates cybersecurity roles and responsibilities.",
    "Supply chain risk management processes are established and monitored.",
    "Identity proofing and binding operations are performed for new entities.",
]


@pytest.fixture(scope="module")
def embedder():
    return SentenceTransformerEmbedding()


def test_encode_batch_same_result_as_individual(embedder):
    """encode_batch must produce identical vectors to N encode() calls."""
    individual = [embedder.encode(t) for t in SAMPLE_TEXTS]
    batched = embedder.encode_batch(SAMPLE_TEXTS)

    assert len(batched) == len(SAMPLE_TEXTS)
    for i, (ind, bat) in enumerate(zip(individual, batched)):
        assert len(ind) == len(bat), f"Dimension mismatch at index {i}"
        for j, (a, b) in enumerate(zip(ind, bat)):
            assert abs(a - b) < 1e-5, (
                f"Vector mismatch at text {i}, dim {j}: {a} vs {b}"
            )


def test_encode_batch_returns_correct_count(embedder):
    texts = SAMPLE_TEXTS[:5]
    result = embedder.encode_batch(texts)
    assert len(result) == 5


def test_encode_batch_returns_correct_dimension(embedder):
    result = embedder.encode_batch(SAMPLE_TEXTS[:3])
    dim = embedder.get_embedding_dimension()
    for vec in result:
        assert len(vec) == dim


def test_encode_batch_faster_than_individual(embedder):
    """
    Batch encoding must be faster than N sequential encode() calls.
    Uses 50 texts to make the difference measurable.
    """
    texts = (SAMPLE_TEXTS * 5)[:50]

    start = time.perf_counter()
    for t in texts:
        embedder.encode(t)
    individual_time = time.perf_counter() - start

    start = time.perf_counter()
    embedder.encode_batch(texts)
    batch_time = time.perf_counter() - start

    assert batch_time < individual_time, (
        f"encode_batch ({batch_time:.3f}s) was not faster than "
        f"individual encode x{len(texts)} ({individual_time:.3f}s)"
    )
