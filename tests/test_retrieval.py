"""Tests for retrieval.py -- embedding math and top-k reference retrieval.

These load the real all-MiniLM-L6-v2 model (first run downloads it, ~90 MB).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from retrieval import Embedder, ReferenceLibrary, cosine_distance, cosine_similarity  # noqa: E402


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


@pytest.fixture(scope="module")
def library(embedder):
    return ReferenceLibrary.from_json(config.REFERENCE_LIBRARY_PATH, embedder)


def test_cosine_distance_identity():
    v = np.array([0.3, -0.7, 0.1, 0.9], dtype=np.float32)
    assert cosine_distance(v, v) == pytest.approx(0.0, abs=1e-6)


def test_cosine_distance_range_and_opposite():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([-1.0, 0.0], dtype=np.float32)
    assert cosine_distance(a, b) == pytest.approx(2.0, abs=1e-6)
    assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)


def test_cosine_distance_handles_zero_vector():
    a = np.zeros(4, dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    assert cosine_distance(a, b) == pytest.approx(1.0)


def test_embedder_returns_unit_vectors(embedder):
    v = embedder.encode("a dog runs across the grass")
    assert v.shape[0] > 0
    assert float(np.linalg.norm(v)) == pytest.approx(1.0, abs=1e-3)


def test_near_paraphrase_is_closer_than_unrelated(embedder):
    base = embedder.encode("A dog sprints across a grassy field chasing a ball.")
    para = embedder.encode("A dog runs over the lawn going after a ball.")
    off = embedder.encode("A mechanical watch movement ticks in macro close-up.")
    assert cosine_distance(base, para) < cosine_distance(base, off)


def test_library_loads_all_entries(library):
    assert len(library) >= 15


def test_top_k_returns_k_sorted_desc(library):
    hits = library.top_k_similar("a dog running across a field", k=3)
    assert len(hits) == 3
    sims = [h.similarity for h in hits]
    assert sims == sorted(sims, reverse=True)


def test_top_k_retrieves_on_topic_reference(library):
    hits = library.top_k_similar(
        "A golden retriever sprints across a grassy field chasing a ball.", k=3
    )
    cats = {h.category for h in hits}
    assert "animals-nature" in cats
    # the dedicated dog-on-grass reference should be the top hit
    assert hits[0].id == "nature-01"


def test_query_specificity_changes_ranking(library):
    vague = library.top_k_similar("a person doing something", k=3)
    specific = library.top_k_similar(
        "A barista steams milk and pours latte art for a customer.", k=3
    )
    assert specific[0].similarity > vague[0].similarity
