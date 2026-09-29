"""CPU-only tests for cross-page character identity resolution.

Coverage (18 cases):
 1.  strong cross-page match
 2.  weak cross-page mismatch
 3.  face + body match
 4.  body-only match
 5.  missing face
 6.  occluded / no-crop character
 7.  multiple characters on same page
 8.  similar-looking characters (same-page anti-collapse)
 9.  one page missing a character
10.  unresolved character (singleton)
11.  ambiguous edge
12.  deterministic cluster ordering
13.  deterministic anonymous labels (A, B, C)
14.  same-page anti-collapse constraint
15.  speaker instance → identity mapping
16.  provider failure (embed returns None)
17.  missing embedding (NullProvider)
18.  cyclic / conflicting graph evidence
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from app.character_identity import (
    CharacterIdentityResolver,
    _IdentityGraph,
    _UnionFind,
    compute_pair_evidence,
    _EmbeddingCache,
)
from app.models.character_crops import CharacterCropPaths
from app.models.character_embeddings import (
    NullEmbeddingProvider,
    _cosine_similarity,
    _l2_norm,
)
from app.schemas.character_identity import CharacterPairEvidence
from app.schemas.page import BoundingBox, CharacterInstance, PageRepresentation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bb(x1, y1, x2, y2) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def make_char(cid: str, panel_id: str | None = None) -> CharacterInstance:
    return CharacterInstance(id=cid, bbox=bb(10, 10, 50, 80), panel_id=panel_id)


def make_page(page_index: int, chars: list[CharacterInstance]) -> PageRepresentation:
    return PageRepresentation(
        page_index=page_index,
        image_path="unused.png",
        characters=chars,
    )


def make_crops(
    tmp_path: Path,
    cid: str,
    *,
    write_files: bool = True,
) -> CharacterCropPaths:
    """Create minimal stub PNG crops and return CharacterCropPaths."""
    import cv2
    import numpy as np

    char_dir = tmp_path / cid
    char_dir.mkdir(parents=True, exist_ok=True)
    char_path = char_dir / "character.png"
    face_path = char_dir / "face.png"
    body_path = char_dir / "body.png"
    if write_files:
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        for p in (char_path, face_path, body_path):
            cv2.imwrite(str(p), img)
    return CharacterCropPaths(
        character=char_path,
        face=face_path,
        body=body_path,
    )


class FixedEmbeddingProvider:
    """Returns a fixed pre-computed embedding for each character id."""

    def __init__(self, embeddings: dict[str, list[float]]) -> None:
        self._emb: dict[str, list[float]] = {
            k: _l2_norm(v) for k, v in embeddings.items()
        }

    def embed(self, image_path: Path) -> list[float] | None:
        # Key on the parent directory name (= character id)
        cid = image_path.parent.name
        return self._emb.get(cid)


class FailingProvider:
    """Always raises on embed()."""

    def embed(self, image_path: Path) -> list[float] | None:
        raise RuntimeError("embedding backend crashed")


# ---------------------------------------------------------------------------
# Test 1: strong cross-page match
# ---------------------------------------------------------------------------

def test_strong_cross_page_match_produces_matched_cluster(tmp_path):
    c0 = make_char("c0")
    c1 = make_char("c1")
    crops = {
        "c0": make_crops(tmp_path / "c0", "c0"),
        "c1": make_crops(tmp_path / "c1", "c1"),
    }
    # Same embedding → cosine = 1.0 → similarity = 1.0
    provider = FixedEmbeddingProvider({"c0": [1.0, 0.0, 0.0], "c1": [1.0, 0.0, 0.0]})
    resolver = CharacterIdentityResolver(provider, match_threshold=0.65)
    pages = [make_page(0, [c0]), make_page(1, [c1]), make_page(2, [])]
    result = resolver.resolve("seq", pages, crops)

    matched = [cl for cl in result.clusters if cl.state == "matched"]
    assert len(matched) == 1
    ids = {m.character_id for m in matched[0].members}
    assert ids == {"c0", "c1"}


# ---------------------------------------------------------------------------
# Test 2: weak cross-page mismatch → not merged
# ---------------------------------------------------------------------------

def test_weak_cross_page_mismatch_not_merged(tmp_path):
    c0 = make_char("c0")
    c1 = make_char("c1")
    crops = {
        "c0": make_crops(tmp_path / "c0", "c0"),
        "c1": make_crops(tmp_path / "c1", "c1"),
    }
    # Orthogonal embeddings → cosine = 0.0 → similarity ~ 0.5 but combined low
    provider = FixedEmbeddingProvider({"c0": [1.0, 0.0, 0.0], "c1": [0.0, 1.0, 0.0]})
    resolver = CharacterIdentityResolver(provider, match_threshold=0.65, ambiguity_low=0.45)
    pages = [make_page(0, [c0]), make_page(1, [c1]), make_page(2, [])]
    result = resolver.resolve("seq", pages, crops)

    matched = [cl for cl in result.clusters if cl.state == "matched"]
    assert len(matched) == 0
    assert len(result.clusters) == 2  # two singletons


# ---------------------------------------------------------------------------
# Test 3: face + body both match → high confidence
# ---------------------------------------------------------------------------

def test_face_and_body_match_yields_combined_embedding_method(tmp_path):
    c0 = make_char("c0")
    c1 = make_char("c1")
    crops = {
        "c0": make_crops(tmp_path / "c0", "c0"),
        "c1": make_crops(tmp_path / "c1", "c1"),
    }
    provider = FixedEmbeddingProvider({"c0": [0.9, 0.1, 0.0], "c1": [0.9, 0.1, 0.0]})
    resolver = CharacterIdentityResolver(provider, match_threshold=0.65)
    pages = [make_page(0, [c0]), make_page(1, [c1]), make_page(2, [])]
    result = resolver.resolve("seq", pages, crops)

    ev = result.pair_evidence[0]
    assert ev.method == "combined_embedding"
    assert ev.confidence >= 0.75


# ---------------------------------------------------------------------------
# Test 4: body-only match (no face crop)
# ---------------------------------------------------------------------------

def test_body_only_match_still_resolves(tmp_path):
    c0 = make_char("c0")
    c1 = make_char("c1")

    # Only body + character crops, no face file
    def _make_no_face(cid: str) -> CharacterCropPaths:
        import cv2
        import numpy as np
        d = tmp_path / cid
        d.mkdir(parents=True, exist_ok=True)
        img = np.full((32, 32, 3), 120, dtype=np.uint8)
        char_p = d / "character.png"
        body_p = d / "body.png"
        cv2.imwrite(str(char_p), img)
        cv2.imwrite(str(body_p), img)
        return CharacterCropPaths(character=char_p, face=None, body=body_p)

    crops = {"c0": _make_no_face("c0"), "c1": _make_no_face("c1")}
    provider = FixedEmbeddingProvider({"c0": [1.0, 0.0], "c1": [1.0, 0.0]})
    resolver = CharacterIdentityResolver(provider, match_threshold=0.65)
    pages = [make_page(0, [c0]), make_page(1, [c1]), make_page(2, [])]
    result = resolver.resolve("seq", pages, crops)

    matched = [cl for cl in result.clusters if cl.state == "matched"]
    assert len(matched) == 1


# ---------------------------------------------------------------------------
# Test 5: missing face → still produces evidence with face_similarity=None
# ---------------------------------------------------------------------------

def test_missing_face_produces_none_face_similarity(tmp_path):
    c0 = make_char("c0")
    c1 = make_char("c1")

    import cv2, numpy as np

    def _no_face(cid):
        d = tmp_path / cid
        d.mkdir(parents=True, exist_ok=True)
        img = np.full((32, 32, 3), 100, dtype=np.uint8)
        char_p = d / "character.png"
        cv2.imwrite(str(char_p), img)
        return CharacterCropPaths(character=char_p)

    crops = {"c0": _no_face("c0"), "c1": _no_face("c1")}
    provider = FixedEmbeddingProvider({"c0": [1.0, 0.0], "c1": [1.0, 0.0]})

    cache = _EmbeddingCache(provider)
    ev = compute_pair_evidence(
        c0, 0, crops["c0"],
        c1, 1, crops["c1"],
        cache,
    )
    assert ev.face_similarity is None
    assert ev.body_similarity is None
    assert ev.full_similarity is not None  # character.png exists


# ---------------------------------------------------------------------------
# Test 6: occluded / no crop → evidence diagnostics recorded
# ---------------------------------------------------------------------------

def test_no_crop_character_gets_no_embedding_diagnostic(tmp_path):
    c0 = make_char("c0")
    c1 = make_char("c1")
    # Only c0 has crops; c1 has none
    crops = {"c0": make_crops(tmp_path / "c0", "c0")}
    provider = FixedEmbeddingProvider({"c0": [1.0, 0.0, 0.0]})
    cache = _EmbeddingCache(provider)
    ev = compute_pair_evidence(c0, 0, crops.get("c0"), c1, 1, crops.get("c1"), cache)
    assert any("no_crops_for" in d for d in ev.diagnostics)
    assert ev.confidence <= 0.55


# ---------------------------------------------------------------------------
# Test 7: multiple characters on same page → all appear in result
# ---------------------------------------------------------------------------

def test_multiple_characters_same_page_all_clustered(tmp_path):
    c0a = make_char("c0a")
    c0b = make_char("c0b")
    c1a = make_char("c1a")
    crops = {
        "c0a": make_crops(tmp_path / "c0a", "c0a"),
        "c0b": make_crops(tmp_path / "c0b", "c0b"),
        "c1a": make_crops(tmp_path / "c1a", "c1a"),
    }
    # c0a similar to c1a; c0b dissimilar
    provider = FixedEmbeddingProvider({
        "c0a": [1.0, 0.0, 0.0],
        "c0b": [0.0, 1.0, 0.0],
        "c1a": [1.0, 0.0, 0.0],
    })
    resolver = CharacterIdentityResolver(provider, match_threshold=0.65)
    pages = [make_page(0, [c0a, c0b]), make_page(1, [c1a]), make_page(2, [])]
    result = resolver.resolve("seq", pages, crops)

    all_ids = {m.character_id for cl in result.clusters for m in cl.members}
    assert "c0a" in all_ids
    assert "c0b" in all_ids
    assert "c1a" in all_ids
    assert len(result.clusters) >= 2


# ---------------------------------------------------------------------------
# Test 8: same-page anti-collapse constraint
# ---------------------------------------------------------------------------

def test_same_page_characters_never_collapsed_into_one_cluster(tmp_path):
    """Two identical-looking characters on the same page must stay in separate clusters."""
    c0a = make_char("c0a")
    c0b = make_char("c0b")  # same page as c0a
    crops = {
        "c0a": make_crops(tmp_path / "c0a", "c0a"),
        "c0b": make_crops(tmp_path / "c0b", "c0b"),
    }
    # Identical embeddings
    provider = FixedEmbeddingProvider({"c0a": [1.0, 0.0], "c0b": [1.0, 0.0]})
    resolver = CharacterIdentityResolver(provider)
    pages = [make_page(0, [c0a, c0b]), make_page(1, []), make_page(2, [])]
    result = resolver.resolve("seq", pages, crops)

    # They must be in different clusters
    cluster_of_c0a = result.character_to_identity.get("c0a")
    cluster_of_c0b = result.character_to_identity.get("c0b")
    assert cluster_of_c0a != cluster_of_c0b


def test_union_find_same_page_constraint_prevents_merge():
    uf = _UnionFind()
    uf.add("a", page=0)
    uf.add("b", page=0)
    # Same page → must not merge
    assert uf.can_merge("a", "b") is False
    merged = uf.union("a", "b")
    assert merged is False
    assert uf.find("a") != uf.find("b")


# ---------------------------------------------------------------------------
# Test 9: one page missing a character → still resolves others
# ---------------------------------------------------------------------------

def test_one_page_missing_character_does_not_break_resolution(tmp_path):
    c0 = make_char("c0")
    c2 = make_char("c2")
    crops = {
        "c0": make_crops(tmp_path / "c0", "c0"),
        "c2": make_crops(tmp_path / "c2", "c2"),
    }
    provider = FixedEmbeddingProvider({"c0": [1.0, 0.0], "c2": [1.0, 0.0]})
    resolver = CharacterIdentityResolver(provider, match_threshold=0.65)
    # Page 1 is empty
    pages = [make_page(0, [c0]), make_page(1, []), make_page(2, [c2])]
    result = resolver.resolve("seq", pages, crops)

    # c0 and c2 should be matched
    assert result.character_to_identity["c0"] == result.character_to_identity["c2"]


# ---------------------------------------------------------------------------
# Test 10: unresolved character → singleton, state="unmatched"
# ---------------------------------------------------------------------------

def test_unresolved_character_is_singleton_unmatched():
    c0 = make_char("c0")
    c1 = make_char("c1")
    # No crops → no embeddings → combined_score = 0.0 < ambiguity_low
    resolver = CharacterIdentityResolver(NullEmbeddingProvider(), match_threshold=0.65)
    pages = [make_page(0, [c0]), make_page(1, [c1]), make_page(2, [])]
    result = resolver.resolve("seq", pages)

    singletons = [cl for cl in result.clusters if cl.state == "unmatched"]
    assert len(singletons) == 2


# ---------------------------------------------------------------------------
# Test 11: ambiguous edge → state="ambiguous"
# ---------------------------------------------------------------------------

def test_ambiguous_edge_yields_ambiguous_state(tmp_path):
    c0 = make_char("c0")
    c1 = make_char("c1")
    crops = {
        "c0": make_crops(tmp_path / "c0", "c0"),
        "c1": make_crops(tmp_path / "c1", "c1"),
    }
    # cosine ≈ 0.5 → (0.5+1)/2 = 0.75 → in ambiguous window [0.45, 0.65)
    # Use carefully chosen vectors: dot ≈ 0.0 → similarity ≈ 0.5
    provider = FixedEmbeddingProvider({
        "c0": [1.0, 0.0, 0.0],
        "c1": [0.0, 1.0, 0.0],  # orthogonal → cosine=0 → sim=0.5
    })
    resolver = CharacterIdentityResolver(
        provider,
        match_threshold=0.65,
        ambiguity_low=0.45,
        ambiguity_high=0.65,
    )
    pages = [make_page(0, [c0]), make_page(1, [c1]), make_page(2, [])]
    result = resolver.resolve("seq", pages, crops)

    # Should be ambiguous (not merged), and diagnostic should mention it
    ambiguous = [cl for cl in result.clusters if cl.state == "ambiguous"]
    ambiguous_diags = [d for d in result.diagnostics if d.code == "ambiguous_edge"]
    assert len(ambiguous_diags) > 0 or len(ambiguous) > 0  # at least one signal


# ---------------------------------------------------------------------------
# Test 12: deterministic cluster ordering
# ---------------------------------------------------------------------------

def test_cluster_ordering_is_deterministic_regardless_of_input_order(tmp_path):
    c0 = make_char("c0")
    c1 = make_char("c1")
    crops = {
        "c0": make_crops(tmp_path / "c0", "c0"),
        "c1": make_crops(tmp_path / "c1", "c1"),
    }
    provider = FixedEmbeddingProvider({"c0": [1.0, 0.0], "c1": [1.0, 0.0]})
    resolver = CharacterIdentityResolver(provider, match_threshold=0.65)
    pages_ab = [make_page(0, [c0]), make_page(1, [c1]), make_page(2, [])]
    pages_ba = [make_page(0, [c1]), make_page(1, [c0]), make_page(2, [])]

    result_ab = resolver.resolve("seq", pages_ab, crops)
    result_ba = resolver.resolve("seq", pages_ba, crops)

    labels_ab = sorted(result_ab.character_to_label.values())
    labels_ba = sorted(result_ba.character_to_label.values())
    assert labels_ab == labels_ba


# ---------------------------------------------------------------------------
# Test 13: deterministic anonymous labels A, B, C
# ---------------------------------------------------------------------------

def test_anonymous_labels_are_A_B_C_in_order():
    chars = [make_char(f"c{i}") for i in range(3)]
    pages = [
        make_page(0, [chars[0]]),
        make_page(1, [chars[1]]),
        make_page(2, [chars[2]]),
    ]
    resolver = CharacterIdentityResolver(NullEmbeddingProvider())
    result = resolver.resolve("seq", pages)
    labels = sorted(cl.label for cl in result.clusters)
    assert labels == ["A", "B", "C"]


def test_label_generator_beyond_Z():
    """Labels past Z should be AA, AB, …"""
    from app.character_identity import _assign_labels, _IdentityGraph
    # Build a graph with 27 singleton nodes across 27 pages
    graph = _IdentityGraph()
    for i in range(27):
        graph.add_node(f"c{i}", i)
    components = {f"c{i}": [f"c{i}"] for i in range(27)}
    clusters = _assign_labels(
        components, graph, ambiguity_low=0.45, ambiguity_high=0.65
    )
    labels = [cl.label for cl in clusters]
    assert labels[25] == "Z"
    assert labels[26] == "AA"


# ---------------------------------------------------------------------------
# Test 14: same-page anti-collapse  (graph-level, 3 chars)
# ---------------------------------------------------------------------------

def test_three_chars_same_page_all_in_separate_clusters(tmp_path):
    chars = [make_char(f"c{i}") for i in range(3)]
    crops = {f"c{i}": make_crops(tmp_path / f"c{i}", f"c{i}") for i in range(3)}
    # All identical embeddings
    provider = FixedEmbeddingProvider({f"c{i}": [1.0, 0.0] for i in range(3)})
    resolver = CharacterIdentityResolver(provider, match_threshold=0.65)
    pages = [make_page(0, chars), make_page(1, []), make_page(2, [])]
    result = resolver.resolve("seq", pages, crops)

    assert len(result.clusters) == 3
    ids_per_cluster = [len(cl.members) for cl in result.clusters]
    assert all(n == 1 for n in ids_per_cluster)


# ---------------------------------------------------------------------------
# Test 15: speaker instance → identity mapping
# ---------------------------------------------------------------------------

def test_speaker_instance_maps_to_identity_via_character_to_label(tmp_path):
    c0 = make_char("c0")
    c1 = make_char("c1")
    crops = {
        "c0": make_crops(tmp_path / "c0", "c0"),
        "c1": make_crops(tmp_path / "c1", "c1"),
    }
    provider = FixedEmbeddingProvider({"c0": [1.0, 0.0], "c1": [1.0, 0.0]})
    resolver = CharacterIdentityResolver(provider, match_threshold=0.65)
    pages = [make_page(0, [c0]), make_page(1, [c1]), make_page(2, [])]
    result = resolver.resolve("seq", pages, crops)

    # Both should map to the same label
    assert result.character_to_label["c0"] == result.character_to_label["c1"]
    assert result.character_to_label["c0"] is not None


# ---------------------------------------------------------------------------
# Test 16: provider failure (embed raises) → treated as None embedding
# ---------------------------------------------------------------------------

def test_provider_failure_handled_gracefully(tmp_path):
    c0 = make_char("c0")
    c1 = make_char("c1")
    crops = {
        "c0": make_crops(tmp_path / "c0", "c0"),
        "c1": make_crops(tmp_path / "c1", "c1"),
    }
    resolver = CharacterIdentityResolver(FailingProvider(), match_threshold=0.65)
    pages = [make_page(0, [c0]), make_page(1, [c1]), make_page(2, [])]
    # Must not raise; should return unmatched singletons
    result = resolver.resolve("seq", pages, crops)
    assert len(result.clusters) == 2
    assert all(cl.state == "unmatched" for cl in result.clusters)


# ---------------------------------------------------------------------------
# Test 17: missing embedding (NullProvider) → singletons, diagnostics
# ---------------------------------------------------------------------------

def test_null_provider_produces_unmatched_singletons():
    chars = [make_char(f"c{i}") for i in range(2)]
    pages = [make_page(0, [chars[0]]), make_page(1, [chars[1]]), make_page(2, [])]
    result = CharacterIdentityResolver(NullEmbeddingProvider()).resolve("seq", pages)

    assert len(result.clusters) == 2
    assert all(cl.state == "unmatched" for cl in result.clusters)


# ---------------------------------------------------------------------------
# Test 18: cyclic / conflicting graph evidence
# ---------------------------------------------------------------------------

def test_conflicting_edges_only_strongest_kept_in_graph():
    """When multiple evidence objects are added for the same pair, the graph
    keeps only the strongest combined_score."""
    graph = _IdentityGraph()
    graph.add_node("a", 0)
    graph.add_node("b", 1)

    ev_weak = CharacterPairEvidence(
        first_character_id="a", second_character_id="b",
        combined_score=0.3, confidence=0.6,
        same_page=False, first_page_index=0, second_page_index=1,
    )
    ev_strong = CharacterPairEvidence(
        first_character_id="a", second_character_id="b",
        combined_score=0.8, confidence=0.9,
        same_page=False, first_page_index=0, second_page_index=1,
    )
    graph.add_or_update_edge(ev_weak)
    graph.add_or_update_edge(ev_strong)

    edges = graph.edges
    assert len(edges) == 1
    assert edges[0].combined_score == 0.8


def test_three_way_cross_page_cycle_resolves_without_error(tmp_path):
    """A → B → C → A cross-page similarity pattern resolves cleanly."""
    c0, c1, c2 = make_char("c0"), make_char("c1"), make_char("c2")
    crops = {f"c{i}": make_crops(tmp_path / f"c{i}", f"c{i}") for i in range(3)}
    # All similar: c0≈c1≈c2 but on different pages
    provider = FixedEmbeddingProvider({f"c{i}": [1.0, 0.0] for i in range(3)})
    resolver = CharacterIdentityResolver(provider, match_threshold=0.65)
    pages = [make_page(0, [c0]), make_page(1, [c1]), make_page(2, [c2])]
    result = resolver.resolve("seq", pages, crops)

    # All three should end up in one matched cluster
    matched = [cl for cl in result.clusters if cl.state == "matched"]
    assert len(matched) == 1
    ids = {m.character_id for m in matched[0].members}
    assert ids == {"c0", "c1", "c2"}


# ---------------------------------------------------------------------------
# Embedding utility tests
# ---------------------------------------------------------------------------

def test_cosine_similarity_of_identical_normalised_vectors():
    a = _l2_norm([3.0, 4.0, 0.0])
    assert abs(_cosine_similarity(a, a) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal_vectors():
    a = _l2_norm([1.0, 0.0])
    b = _l2_norm([0.0, 1.0])
    assert abs(_cosine_similarity(a, b)) < 1e-6


def test_null_provider_returns_none():
    provider = NullEmbeddingProvider()
    assert provider.embed(Path("anything.png")) is None


def test_l2_norm_zero_vector_unchanged():
    z = [0.0, 0.0, 0.0]
    assert _l2_norm(z) == z


def test_embedding_cache_calls_provider_once_per_path(tmp_path):
    """The cache should not call embed() more than once for the same path."""
    call_count = [0]

    class CountingProvider:
        def embed(self, image_path):
            call_count[0] += 1
            return [1.0, 0.0]

    cache = _EmbeddingCache(CountingProvider())
    p = tmp_path / "crop.png"
    p.write_bytes(b"")
    _ = cache.get(p)
    _ = cache.get(p)
    assert call_count[0] == 1


def test_sequence_character_identity_schema_validates():
    from app.schemas.character_identity import (
        CharacterIdentityCluster,
        IdentityMember,
        SequenceCharacterIdentity,
    )
    cluster = CharacterIdentityCluster(
        identity_id="identity-001",
        label="A",
        members=[IdentityMember(character_id="c0", page_index=0)],
        confidence=0.9,
        state="matched",
    )
    sci = SequenceCharacterIdentity(
        sequence_id="seq-test",
        clusters=[cluster],
        character_to_identity={"c0": "identity-001"},
        character_to_label={"c0": "A"},
    )
    assert sci.clusters[0].label == "A"
    assert sci.character_to_label["c0"] == "A"
