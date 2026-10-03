"""Sequence-level cross-page character identity resolution.

Pipeline:
    page-scoped CharacterInstances (3 pages)
        ↓
    per-instance visual embeddings (full / face / body crop)
        ↓
    cross-page pairwise similarity evidence
        ↓
    identity graph  (nodes = instances, edges = similarity scores)
        ↓
    constrained clustering  (at most one instance per page per cluster)
        ↓
    anonymous deterministic labels  (Character A, Character B, …)
        ↓
    SequenceCharacterIdentity

Constraints:
    - Same-page instances are NEVER auto-collapsed into one identity.
    - Weak edges (score < match_threshold) are discarded.
    - Ambiguous edges (score in [ambiguity_low, ambiguity_high]) carry uncertainty.
    - Cycles / conflicts are resolved by keeping only the strongest edge per node-pair.
    - The embedding provider is fully injectable.
"""

from __future__ import annotations

import math
import string
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol, Sequence

from app.models.character_crops import CharacterCropPaths
from app.models.character_embeddings import (
    CharacterEmbeddingProvider,
    NullEmbeddingProvider,
    _cosine_similarity,
    _l2_norm,
)
from app.schemas.character_identity import (
    CharacterIdentityCluster,
    CharacterPairEvidence,
    IdentityDiagnostic,
    IdentityMember,
    SequenceCharacterIdentity,
)
from app.schemas.page import CharacterInstance, PageRepresentation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_cosine(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    if len(a) != len(b) or not a:
        return None
    return _cosine_similarity(a, b)


def _sigmoid_compress(score: float, midpoint: float = 0.5, steepness: float = 10.0) -> float:
    """Map raw cosine [-1,1] to [0,1] with sigmoid, centred at midpoint."""
    x = score - midpoint
    return 1.0 / (1.0 + math.exp(-steepness * x))


def _combine_scores(
    full: float | None,
    face: float | None,
    body: float | None,
    *,
    full_weight: float = 1.0,
    face_weight: float = 2.0,
    body_weight: float = 1.5,
) -> tuple[float, str]:
    """Weighted average of available scores; return (score, method_tag)."""
    parts: list[tuple[float, float]] = []
    methods: list[str] = []
    if face is not None:
        parts.append((face, face_weight))
        methods.append("face")
    if body is not None:
        parts.append((body, body_weight))
        methods.append("body")
    if full is not None:
        parts.append((full, full_weight))
        methods.append("full")
    if not parts:
        return 0.0, "none"
    total_weight = sum(w for _, w in parts)
    score = sum(s * w for s, w in parts) / total_weight
    return max(0.0, min(1.0, score)), "+".join(methods)


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

class _EmbeddingCache:
    """Compute and cache embeddings for each crop path, per character instance."""

    def __init__(self, provider: CharacterEmbeddingProvider) -> None:
        self._provider = provider
        self._cache: dict[str, list[float] | None] = {}

    def get(self, path: Path | None) -> list[float] | None:
        if path is None:
            return None
        key = str(path)
        if key not in self._cache:
            try:
                self._cache[key] = self._provider.embed(path)
            except Exception:  # noqa: BLE001
                self._cache[key] = None
        return self._cache[key]


# ---------------------------------------------------------------------------
# Pairwise evidence
# ---------------------------------------------------------------------------

def compute_pair_evidence(
    char_a: CharacterInstance,
    page_a: int,
    crops_a: CharacterCropPaths | None,
    char_b: CharacterInstance,
    page_b: int,
    crops_b: CharacterCropPaths | None,
    cache: _EmbeddingCache,
    *,
    ambiguity_low: float = 0.45,
    ambiguity_high: float = 0.65,
) -> CharacterPairEvidence:
    """Compute all visual evidence for a character pair."""
    same_page = page_a == page_b
    diags: list[str] = []

    # Obtain embeddings for every available crop type
    emb_full_a = cache.get(crops_a.character if crops_a else None)
    emb_full_b = cache.get(crops_b.character if crops_b else None)
    emb_face_a = cache.get(crops_a.face if crops_a else None)
    emb_face_b = cache.get(crops_b.face if crops_b else None)
    emb_body_a = cache.get(crops_a.body if crops_a else None)
    emb_body_b = cache.get(crops_b.body if crops_b else None)

    if emb_full_a is None and crops_a is not None:
        diags.append(f"full_embedding_missing_for_{char_a.id}")
    if emb_full_b is None and crops_b is not None:
        diags.append(f"full_embedding_missing_for_{char_b.id}")
    if crops_a is None:
        diags.append(f"no_crops_for_{char_a.id}")
    if crops_b is None:
        diags.append(f"no_crops_for_{char_b.id}")

    # Raw cosine scores in [-1, 1] → mapped to [0, 1]
    def to_sim(raw: float | None) -> float | None:
        if raw is None:
            return None
        return max(0.0, min(1.0, (raw + 1.0) / 2.0))

    full_sim = to_sim(_safe_cosine(emb_full_a, emb_full_b))
    face_sim = to_sim(_safe_cosine(emb_face_a, emb_face_b))
    body_sim = to_sim(_safe_cosine(emb_body_a, emb_body_b))

    combined, method_tag = _combine_scores(full_sim, face_sim, body_sim)

    # Confidence degrades when fewer evidence types are available
    n_available = sum(x is not None for x in (full_sim, face_sim, body_sim))
    if n_available == 0:
        confidence = 0.0
        diags.append("no_embedding_evidence_available")
    elif n_available == 1:
        confidence = 0.55
    elif n_available == 2:
        confidence = 0.75
    else:
        confidence = 0.90

    method_label: Any
    if n_available == 0:
        method_label = "fallback_geometry"
    elif face_sim is not None and body_sim is not None:
        method_label = "combined_embedding"
    elif face_sim is not None:
        method_label = "face_embedding"
    elif body_sim is not None:
        method_label = "body_embedding"
    else:
        method_label = "visual_embedding"

    return CharacterPairEvidence(
        first_character_id=char_a.id,
        second_character_id=char_b.id,
        full_similarity=full_sim,
        face_similarity=face_sim,
        body_similarity=body_sim,
        combined_score=combined,
        confidence=confidence,
        same_page=same_page,
        first_page_index=page_a,
        second_page_index=page_b,
        method=method_label,
        evidence={
            "method_tag": method_tag,
            "n_evidence_types": n_available,
            "face_available_a": emb_face_a is not None,
            "face_available_b": emb_face_b is not None,
            "body_available_a": emb_body_a is not None,
            "body_available_b": emb_body_b is not None,
        },
        diagnostics=diags,
    )


# ---------------------------------------------------------------------------
# Identity graph
# ---------------------------------------------------------------------------

class _IdentityGraph:
    """Directed (symmetric) similarity graph over CharacterInstance IDs."""

    def __init__(self) -> None:
        # (id_a, id_b) → evidence, with id_a < id_b (canonical key order)
        self._edges: dict[tuple[str, str], CharacterPairEvidence] = {}
        self._nodes: dict[str, int] = {}  # id → page_index
        self._order: dict[str, tuple[Any, ...]] = {}

    def add_node(
        self,
        character_id: str,
        page_index: int,
        *,
        order: tuple[Any, ...] | None = None,
    ) -> None:
        self._nodes[character_id] = page_index
        self._order[character_id] = order or (page_index, character_id)

    def add_or_update_edge(self, evidence: CharacterPairEvidence) -> None:
        key = tuple(sorted([evidence.first_character_id, evidence.second_character_id]))
        existing = self._edges.get(key)  # type: ignore[arg-type]
        if existing is None or evidence.combined_score > existing.combined_score:
            self._edges[key] = evidence  # type: ignore[index]

    @property
    def nodes(self) -> dict[str, int]:
        return dict(self._nodes)

    @property
    def edges(self) -> list[CharacterPairEvidence]:
        return list(self._edges.values())

    def neighbors(self, node_id: str, *, min_score: float) -> list[tuple[str, float]]:
        result: list[tuple[str, float]] = []
        for (a, b), ev in self._edges.items():
            if ev.combined_score < min_score:
                continue
            if a == node_id:
                result.append((b, ev.combined_score))
            elif b == node_id:
                result.append((a, ev.combined_score))
        return result

    def page_of(self, node_id: str) -> int | None:
        return self._nodes.get(node_id)

    def order_of(self, node_id: str) -> tuple[Any, ...]:
        return self._order.get(node_id, (9999, node_id))


# ---------------------------------------------------------------------------
# Constrained clustering
# ---------------------------------------------------------------------------

class _UnionFind:
    """Union-Find with page-constraint check."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}
        self._page: dict[str, int] = {}

    def add(self, node: str, page: int) -> None:
        if node not in self._parent:
            self._parent[node] = node
            self._rank[node] = 0
            self._page[node] = page

    def find(self, node: str) -> str:
        root = node
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression
        while self._parent[node] != root:
            self._parent[node], node = root, self._parent[node]
        return root

    def pages_in_component(self, node: str) -> set[int]:
        root = self.find(node)
        return {self._page[n] for n in self._parent if self.find(n) == root}

    def can_merge(self, a: str, b: str) -> bool:
        """Return True if merging a and b would not violate same-page uniqueness."""
        if self.find(a) == self.find(b):
            return True  # already same component
        pages_a = self.pages_in_component(a)
        pages_b = self.pages_in_component(b)
        return pages_a.isdisjoint(pages_b)

    def union(self, a: str, b: str) -> bool:
        """Attempt to merge; return True on success, False if constraint violated."""
        if not self.can_merge(a, b):
            return False
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1
        return True

    def components(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for node in self._parent:
            groups[self.find(node)].append(node)
        return dict(groups)


def _cluster_graph(
    graph: _IdentityGraph,
    *,
    match_threshold: float,
    ambiguity_low: float,
    ambiguity_high: float,
) -> tuple[_UnionFind, list[IdentityDiagnostic]]:
    """Build constrained clusters from the identity graph.

    Strategy:
    1. Sort all cross-page edges by combined_score descending.
    2. For each edge above match_threshold: attempt to merge (same-page constraint).
    3. Edges in [ambiguity_low, ambiguity_high]: flag ambiguous, skip merge.
    4. Edges below ambiguity_low: discard.
    """
    uf = _UnionFind()
    diagnostics: list[IdentityDiagnostic] = []

    for node_id, page_idx in graph.nodes.items():
        uf.add(node_id, page_idx)

    cross_page_edges = [
        ev for ev in graph.edges
        if not ev.same_page
    ]
    cross_page_edges.sort(key=lambda e: -e.combined_score)

    for ev in cross_page_edges:
        score = ev.combined_score
        a, b = ev.first_character_id, ev.second_character_id

        if score < ambiguity_low:
            continue  # too weak

        if ambiguity_low <= score < ambiguity_high:
            diagnostics.append(IdentityDiagnostic(
                component="clustering",
                code="ambiguous_edge",
                message=(
                    f"Ambiguous match ({score:.3f}) between {a!r} and {b!r}; "
                    "skipped without forcing."
                ),
                character_ids=[a, b],
            ))
            continue

        # score >= match_threshold (or in upper ambiguity region)
        if not uf.can_merge(a, b):
            diagnostics.append(IdentityDiagnostic(
                component="clustering",
                code="same_page_constraint_violated",
                message=(
                    f"Cannot merge {a!r} and {b!r} (score={score:.3f}): "
                    "would place two instances from the same page in one cluster."
                ),
                character_ids=[a, b],
            ))
            continue

        uf.union(a, b)

    return uf, diagnostics


# ---------------------------------------------------------------------------
# Anonymous label assignment
# ---------------------------------------------------------------------------

def _assign_labels(
    components: dict[str, list[str]],
    graph: _IdentityGraph,
    *,
    ambiguity_low: float,
    ambiguity_high: float,
) -> list[CharacterIdentityCluster]:
    """Build CharacterIdentityCluster list with deterministic anonymous labels.

    Ordering rule:
    - For each component, find the minimum sequence appearance key member.
    - Sort components by that key → ascending.
    - Assign labels Character A, Character B, Character C, …
    """

    def _label(n: int) -> str:
        """0→A, 1→B, …, 25→Z, 26→AA, 27→AB, …"""
        letters = string.ascii_uppercase
        result = ""
        n += 1
        while n > 0:
            n, rem = divmod(n - 1, 26)
            result = letters[rem] + result
        return result

    def _min_key(member_ids: list[str]) -> tuple[int, str]:
        return min(graph.order_of(cid) for cid in member_ids)

    sorted_components = sorted(
        components.values(),
        key=lambda members: _min_key(members),
    )

    clusters: list[CharacterIdentityCluster] = []
    cross_page_ambiguous_ids: set[str] = {
        cid
        for ev in graph.edges
        if not ev.same_page
        and ambiguity_low <= ev.combined_score < ambiguity_high
        for cid in [ev.first_character_id, ev.second_character_id]
    }

    for idx, member_ids in enumerate(sorted_components):
        label = f"Character {_label(idx)}"
        identity_id = f"identity-{idx + 1:03d}"

        members = []
        for cid in sorted(member_ids):
            page_idx = graph.page_of(cid) or 0
            # Recover panel_id from member_ids - not stored in graph directly,
            # so we record it as None here; it can be enriched by the caller.
            members.append(IdentityMember(character_id=cid, page_index=page_idx))

        # State: a singleton with ambiguous partners = "ambiguous"
        # a singleton with no cross-page edges = "unmatched"
        # multi-page cluster = "matched"
        unique_pages = {m.page_index for m in members}
        has_ambiguous = any(cid in cross_page_ambiguous_ids for cid in member_ids)

        if len(unique_pages) > 1:
            state = "matched"
            confidence = 0.80
        elif has_ambiguous:
            state = "ambiguous"
            confidence = 0.50
        else:
            state = "unmatched"
            confidence = 1.0  # certain it's a singleton

        clusters.append(CharacterIdentityCluster(
            identity_id=identity_id,
            label=label,
            members=members,
            confidence=confidence,
            state=state,
        ))
    return clusters


# ---------------------------------------------------------------------------
# Main resolver
# ---------------------------------------------------------------------------

class CharacterIdentityResolver:
    """Resolve cross-page character identity from visual embeddings.

    Injectable:
        provider: CharacterEmbeddingProvider  (default: NullEmbeddingProvider)
        match_threshold, ambiguity_low, ambiguity_high
    """

    def __init__(
        self,
        provider: CharacterEmbeddingProvider | None = None,
        *,
        match_threshold: float = 0.65,
        ambiguity_low: float = 0.45,
        ambiguity_high: float = 0.65,
    ) -> None:
        if not 0.0 <= ambiguity_low <= ambiguity_high <= 1.0:
            raise ValueError("thresholds must satisfy 0 ≤ ambiguity_low ≤ ambiguity_high ≤ 1")
        if not ambiguity_high <= match_threshold <= 1.0:
            # Allow match_threshold == ambiguity_high (contiguous boundary)
            if match_threshold < ambiguity_high:
                raise ValueError("match_threshold must be ≥ ambiguity_high")
        self._provider = provider or NullEmbeddingProvider()
        self.match_threshold = match_threshold
        self.ambiguity_low = ambiguity_low
        self.ambiguity_high = ambiguity_high

    def resolve(
        self,
        sequence_id: str,
        pages: Sequence[PageRepresentation],
        crops_by_character: dict[str, CharacterCropPaths] | None = None,
    ) -> SequenceCharacterIdentity:
        """Build a SequenceCharacterIdentity from three (or more) consecutive pages.

        Args:
            sequence_id:          Sequence identifier string.
            pages:                PageRepresentation list in narrative order.
            crops_by_character:   Map of character_id → CharacterCropPaths.
                                  Pass None or empty dict when crops are unavailable.
        """
        crops = crops_by_character or {}
        cache = _EmbeddingCache(self._provider)
        diagnostics: list[IdentityDiagnostic] = []

        # --- build a flat list of (character, page_index) ---
        all_chars: list[tuple[CharacterInstance, int]] = []
        character_order: dict[str, tuple[Any, ...]] = {}
        for page in pages:
            panel_order = {panel.id: index for index, panel in enumerate(page.panels)}
            for char in page.characters:
                all_chars.append((char, page.page_index))
                character_order[char.id] = (
                    page.page_index,
                    panel_order.get(char.panel_id, len(panel_order)),
                    char.bbox.y1,
                    char.bbox.x1,
                    char.id,
                )

        if not all_chars:
            return SequenceCharacterIdentity(
                sequence_id=sequence_id,
                clusters=[],
                character_to_identity={},
                character_to_label={},
                pair_evidence=[],
                diagnostics=[IdentityDiagnostic(
                    component="resolver",
                    code="no_characters",
                    message="No CharacterInstances found across all pages.",
                )],
            )

        # --- build identity graph ---
        graph = _IdentityGraph()
        for char, page_idx in all_chars:
            graph.add_node(char.id, page_idx, order=character_order[char.id])

        pair_evidence: list[CharacterPairEvidence] = []

        # Compare every cross-page pair
        for i in range(len(all_chars)):
            char_a, page_a = all_chars[i]
            for j in range(i + 1, len(all_chars)):
                char_b, page_b = all_chars[j]
                if page_a == page_b:
                    # Same-page pairs: compute evidence for diagnostics only,
                    # but do NOT feed into clustering.
                    continue
                ev = compute_pair_evidence(
                    char_a, page_a, crops.get(char_a.id),
                    char_b, page_b, crops.get(char_b.id),
                    cache,
                    ambiguity_low=self.ambiguity_low,
                    ambiguity_high=self.ambiguity_high,
                )
                pair_evidence.append(ev)
                graph.add_or_update_edge(ev)

        # --- constrained clustering ---
        uf, cluster_diags = _cluster_graph(
            graph,
            match_threshold=self.match_threshold,
            ambiguity_low=self.ambiguity_low,
            ambiguity_high=self.ambiguity_high,
        )
        diagnostics.extend(cluster_diags)

        # --- build clusters with panel_id enrichment ---
        components = uf.components()
        panel_map: dict[str, str | None] = {
            char.id: char.panel_id
            for char, _ in all_chars
        }
        clusters = _assign_labels(
            components,
            graph,
            ambiguity_low=self.ambiguity_low,
            ambiguity_high=self.ambiguity_high,
        )
        # Enrich panel_id on each member
        for cluster in clusters:
            for member in cluster.members:
                member.panel_id = panel_map.get(member.character_id)

        # --- build flat maps ---
        character_to_identity: dict[str, str | None] = {}
        character_to_label: dict[str, str | None] = {}
        for cluster in clusters:
            for member in cluster.members:
                character_to_identity[member.character_id] = cluster.identity_id
                character_to_label[member.character_id] = cluster.label

        # Characters with no crops or very low evidence get recorded
        for char, _ in all_chars:
            if char.id not in character_to_identity:
                character_to_identity[char.id] = None
                character_to_label[char.id] = None
                diagnostics.append(IdentityDiagnostic(
                    component="resolver",
                    code="character_unresolved",
                    message=f"Character {char.id!r} was not assigned to any cluster.",
                    character_ids=[char.id],
                ))

        return SequenceCharacterIdentity(
            sequence_id=sequence_id,
            clusters=clusters,
            character_to_identity=character_to_identity,
            character_to_label=character_to_label,
            pair_evidence=pair_evidence,
            diagnostics=diagnostics,
        )
