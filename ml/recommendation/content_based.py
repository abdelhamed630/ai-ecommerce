"""Content-based product similarity: TF-IDF + cosine similarity.

Pure ML logic — no FastAPI, no SQLAlchemy. The service layer
(services/recommendation_service.py) loads product rows and hands plain
values to this module, which keeps the algorithm testable in isolation and
reusable by later phases (collaborative filtering, hybrid, personalized).

Pipeline:

    product fields -> build_document() -> TF-IDF (fit once per catalog state)
        -> ContentIndex.similar_to(product_id, limit) -> [(product_id, score)]

Because TfidfVectorizer L2-normalizes each row, cosine similarity between two
products is just the dot product of their rows, so a single target row is
multiplied against the matrix — the full N x N similarity matrix is never
built for a "similar products" request.

Caching: fitting is the expensive step, so the last fitted index is reused
while the catalog text is unchanged. The cache key is a hash of the exact
(product id, document) pairs, so it is correct without any invalidation
hooks: any product add/edit/delete changes the key and triggers a refit.
"""

import hashlib
import threading
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


def normalize_text(text: Optional[str]) -> str:
    """Lowercase, Unicode-normalize and collapse whitespace. None -> ''."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.casefold().split())


def build_document(
    name: Optional[str],
    description: Optional[str],
    category: Optional[str],
    brand: Optional[str],
) -> str:
    """Combined text for one product. Missing/None fields are treated as
    empty. Price and image_url are intentionally excluded: this engine
    measures what a product *is*, not what it costs."""
    parts = (normalize_text(f) for f in (name, description, category, brand))
    return " ".join(part for part in parts if part)


@dataclass(frozen=True)
class ContentIndex:
    """A fitted TF-IDF matrix for one catalog snapshot."""

    product_ids: Tuple[int, ...]
    matrix: sparse.csr_matrix  # (n_products, n_terms), rows L2-normalized

    def similar_to(self, product_id: int, limit: int) -> List[Tuple[int, float]]:
        """Top `limit` (product_id, cosine_score) pairs, best first.

        The target is never included. Ties keep catalog (id) order, so the
        result is deterministic. Scores are floats in [0, 1]; products that
        share no vocabulary with the target score 0.0 and sort last.
        """
        try:
            target_pos = self.product_ids.index(product_id)
        except ValueError:
            return []
        if limit <= 0:
            return []

        scores = (self.matrix @ self.matrix[target_pos].T).toarray().ravel()
        order = np.argsort(-scores, kind="stable")
        result: List[Tuple[int, float]] = []
        for pos in order:
            if pos == target_pos:
                continue
            result.append((self.product_ids[pos], float(min(1.0, max(0.0, scores[pos])))))
            if len(result) == limit:
                break
        return result


    def score_profile(self, interest: Mapping[int, float]) -> Dict[int, float]:
        """Personalized content scores for one user.

        `interest` maps product_id -> positive weight (the user's weighted
        interactions). For every catalog product NOT in `interest`, returns
        sum_i weight_i * cosine(i, candidate). Because rows are L2-normalized
        this equals candidate_row . (sum_i weight_i * row_i), so one sparse
        profile vector is built and multiplied against the matrix: no N x N
        similarity matrix.

        Only products with a strictly positive score are returned; products
        unknown to this index are ignored. Returns {} when nothing matches.
        """
        positions = {pid: pos for pos, pid in enumerate(self.product_ids)}
        rows, weights = [], []
        for pid, weight in interest.items():
            pos = positions.get(pid)
            if pos is not None and weight > 0:
                rows.append(pos)
                weights.append(float(weight))
        if not rows:
            return {}

        weight_vector = sparse.csr_matrix(
            (weights, ([0] * len(rows), rows)), shape=(1, len(self.product_ids))
        )
        profile = weight_vector @ self.matrix  # (1, n_terms)
        scores = (self.matrix @ profile.T).toarray().ravel()

        result: Dict[int, float] = {}
        for pos, score in enumerate(scores):
            pid = self.product_ids[pos]
            if pid in interest or score <= 1e-12:
                continue
            result[pid] = float(score)
        return result


def _fit_matrix(documents: Sequence[str]) -> Optional[sparse.csr_matrix]:
    """Fit TF-IDF; None if there is no usable vocabulary at all."""
    if not any(doc.strip() for doc in documents):
        return None

    def _vectorizer(stop_words):
        return TfidfVectorizer(
            stop_words=stop_words, strip_accents="unicode", sublinear_tf=True
        )

    try:
        return _vectorizer("english").fit_transform(documents).tocsr()
    except ValueError:
        pass
    try:
        # Every token was an English stop word (e.g. a catalog of products
        # named "The One"): retry keeping them rather than returning nothing.
        return _vectorizer(None).fit_transform(documents).tocsr()
    except ValueError:
        return None


_cache_lock = threading.Lock()
_cache: Dict[str, Optional[ContentIndex]] = {}  # holds at most one entry


def _fingerprint(product_ids: Sequence[int], documents: Sequence[str]) -> str:
    digest = hashlib.sha1()
    for pid, doc in zip(product_ids, documents):
        digest.update(str(pid).encode())
        digest.update(b"\x00")
        digest.update(doc.encode("utf-8"))
        digest.update(b"\x01")
    return digest.hexdigest()


def get_or_build_index(
    product_ids: Sequence[int], documents: Sequence[str]
) -> Optional[ContentIndex]:
    """Return the ContentIndex for this exact catalog snapshot, refitting
    only when the (id, document) contents changed since the last call.
    Returns None when the catalog has no usable text."""
    if len(product_ids) != len(documents):
        raise ValueError("product_ids and documents must have the same length")

    key = _fingerprint(product_ids, documents)
    with _cache_lock:
        if key in _cache:
            return _cache[key]

    matrix = _fit_matrix(documents)
    index = None if matrix is None else ContentIndex(tuple(product_ids), matrix)

    with _cache_lock:
        _cache.clear()  # single-entry cache: never grows
        _cache[key] = index
    return index


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()
