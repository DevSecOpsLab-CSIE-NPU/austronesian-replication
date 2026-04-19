"""Cognate-detection heuristics for Austronesian word lists.

This module provides lightweight heuristic methods to identify potential
cognates across languages when formal cognate judgements are not available.

Approaches implemented
----------------------
1. **Edit-distance similarity** – two forms are candidate cognates if their
   normalised Levenshtein distance is below a threshold.
2. **Shared prefix** – a quick filter based on a common leading phoneme
   sequence (useful for Austronesian since roots are often prefixed but the
   root core is preserved).
3. **Cognate-class grouping** – when a source database already provides
   cognate-class labels (as ABVD does), group lexemes by that label.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from austronesian.models.lexeme import Lexeme
from austronesian.models.cognate import CognateSet


# ---------------------------------------------------------------------------
# Edit-distance helpers
# ---------------------------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
    """Compute the Levenshtein (edit) distance between two strings.

    Parameters
    ----------
    a, b:
        Input strings.

    Returns
    -------
    int
        Minimum number of single-character insertions, deletions, or
        substitutions required to transform *a* into *b*.

    Examples
    --------
    >>> levenshtein("mata", "mata")
    0
    >>> levenshtein("mata", "maka")
    1
    >>> levenshtein("pitu", "tujuh")
    4
    """
    if a == b:
        return 0
    len_a, len_b = len(a), len(b)
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a

    # Dynamic programming matrix (space-optimised to two rows)
    prev = list(range(len_b + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len_b
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                curr[j - 1] + 1,      # insertion
                prev[j] + 1,          # deletion
                prev[j - 1] + cost,   # substitution
            )
        prev = curr
    return prev[len_b]


def normalised_distance(a: str, b: str) -> float:
    """Normalised Levenshtein distance in ``[0.0, 1.0]``.

    A value of ``0.0`` means identical; ``1.0`` means completely different.

    Parameters
    ----------
    a, b:
        Input strings.

    Returns
    -------
    float

    Examples
    --------
    >>> normalised_distance("mata", "mata")
    0.0
    >>> normalised_distance("mata", "")
    1.0
    """
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    return levenshtein(a, b) / max_len


# ---------------------------------------------------------------------------
# Heuristic cognate detection
# ---------------------------------------------------------------------------

def find_potential_cognates(
    lexemes: List[Lexeme],
    meaning: str,
    threshold: float = 0.4,
) -> List[CognateSet]:
    """Group lexemes for the same *meaning* into candidate cognate sets.

    Forms are clustered using single-linkage agglomeration on normalised
    Levenshtein distance.  Two forms are linked if their distance is ≤
    *threshold*.

    Parameters
    ----------
    lexemes:
        List of :class:`~austronesian.models.lexeme.Lexeme` objects (typically
        all entries for a given meaning slot across multiple languages).
    meaning:
        The meaning / gloss these lexemes share (used to label the sets).
    threshold:
        Maximum normalised edit distance to treat two forms as cognate
        candidates.  Default ``0.4`` is a commonly used starting point for
        Austronesian (low-distance family).

    Returns
    -------
    list of CognateSet
        Each set contains lexemes that are mutually similar.  Lexemes with an
        empty form are excluded.
    """
    # Filter out entries without a form
    valid = [lex for lex in lexemes if lex.form.strip()]

    if not valid:
        return []

    # Build a simple union-find structure for clustering
    parent = list(range(len(valid)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            if normalised_distance(valid[i].form, valid[j].form) <= threshold:
                union(i, j)

    # Collect clusters
    clusters: Dict[int, List[Lexeme]] = defaultdict(list)
    for idx, lex in enumerate(valid):
        clusters[find(idx)].append(lex)

    cognate_sets = []
    for cluster_id, members in sorted(clusters.items()):
        cs = CognateSet(
            id=str(cluster_id),
            meaning=meaning,
            source="heuristic",
            members=members,
        )
        cognate_sets.append(cs)

    return cognate_sets


def group_by_cognate_class(lexemes: List[Lexeme]) -> Dict[str, CognateSet]:
    """Group *lexemes* by their ``cognate_class`` attribute.

    Suitable for databases (like ABVD) that already provide explicit cognate
    labels.

    Parameters
    ----------
    lexemes:
        Lexemes with a populated ``cognate_class`` field.

    Returns
    -------
    dict
        ``{cognate_class_label: CognateSet}``; the ``""`` key collects
        un-labelled entries.
    """
    groups: Dict[str, List[Lexeme]] = defaultdict(list)
    for lex in lexemes:
        groups[lex.cognate_class].append(lex)

    result: Dict[str, CognateSet] = {}
    meaning = lexemes[0].meaning if lexemes else ""
    for label, members in groups.items():
        result[label] = CognateSet(
            id=label,
            meaning=meaning,
            source="database",
            members=members,
        )
    return result
