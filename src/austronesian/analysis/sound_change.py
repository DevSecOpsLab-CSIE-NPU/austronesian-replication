"""Sound-change and phoneme-correspondence analysis utilities.

This module provides tools to:

1. Build a *sound correspondence table* from a collection of cognate sets.
2. Normalise phonemic strings to individual phoneme tokens.
3. Apply simple sound-change rules expressed as ``{source: target}`` dicts.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from austronesian.models.cognate import CognateSet


# ---------------------------------------------------------------------------
# Phoneme tokenisation
# ---------------------------------------------------------------------------

# Multi-character digraphs common in Austronesian transcriptions
_DIGRAPHS = re.compile(
    r"ng|mb|nd|ngg?|ts|tʃ|dʒ|[ʔʼʻ]|"   # IPA / orthographic clusters
    r"[aeiouáéíóúāēīōū]h|"               # vowel + h sequences
    r"[a-zA-ZÀ-ÖØ-öø-ÿ]",              # single characters
    re.UNICODE,
)


def tokenise(form: str) -> List[str]:
    """Split a word form into a list of phoneme tokens.

    Handles common Austronesian digraphs (``ng``, ``mb``, ``nd``, …) and basic
    IPA symbols.

    Parameters
    ----------
    form:
        Orthographic or phonemic string, e.g. ``"mata"`` or ``"ŋaŋa"``.

    Returns
    -------
    list of str
        Ordered list of phoneme tokens.

    Examples
    --------
    >>> tokenise("mata")
    ['m', 'a', 't', 'a']
    >>> tokenise("ngipen")
    ['ng', 'i', 'p', 'e', 'n']
    """
    # Strip diacritics that mark stress, tone, etc. for simpler comparison,
    # but keep length marks and glottal stops.
    cleaned = form.strip()
    tokens = _DIGRAPHS.findall(cleaned)
    return [t for t in tokens if t.strip()]


# ---------------------------------------------------------------------------
# Sound-change rules
# ---------------------------------------------------------------------------

def apply_rules(form: str, rules: Dict[str, str]) -> str:
    """Apply a set of ordered sound-change rules to *form*.

    Rules are applied as simple string replacements in the order they appear
    in *rules*.  More specific (longer) rules should appear before shorter
    ones.

    Parameters
    ----------
    form:
        Input form, e.g. ``"mata"``.
    rules:
        Mapping of ``{source_string: target_string}``, e.g.
        ``{"t": "k", "p": "f"}``.

    Returns
    -------
    str
        The form after all rules have been applied.

    Examples
    --------
    >>> apply_rules("mata", {"t": "d"})
    'mada'
    >>> apply_rules("pitu", {"p": "f", "t": "d"})
    'fidu'
    """
    result = form
    for src, tgt in rules.items():
        result = result.replace(src, tgt)
    return result


# ---------------------------------------------------------------------------
# Correspondence tables
# ---------------------------------------------------------------------------

def build_correspondence_table(
    cognate_sets: List[CognateSet],
    lang_a: str,
    lang_b: str,
) -> Dict[Tuple[str, str], int]:
    """Build a phoneme-correspondence frequency table between two languages.

    For each cognate set that contains entries for both *lang_a* and *lang_b*
    the function aligns the tokenised forms (by position, truncating to the
    shorter length) and tallies how often phoneme *x* in *lang_a* corresponds
    to phoneme *y* in *lang_b*.

    Parameters
    ----------
    cognate_sets:
        List of :class:`~austronesian.models.cognate.CognateSet` objects.
    lang_a:
        Name of the first language (e.g. ``"Amis"``).
    lang_b:
        Name of the second language (e.g. ``"Tagalog"``).

    Returns
    -------
    dict
        ``{(phoneme_a, phoneme_b): count}`` sorted by count descending.

    Examples
    --------
    >>> from austronesian.models.cognate import CognateSet
    >>> from austronesian.models.lexeme import Lexeme
    >>> cs = CognateSet(proto_form="*mata", meaning="eye", members=[
    ...     Lexeme(language_name="Amis", form="mata"),
    ...     Lexeme(language_name="Tagalog", form="mata"),
    ... ])
    >>> table = build_correspondence_table([cs], "Amis", "Tagalog")
    >>> table[("m", "m")]
    1
    """
    counts: Counter = Counter()

    for cs in cognate_sets:
        forms_a = cs.forms_by_language().get(lang_a, [])
        forms_b = cs.forms_by_language().get(lang_b, [])
        if not forms_a or not forms_b:
            continue
        # Use the first attested form for each language
        toks_a = tokenise(forms_a[0])
        toks_b = tokenise(forms_b[0])
        for pa, pb in zip(toks_a, toks_b):
            counts[(pa, pb)] += 1

    return dict(counts.most_common())


def top_correspondences(
    table: Dict[Tuple[str, str], int],
    n: int = 20,
) -> List[Tuple[Tuple[str, str], int]]:
    """Return the *n* most frequent phoneme correspondences from *table*.

    Parameters
    ----------
    table:
        Output of :func:`build_correspondence_table`.
    n:
        Number of top entries to return.

    Returns
    -------
    list of ((phoneme_a, phoneme_b), count)
    """
    sorted_items = sorted(table.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:n]
