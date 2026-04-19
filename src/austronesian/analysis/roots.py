"""Root-reconstruction helpers for Proto-Austronesian research.

This module provides utilities for working with reconstructed roots:

* Parsing and normalising proto-form strings (e.g. ``"*mata"``)
* Generating simple proto-form candidates from a set of cognate forms using a
  majority-vote per position strategy
* Formatting comparative tables for display
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional

from austronesian.analysis.sound_change import tokenise
from austronesian.models.cognate import CognateSet


_PROTO_PREFIX = re.compile(r"^\*+")  # strip leading asterisks


def normalise_proto(form: str) -> str:
    """Strip leading asterisk(s) from a proto-form and lower-case it.

    Parameters
    ----------
    form:
        Raw proto-form string, e.g. ``"*mata"`` or ``"**ma-qetil"``.

    Returns
    -------
    str
        Clean form, e.g. ``"mata"``.

    Examples
    --------
    >>> normalise_proto("*mata")
    'mata'
    >>> normalise_proto("**ma-qetil")
    'ma-qetil'
    """
    return _PROTO_PREFIX.sub("", form).strip().lower()


def reconstruct_proto(cognate_set: CognateSet) -> str:
    """Produce a naive proto-form candidate from a :class:`CognateSet`.

    This uses a simple **majority-vote per phoneme position** approach:

    1. Tokenise each member form.
    2. At each position, pick the most common phoneme across all forms.
    3. Concatenate the winners and prefix with ``*``.

    This is *not* a rigorous comparative-method reconstruction.  It is a
    quick heuristic to suggest a candidate root for further investigation.

    Parameters
    ----------
    cognate_set:
        A cognate set whose ``members`` have non-empty ``form`` attributes.

    Returns
    -------
    str
        Candidate proto-form prefixed with ``*``, e.g. ``"*mata"``.
        Returns ``""`` if no valid forms are present.

    Examples
    --------
    >>> from austronesian.models.cognate import CognateSet
    >>> from austronesian.models.lexeme import Lexeme
    >>> cs = CognateSet(members=[
    ...     Lexeme(form="mata"), Lexeme(form="mata"), Lexeme(form="maca"),
    ... ])
    >>> reconstruct_proto(cs)
    '*mata'
    """
    token_lists = [
        tokenise(m.form) for m in cognate_set.members if m.form.strip()
    ]
    if not token_lists:
        return ""

    # Use the median length as target length to avoid outlier influence
    lengths = sorted(len(t) for t in token_lists)
    target_len = lengths[len(lengths) // 2]

    reconstructed: List[str] = []
    for pos in range(target_len):
        phonemes_at_pos = [
            tl[pos] for tl in token_lists if pos < len(tl)
        ]
        if not phonemes_at_pos:
            break
        winner = Counter(phonemes_at_pos).most_common(1)[0][0]
        reconstructed.append(winner)

    return "*" + "".join(reconstructed)


def format_comparison_table(
    cognate_set: CognateSet,
    languages: Optional[List[str]] = None,
    include_proto: bool = True,
) -> str:
    """Format a cognate set as a plain-text comparison table.

    Parameters
    ----------
    cognate_set:
        The cognate set to display.
    languages:
        Ordered list of language names to include.  If ``None``, all
        languages in the set are shown (sorted alphabetically).
    include_proto:
        If ``True`` (default), include the proto-form row at the top.

    Returns
    -------
    str
        A formatted table string, e.g.::

            Meaning : eye
            Proto   : *mata
            ─────────────────────────
            Language       Form
            ─────────────────────────
            Amis           mata
            Tagalog        mata
            Malay          mata
            ─────────────────────────

    Examples
    --------
    >>> from austronesian.models.cognate import CognateSet
    >>> from austronesian.models.lexeme import Lexeme
    >>> cs = CognateSet(proto_form="*mata", meaning="eye", members=[
    ...     Lexeme(language_name="Amis", form="mata"),
    ...     Lexeme(language_name="Tagalog", form="mata"),
    ... ])
    >>> print(format_comparison_table(cs))  # doctest: +NORMALIZE_WHITESPACE
    Meaning : eye
    Proto   : *mata
    ...
    """
    forms_by_lang = cognate_set.forms_by_language()
    if languages is None:
        languages = sorted(forms_by_lang.keys())

    col_width = max((len(lang) for lang in languages), default=8) + 2
    separator = "─" * (col_width + 20)
    header = f"{'Language':<{col_width}} Form"
    lines = [
        f"Meaning : {cognate_set.meaning}",
    ]
    if include_proto and cognate_set.proto_form:
        lines.append(f"Proto   : {cognate_set.proto_form}")
    lines += [separator, header, separator]
    for lang in languages:
        forms = forms_by_lang.get(lang, [])
        form_str = ", ".join(forms) if forms else "—"
        lines.append(f"{lang:<{col_width}} {form_str}")
    lines.append(separator)
    return "\n".join(lines)
