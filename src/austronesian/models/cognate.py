"""CognateSet model – a group of lexemes that share a common origin."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from austronesian.models.lexeme import Lexeme


@dataclass
class CognateSet:
    """A set of lexemes judged to be cognate (derived from a common ancestor).

    Attributes
    ----------
    id:
        Identifier for this cognate set (e.g. the ACD root entry id).
    proto_form:
        Reconstructed Proto-Austronesian (or sub-proto) form, prefixed with
        ``*`` by convention (e.g. ``"*mata"``).
    meaning:
        Gloss / English meaning shared by the set.
    source:
        Name of the database or publication this set is drawn from
        (e.g. ``"ABVD"``, ``"ACD"``).
    members:
        Individual :class:`~austronesian.models.lexeme.Lexeme` objects belonging
        to this cognate set.
    notes:
        Free-text notes.
    """

    id: Optional[str] = None
    proto_form: str = ""
    meaning: str = ""
    source: str = ""
    members: List[Lexeme] = field(default_factory=list)
    notes: str = ""

    # ------------------------------------------------------------------ #
    # Convenience helpers                                                  #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.members)

    def __str__(self) -> str:
        proto = self.proto_form or "?"
        return (
            f"CognateSet({proto!r}, meaning={self.meaning!r}, "
            f"members={len(self.members)})"
        )

    def language_names(self) -> List[str]:
        """Return a sorted list of unique language names represented."""
        return sorted({m.language_name for m in self.members if m.language_name})

    def forms_by_language(self) -> dict:
        """Return a ``{language_name: [form, ...]}`` mapping."""
        result: dict = {}
        for member in self.members:
            result.setdefault(member.language_name, []).append(member.form)
        return result
