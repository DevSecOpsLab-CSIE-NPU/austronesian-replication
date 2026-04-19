"""Lexeme model – a word form in a particular language."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Lexeme:
    """A single word form (lexeme) recorded in a specific language.

    Attributes
    ----------
    id:
        Source-database row identifier.
    language_id:
        Foreign key to the :class:`~austronesian.models.language.Language`.
    language_name:
        Denormalized language name for convenience.
    word_id:
        Numeric identifier of the *meaning* (concept) slot, e.g. ABVD word id 1
        = "hand", word id 2 = "left hand", …
    meaning:
        Gloss / English meaning of the concept slot.
    form:
        The actual phonetic / orthographic form recorded (e.g. "mata").
    phonemic:
        Phonemic transcription, if available.
    notes:
        Source notes on this entry.
    cognate_class:
        Cognate class label assigned by the source database.
    loan:
        ``True`` if this form is flagged as a loanword.
    """

    id: Optional[int] = None
    language_id: Optional[int] = None
    language_name: str = ""
    word_id: Optional[int] = None
    meaning: str = ""
    form: str = ""
    phonemic: str = ""
    notes: str = ""
    cognate_class: str = ""
    loan: bool = False

    # ------------------------------------------------------------------ #
    # Convenience helpers                                                  #
    # ------------------------------------------------------------------ #

    def __str__(self) -> str:
        return f"{self.form!r} ({self.meaning}) [{self.language_name}]"

    @classmethod
    def from_abvd_dict(cls, data: dict, language_id: Optional[int] = None,
                       language_name: str = "") -> "Lexeme":
        """Create a Lexeme from a raw ABVD JSON word object.

        ABVD word objects have the form::

            {
                "id": "12345",
                "word_id": "1",
                "word": "hand",
                "item": "lima",
                "annotation": "",
                "loan": "",
                "cognacy": "1",
                "pmpcognacy": ""
            }
        """
        loan_raw = data.get("loan", "") or ""
        loan = bool(loan_raw.strip())

        return cls(
            id=int(data["id"]) if data.get("id") else None,
            language_id=language_id,
            language_name=language_name,
            word_id=int(data["word_id"]) if data.get("word_id") else None,
            meaning=data.get("word", ""),
            form=data.get("item", ""),
            phonemic=data.get("annotation", ""),
            notes="",
            cognate_class=data.get("cognacy", ""),
            loan=loan,
        )
