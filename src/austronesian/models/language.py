"""Language model representing a single Austronesian language."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Language:
    """Represents one language entry (e.g. from ABVD or Glottolog).

    Attributes
    ----------
    id:
        Numeric identifier used by the source database (e.g. ABVD language id).
    name:
        Human-readable language name (e.g. "Amis").
    family:
        Top-level family string (e.g. "Formosan", "Philippine", "Malayo-Polynesian").
    subfamily:
        More specific sub-group within the family.
    glottocode:
        Glottolog language code, used to link to https://glottolog.org/.
    iso639_3:
        ISO 639-3 three-letter language code.
    region:
        Broad geographic region (e.g. "Taiwan", "Philippines", "Indonesia").
    latitude:
        Approximate geographic latitude of the language area.
    longitude:
        Approximate geographic longitude of the language area.
    notes:
        Free-text notes or source remarks.
    """

    id: Optional[int] = None
    name: str = ""
    family: str = ""
    subfamily: str = ""
    glottocode: str = ""
    iso639_3: str = ""
    region: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    notes: str = ""

    # ------------------------------------------------------------------ #
    # Convenience helpers                                                  #
    # ------------------------------------------------------------------ #

    def __str__(self) -> str:
        parts = [self.name]
        if self.iso639_3:
            parts.append(f"[{self.iso639_3}]")
        if self.region:
            parts.append(f"({self.region})")
        return " ".join(parts)

    @classmethod
    def from_abvd_dict(cls, data: dict) -> "Language":
        """Create a Language from a raw ABVD JSON language object.

        The ABVD REST API returns objects of the form::

            {
                "id": "1",
                "language": "Malagasy",
                "author": "...",
                "silcode": "mlg",
                "glottocode": "mala1537",
                "notes": "...",
                "typedby": "...",
                "checkedby": "...",
                "latitude": "-18.9",
                "longitude": "47.5",
                "location": "Madagascar"
            }
        """
        try:
            lat = float(data.get("latitude") or 0) or None
        except (TypeError, ValueError):
            lat = None
        try:
            lon = float(data.get("longitude") or 0) or None
        except (TypeError, ValueError):
            lon = None

        return cls(
            id=int(data["id"]) if data.get("id") else None,
            name=data.get("language", ""),
            iso639_3=data.get("silcode", ""),
            glottocode=data.get("glottocode", ""),
            region=data.get("location", ""),
            latitude=lat,
            longitude=lon,
            notes=data.get("notes", ""),
        )
