"""ABVD – Austronesian Basic Vocabulary Database client.

API reference
-------------
The ABVD exposes a simple REST/JSON endpoint at:

    https://abvd.eva.mpg.de/utils/save/?type=json&data=<resource>&id=<id>

Supported *data* values:

* ``language``  – metadata for a single language record
* ``word``      – all word entries for a language record
* ``languages`` – (index) list of all language records (no ``id`` needed)

Examples::

    # Language metadata
    https://abvd.eva.mpg.de/utils/save/?type=json&data=language&id=1

    # All 210 word slots for language id 1
    https://abvd.eva.mpg.de/utils/save/?type=json&data=word&id=1

    # Full language index
    https://abvd.eva.mpg.de/utils/save/?type=json&data=languages
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import requests

from austronesian.models.language import Language
from austronesian.models.lexeme import Lexeme

_BASE_URL = "https://abvd.eva.mpg.de/utils/save/"
_DEFAULT_TIMEOUT = 30  # seconds
_DEFAULT_DELAY = 0.5   # polite crawl delay between requests (seconds)


class ABVDClient:
    """HTTP client for the Austronesian Basic Vocabulary Database (ABVD).

    Parameters
    ----------
    timeout:
        HTTP request timeout in seconds.
    delay:
        Seconds to wait between successive API calls (be a polite crawler).
    session:
        Optional pre-configured :class:`requests.Session` (useful for testing).
    """

    def __init__(
        self,
        timeout: int = _DEFAULT_TIMEOUT,
        delay: float = _DEFAULT_DELAY,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.timeout = timeout
        self.delay = delay
        self._session = session or requests.Session()
        self._session.headers.update(
            {"User-Agent": "AustronesianResearchToolkit/0.1 (academic research)"}
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get(self, params: dict) -> dict:
        """Perform a GET request and return the parsed JSON payload."""
        params = {**params, "type": "json"}
        response = self._session.get(
            _BASE_URL, params=params, timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_language(self, language_id: int) -> Language:
        """Fetch metadata for a single language by its ABVD *id*.

        Parameters
        ----------
        language_id:
            ABVD integer identifier (e.g. 1 for Malagasy).

        Returns
        -------
        Language
            Populated :class:`~austronesian.models.language.Language` instance.
        """
        data = self._get({"data": "language", "id": language_id})
        # ABVD wraps the result in {"data": [...], ...}
        records = data.get("data", [])
        if not records:
            raise ValueError(f"No language found for id={language_id}")
        time.sleep(self.delay)
        return Language.from_abvd_dict(records[0])

    def get_words(self, language_id: int) -> List[Lexeme]:
        """Fetch all word entries (up to 210 slots) for a language.

        Parameters
        ----------
        language_id:
            ABVD integer identifier.

        Returns
        -------
        list of Lexeme
        """
        data = self._get({"data": "word", "id": language_id})
        records = data.get("data", [])
        time.sleep(self.delay)

        # Retrieve the language name from the nested language object if present
        language_name = ""
        if records and "language" in records[0]:
            language_name = records[0]["language"].get("language", "")

        return [
            Lexeme.from_abvd_dict(row, language_id=language_id,
                                  language_name=language_name)
            for row in records
        ]

    def list_languages(self) -> List[Language]:
        """Return metadata for *all* languages in the ABVD.

        Returns
        -------
        list of Language
            Sorted by language id.
        """
        data = self._get({"data": "languages"})
        records = data.get("data", [])
        time.sleep(self.delay)
        return [Language.from_abvd_dict(r) for r in records]

    def search_languages(self, query: str) -> List[Language]:
        """Return languages whose name contains *query* (case-insensitive).

        Parameters
        ----------
        query:
            Substring to search for, e.g. ``"Amis"``.

        Returns
        -------
        list of Language
        """
        all_langs = self.list_languages()
        q = query.lower()
        return [lang for lang in all_langs if q in lang.name.lower()]

    def compare_word(
        self,
        meaning: str,
        language_ids: List[int],
    ) -> Dict[str, List[str]]:
        """Compare a single meaning across multiple languages.

        Parameters
        ----------
        meaning:
            English gloss to look up (e.g. ``"hand"``).
        language_ids:
            ABVD language ids to compare.

        Returns
        -------
        dict
            ``{language_name: [form, ...]}`` mapping (forms may be empty).
        """
        result: Dict[str, List[str]] = {}
        m_lower = meaning.lower().strip()
        for lang_id in language_ids:
            words = self.get_words(lang_id)
            matched = [
                w.form
                for w in words
                if w.meaning.lower().strip() == m_lower and w.form
            ]
            lang_name = words[0].language_name if words else str(lang_id)
            result[lang_name] = matched
        return result
