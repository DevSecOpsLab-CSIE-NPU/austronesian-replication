"""ACD – Austronesian Comparative Dictionary client.

The ACD (https://www.trussel2.com/ACD/) is a static HTML site.  This module
provides a lightweight scraper that parses the *search results* page to extract
Proto-Austronesian root entries and their daughter-language reflexes.

ACD search endpoint (GET)::

    https://www.trussel2.com/ACD/acd-s_search.htm?q=<query>

Each result page lists matching root entries.  For each root the page shows:

* The reconstructed proto-form (e.g. ``*mata``)
* The English gloss
* A table of daughter-language reflexes grouped by sub-proto language

Because the site is static HTML the scraper is intentionally conservative and
falls back gracefully when the HTML structure changes.
"""

from __future__ import annotations

import time
from typing import List, Optional
from urllib.parse import urljoin, urlencode

import requests
from bs4 import BeautifulSoup

from austronesian.models.cognate import CognateSet
from austronesian.models.lexeme import Lexeme

_BASE_URL = "https://www.trussel2.com/ACD/"
_SEARCH_URL = urljoin(_BASE_URL, "acd-s_search.htm")
_DEFAULT_TIMEOUT = 30
_DEFAULT_DELAY = 1.0  # be polite – the ACD is a small academic server


class ACDClient:
    """HTTP scraper for the Austronesian Comparative Dictionary (ACD).

    Parameters
    ----------
    timeout:
        HTTP request timeout in seconds.
    delay:
        Seconds to wait between requests.
    session:
        Optional pre-configured :class:`requests.Session`.
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

    def _get_html(self, url: str, params: Optional[dict] = None) -> BeautifulSoup:
        """Fetch *url* and return a parsed :class:`~bs4.BeautifulSoup` tree."""
        response = self._session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        time.sleep(self.delay)
        return BeautifulSoup(response.text, "lxml")

    @staticmethod
    def _parse_root_entry(section: BeautifulSoup) -> Optional[CognateSet]:
        """Parse a single ACD root-entry HTML block into a :class:`CognateSet`.

        The ACD marks each entry with a ``<dt>`` containing the proto-form and
        gloss, followed by a ``<dd>`` containing reflex tables.  The exact HTML
        varies, so the parser is designed to be tolerant of variations.
        """
        # Try to find the proto-form (usually wrapped in <b> or <i>)
        proto_tag = section.find(["b", "i"])
        proto_form = proto_tag.get_text(strip=True) if proto_tag else ""
        if not proto_form.startswith("*"):
            proto_form = "*" + proto_form if proto_form else ""

        # The gloss / meaning is typically plain text after the proto-form tag
        full_text = section.get_text(" ", strip=True)
        meaning = full_text.replace(proto_form, "").strip().lstrip(".,: ")

        members: List[Lexeme] = []
        # Look for reflex rows: <tr> elements with two cells (language | form)
        for row in section.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if len(cells) >= 2:
                lang_name = cells[0].get_text(strip=True)
                form = cells[1].get_text(strip=True)
                if lang_name and form and lang_name.lower() != "language":
                    members.append(
                        Lexeme(
                            language_name=lang_name,
                            form=form,
                            meaning=meaning,
                        )
                    )

        if not proto_form and not members:
            return None

        return CognateSet(
            proto_form=proto_form,
            meaning=meaning,
            source="ACD",
            members=members,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def search(self, query: str) -> List[CognateSet]:
        """Search the ACD for root entries matching *query*.

        Parameters
        ----------
        query:
            Search string (e.g. ``"mata"``, ``"hand"``, ``"eye"``).

        Returns
        -------
        list of CognateSet
            One entry per matching ACD root entry.  May be empty if no results.
        """
        soup = self._get_html(_SEARCH_URL, params={"q": query})
        results: List[CognateSet] = []

        # The ACD search results page uses <dt>/<dd> pairs or <div class="entry">
        # Try both structural patterns.
        entries = soup.find_all("div", class_="entry")
        if not entries:
            # Fallback: treat every <dt> as a potential root header
            for dt in soup.find_all("dt"):
                cognate = self._parse_root_entry(dt)
                if cognate:
                    results.append(cognate)
        else:
            for entry in entries:
                cognate = self._parse_root_entry(entry)
                if cognate:
                    results.append(cognate)

        return results
