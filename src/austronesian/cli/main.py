"""Command-line interface for the Austronesian research toolkit.

Usage examples::

    # List ABVD languages matching a name fragment
    austronesian abvd search-lang "Amis"

    # Fetch language metadata
    austronesian abvd language 1

    # Compare a meaning across several ABVD languages
    austronesian abvd compare "eye" --ids 1,2,3

    # Search the ACD for a root
    austronesian acd search "mata"
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional


def _abvd_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("abvd", help="Austronesian Basic Vocabulary Database tools")
    sub2 = p.add_subparsers(dest="abvd_cmd", required=True)

    # search-lang
    s = sub2.add_parser("search-lang", help="Search languages by name")
    s.add_argument("query", help="Name fragment to search for")

    # language
    s = sub2.add_parser("language", help="Fetch metadata for one language")
    s.add_argument("id", type=int, help="ABVD language id")

    # compare
    s = sub2.add_parser(
        "compare",
        help="Compare a meaning across multiple languages",
    )
    s.add_argument("meaning", help='English meaning, e.g. "eye"')
    s.add_argument(
        "--ids",
        required=True,
        help="Comma-separated ABVD language ids, e.g. 1,2,3",
    )


def _acd_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("acd", help="Austronesian Comparative Dictionary tools")
    sub2 = p.add_subparsers(dest="acd_cmd", required=True)

    s = sub2.add_parser("search", help="Search ACD for root entries")
    s.add_argument("query", help="Root form or English gloss to search for")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="austronesian",
        description="Austronesian language research toolkit",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _abvd_subparser(sub)
    _acd_subparser(sub)
    return parser


def _run_abvd(args: argparse.Namespace) -> int:
    from austronesian.databases.abvd import ABVDClient

    client = ABVDClient()

    if args.abvd_cmd == "search-lang":
        langs = client.search_languages(args.query)
        if not langs:
            print(f"No languages found matching {args.query!r}.")
            return 1
        for lang in langs:
            print(f"  {lang.id:>5}  {lang}")
        return 0

    if args.abvd_cmd == "language":
        lang = client.get_language(args.id)
        print(json.dumps(lang.__dict__, indent=2, default=str))
        return 0

    if args.abvd_cmd == "compare":
        ids: List[int] = [int(x.strip()) for x in args.ids.split(",")]
        table = client.compare_word(args.meaning, ids)
        print(f"\nMeaning: {args.meaning!r}\n")
        col = max((len(lang) for lang in table), default=8) + 2
        print(f"{'Language':<{col}}  Forms")
        print("─" * (col + 20))
        for lang, forms in table.items():
            print(f"{lang:<{col}}  {', '.join(forms) if forms else '—'}")
        return 0

    return 1


def _run_acd(args: argparse.Namespace) -> int:
    from austronesian.databases.acd import ACDClient
    from austronesian.analysis.roots import format_comparison_table

    client = ACDClient()

    if args.acd_cmd == "search":
        results = client.search(args.query)
        if not results:
            print(f"No ACD entries found for {args.query!r}.")
            return 1
        for cs in results:
            print(format_comparison_table(cs))
            print()
        return 0

    return 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "abvd":
        return _run_abvd(args)
    if args.command == "acd":
        return _run_acd(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
