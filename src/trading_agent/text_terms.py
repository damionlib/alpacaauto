from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache


@lru_cache(maxsize=4096)
def _term_pattern(term: str) -> re.Pattern[str]:
    # Whole-word/phrase match. A plain substring test made "war" match
    # "software", "rate" match "accelerate", "fed" match "federal" and "cut"
    # match "execute", which tripped the macro and negative-news blocks on most
    # ordinary headlines.
    return re.compile(r"(?<![a-z0-9])" + re.escape(term.lower()) + r"(?![a-z0-9])")


def matched_terms(text: str, terms: Iterable[str]) -> list[str]:
    lowered = text.lower()
    return sorted(term for term in set(terms) if _term_pattern(term).search(lowered))


def count_terms(text: str, terms: Iterable[str]) -> int:
    """Number of distinct terms that occur in text as whole words/phrases."""
    return len(matched_terms(text, terms))
