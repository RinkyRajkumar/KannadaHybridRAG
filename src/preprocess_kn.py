"""Light Kannada text normalization utilities.

The normalization is intentionally conservative: Unicode NFC plus whitespace
cleanup. This keeps the benchmark close to the source text while still making
tokenization and exact file comparisons predictable.
"""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: object) -> str:
    """Normalize text with Unicode NFC and collapsed whitespace."""

    if text is None:
        return ""
    normalized = unicodedata.normalize("NFC", str(text))
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def tokenize(text: object) -> list[str]:
    """Whitespace tokenizer after light normalization."""

    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split(" ")

