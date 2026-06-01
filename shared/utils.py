"""Shared utilities."""

import time
import hashlib
import re


def now_ts() -> int:
    return int(time.time())


def short_hash(text: str, length: int = 8) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:length]


def extract_keywords(text: str, max_kw: int = 8) -> list[str]:
    """Extract meaningful keywords without LLM calls.

    Uses simple heuristics: English identifiers, CamelCase,
    snake_case, and common technical terms.
    """
    # Match code identifiers
    identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b', text)

    # Match CamelCase and snake_case components
    atoms = set()
    for ident in identifiers:
        # Split CamelCase
        parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)', ident)
        atoms.update(p.lower() for p in parts if len(p) > 1)
        # Split snake_case
        atoms.update(p.lower() for p in ident.split('_') if len(p) > 1)

    # Common stop words to filter
    stop = {'the', 'and', 'for', 'this', 'that', 'with', 'from',
            'have', 'has', 'had', 'not', 'are', 'was', 'were',
            'will', 'would', 'could', 'should', 'can', 'may',
            'its', 'his', 'her', 'our', 'their', 'all', 'any',
            'but', 'also', 'very', 'just', 'then', 'now', 'here'}

    atoms.difference_update(stop)
    return list(atoms)[:max_kw]


def detect_error_pattern(text: str) -> str | None:
    """Detect common error patterns in model output."""
    text_lower = text.lower()

    patterns = {
        "syntax_error": ["syntaxerror", "syntax error", "unexpected token"],
        "type_error": ["typeerror", "type error", "cannot read property"],
        "import_error": ["importerror", "module not found", "no module named"],
        "auth_error": ["unauthorized", "401", "403", "permission denied"],
        "timeout": ["timeout", "timed out", "connection refused"],
        "null_ref": ["none", "null", "undefined", "cannot read .* of null"],
        "race_condition": ["race condition", "deadlock", "concurrent"],
        "oom": ["out of memory", "memory error", "killed"],
    }

    for category, keywords in patterns.items():
        for kw in keywords:
            if re.search(kw, text_lower):
                return category
    return None
