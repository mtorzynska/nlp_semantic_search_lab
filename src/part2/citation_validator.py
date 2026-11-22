# Programmatic citation validation for RAG answers.
# Checks that every [n] citation in an answer refers to a document that was
# actually retrieved (structural validity). Does not verify claim truthfulness
# — for that you need an NLI model or a second LLM pass.

import re
from dataclasses import dataclass, field
from typing import Iterable

from langchain_core.documents import Document

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass
class CitationCheck:
    citation_numbers: list[int] = field(default_factory=list)
    valid_numbers: list[int] = field(default_factory=list)
    invalid_numbers: list[int] = field(default_factory=list)
    out_of_range: list[int] = field(default_factory=list)
    coverage: float = 0.0  # fraction of cited numbers that point to real docs


def extract_citations(answer: str) -> list[int]:
    """Return every citation number mentioned in the answer (1-indexed)."""
    numbers: list[int] = []
    for match in _CITATION_RE.finditer(answer):
        for part in match.group(1).split(","):
            part = part.strip()
            if part.isdigit():
                numbers.append(int(part))
    return numbers


def validate_citations(answer: str, docs: Iterable[Document]) -> CitationCheck:
    """Structural check: are all cited numbers valid indices into docs?"""
    docs_list = list(docs)
    n_docs = len(docs_list)
    numbers = extract_citations(answer)

    check = CitationCheck(citation_numbers=numbers)

    if n_docs == 0:
        check.invalid_numbers = numbers
        return check

    for n in numbers:
        if 1 <= n <= n_docs:
            check.valid_numbers.append(n)
        else:
            check.out_of_range.append(n)
            check.invalid_numbers.append(n)

    check.coverage = len(check.valid_numbers) / len(numbers) if numbers else 1.0
    return check
