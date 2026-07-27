from __future__ import annotations

from collections.abc import Iterable


def markdown_table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    """Render a compact GitHub-flavored Markdown table."""

    rendered = [[str(value) for value in row] for row in rows]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rendered)
    return "\n".join(lines)
