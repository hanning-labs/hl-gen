"""Shared formatting helpers used across pipelines."""


def _format_articles(articles: list[dict]) -> str:
    lines = []
    for i, a in enumerate(articles):
        title = (a.get("title") or "").strip()
        body = (a.get("body") or "").strip()
        lines.append(f"[{i}] {title}")
        if body:
            lines.append(f"    {body}")
    return "\n".join(lines)
