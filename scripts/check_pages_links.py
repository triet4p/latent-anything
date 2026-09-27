"""Fail when a real cross-page link in the merged Pages tree is broken.

Walks every ``*.html`` under the merged deploy tree (default
``.gh-pages-build``, theory at root plus the API guide under ``api/``) and
verifies page-to-page ``href`` targets plus same-page ``#anchors`` against
``id`` attributes. Two pre-existing theory-site patterns are excluded, with
the reason recorded in the report instead of failing the gate:

- notebook ``../research/*.md`` hrefs and sibling ``*.ipynb`` hrefs: raw ``.md``/``.ipynb`` links inside
  ``latent-anything-theory`` notebooks render verbatim through
  ``mkdocs-jupyter`` (``execute: false``); the same links ship on the live
  theory site today and ``mkdocs build --strict`` already passes. Fixing them
  means editing 96 notebooks, which is outside the Task 9 publication scope.
- duplicate-heading ``what-you-see``/``*_N`` and truncated ``experiment-*`` TOC anchors:
  ``mkdocs-jupyter`` renders notebook
  headings with colliding ``id`` values (e.g. two ``what-you-see``), so the
  Material TOC suffixes one copy with ``_1`` but the duplicate ``id`` never
  lands in the page. Both are byte-identical on the live site.
- ``/latent-anything/...`` absolute links on the built-in 404 pages: those
  pages hard-code the repository base path, which only resolves when served
  from Pages, not from a local build directory.

External ``http(s)``/``mailto`` links are skipped: GitHub blob pins and PyPI
URLs are checked by reviewers, not by this gate. Exits nonzero listing every
unexpected broken ``(page, link, reason)`` triple.
"""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path


class _Links(HTMLParser):
    """Collect ``href`` values from anchor tags in one page."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Record the ``href`` of each anchor start tag."""
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)


class _Ids(HTMLParser):
    """Collect ``id`` attributes in one page for anchor checks."""

    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Record the ``id`` of each tagged element."""
        _ = tag
        element_id = dict(attrs).get("id")
        if element_id:
            self.ids.add(element_id)


def is_known_theory_pattern(page_rel: str, link: str) -> str | None:
    """Classify a known pre-existing benign pattern, or ``None``."""
    if link.startswith("/latent-anything/") and page_rel.endswith("404.html"):
        return "404-absolute-base-path"
    if link.endswith(".md") and "/notebooks/" in page_rel:
        return "notebook-raw-md-link"
    if link.endswith(".ipynb") and "/notebooks/" in page_rel:
        return "notebook-raw-ipynb-link"
    fragment = link.partition("#")[2]
    if (
        fragment.startswith("experiment-")
        or fragment.startswith("what-you-see")
        or fragment.endswith("_1")
        or fragment.endswith("_2")
        or fragment.endswith("_3")
        or fragment.endswith("_4")
        or fragment.endswith("_5")
        or fragment.endswith("_6")
    ) and "/notebooks/" in page_rel:
        return "notebook-duplicate-heading-anchor"
    return None


def page_path(root: Path, url_path: str) -> Path | None:
    """Resolve a tree-relative URL to a built page or asset file."""
    candidate = root / url_path.lstrip("/")
    if (candidate / "index.html").is_file():
        return candidate / "index.html"
    if candidate.with_suffix(".html").is_file():
        return candidate.with_suffix(".html")
    if candidate.is_file():
        return candidate
    return None


def check_tree(
    root: Path,
) -> tuple[int, int, list[tuple[str, str, str]], dict[str, int]]:
    """Audit internal links; return ``(checked, anchors, broken, known)``."""
    root_resolved = root.resolve()
    broken: list[tuple[str, str, str]] = []
    known: dict[str, int] = {}
    checked = 0
    anchors = 0
    ids_cache: dict[Path, set[str]] = {}

    def page_ids(page: Path) -> set[str]:
        """Parse and cache the ``id`` set of one built page."""
        if page not in ids_cache:
            parser = _Ids()
            parser.feed(page.read_text(encoding="utf-8", errors="replace"))
            ids_cache[page] = parser.ids
        return ids_cache[page]

    for page in sorted(root.rglob("*.html")):
        rel = page.relative_to(root).as_posix()
        parser = _Links()
        parser.feed(page.read_text(encoding="utf-8", errors="replace"))
        for link in parser.links:
            if link.startswith(("http", "mailto:", "javascript:")):
                continue
            if link.startswith("#"):
                anchors += 1
                if link[1:] not in page_ids(page):
                    pattern = is_known_theory_pattern(rel, link)
                    if pattern is None:
                        broken.append((rel, link, "missing-anchor"))
                    else:
                        known[pattern] = known.get(pattern, 0) + 1
                checked += 1
                continue
            path, _, fragment = link.partition("#")
            target = (page.parent / path).resolve() if path else page
            try:
                target_rel = target.relative_to(root_resolved).as_posix()
            except ValueError:
                pattern = is_known_theory_pattern(rel, link)
                if pattern is None:
                    broken.append((rel, link, "outside-tree"))
                else:
                    known[pattern] = known.get(pattern, 0) + 1
                continue
            resolved = page_path(root, "/" + target_rel)
            if resolved is None:
                pattern = is_known_theory_pattern(rel, link)
                if pattern is None:
                    broken.append((rel, link, "missing-target"))
                else:
                    known[pattern] = known.get(pattern, 0) + 1
                continue
            checked += 1
            if fragment and resolved.suffix == ".html":
                anchors += 1
                if fragment not in page_ids(resolved):
                    pattern = is_known_theory_pattern(rel, link)
                    if pattern is None:
                        broken.append((rel, link, "missing-anchor"))
                    else:
                        known[pattern] = known.get(pattern, 0) + 1
    return checked, anchors, broken, known


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, audit the tree, and report the result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy-dir", default=".gh-pages-build")
    args = parser.parse_args(argv)
    root = Path(args.deploy_dir)
    checked, anchors, broken, known = check_tree(root)
    print(f"checked {checked} internal links ({anchors} anchors) under {root}; unexpected broken: {len(broken)}")
    for name, count in sorted(known.items()):
        print(f"  known-pattern {name}: {count}")
    for page, link, reason in broken:
        print(f"  BROKEN {page}: {link} ({reason})")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
