"""Merge the separately built theory and API MkDocs trees into one Pages tree.

The theory site (``mkdocs.yml``) builds into ``.gh-pages-build`` and the API
guide (``mkdocs-api.yml``) builds into ``.api-pages-build``. MkDocs ``--clean``
(the default) only wipes its own configured ``site_dir``, so neither strict
build erases the other; this script copies the API tree under ``api/`` of the
theory tree *after* both builds succeed. It is the single merge implementation
shared by local verification and the combined deploy workflow — do not
duplicate the copy logic in workflow YAML.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def merge_pages_trees(theory_dir: Path, api_dir: Path, subdir: str = "api") -> tuple[int, int]:
    """Copy the API tree under ``subdir`` of the theory tree.

    Returns ``(deploy_tree_files, api_files)``. Stale ``subdir`` content from a
    previous merge is removed first so deleted guide pages do not linger.
    Raises ``FileNotFoundError``/``OSError`` when either built tree is absent.
    """
    theory_index = theory_dir / "index.html"
    api_index = api_dir / "index.html"
    if not theory_index.is_file():
        raise FileNotFoundError(f"theory tree missing built page: {theory_index}")
    if not api_index.is_file():
        raise FileNotFoundError(f"API tree missing built page: {api_index}")
    dest = theory_dir / subdir
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(api_dir, dest)
    if not (dest / "index.html").is_file():
        raise FileNotFoundError(f"merged tree missing built page: {dest}/index.html")
    deploy_files = sum(1 for _ in theory_dir.rglob("*") if _.is_file())
    api_files = sum(1 for _ in dest.rglob("*") if _.is_file())
    return deploy_files, api_files


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, merge the trees, and report file counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theory-dir", default=".gh-pages-build")
    parser.add_argument("--api-dir", default=".api-pages-build")
    parser.add_argument("--subdir", default="api")
    args = parser.parse_args(argv)
    try:
        deploy_files, api_files = merge_pages_trees(Path(args.theory_dir), Path(args.api_dir), args.subdir)
    except (FileNotFoundError, OSError) as exc:
        print(f"merge_pages_trees: {exc}", file=sys.stderr)
        return 1
    print(
        f"merged {api_files} API files under "
        f"{args.theory_dir}/{args.subdir}; "
        f"deploy tree now holds {deploy_files} files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
