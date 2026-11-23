"""manuscript_source.py - Locate and expand the sectioned NCAD-CS manuscript.

The manuscript compiles directly from ``paper/NCAD_CS/NCAD_CS.tex``, which pulls
every part of the paper from ``sections/`` with ``\\input``. Overleaf and the local
verifier both compile that master file, so there is no generated standalone copy.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper" / "NCAD_CS"
SECTIONS_DIR = PAPER_DIR / "sections"
MAIN_TEX = PAPER_DIR / "NCAD_CS.tex"
MAIN_PDF = MAIN_TEX.with_suffix(".pdf")
MAIN_LOG = MAIN_TEX.with_suffix(".log")

_INPUT_RE = re.compile(r"^(?P<lead>[^%\n]*?)\\input\{(?P<path>[^}]+)\}", re.M)


def _resolve(rel: str) -> Path:
    path = PAPER_DIR / rel
    return path if path.suffix == ".tex" else path.with_suffix(".tex")


def expanded_source(recursive: bool = True, main: Path = MAIN_TEX) -> str:
    """Return the master file with ``\\input`` lines inlined.

    With ``recursive=False`` only the master file's own inputs (the section files)
    are inlined, leaving nested table inputs in place.
    """
    text = main.read_text(encoding="utf-8")

    def inline(src: str, depth: int) -> str:
        def repl(m: re.Match) -> str:
            path = _resolve(m.group("path"))
            if not path.exists():
                return m.group(0)
            body = path.read_text(encoding="utf-8")
            if recursive and depth < 4:
                body = inline(body, depth + 1)
            return f"{m.group('lead')}{body}"

        return _INPUT_RE.sub(repl, src)

    return inline(text, 0)


def input_files(main: Path = MAIN_TEX) -> list[Path]:
    """All ``.tex`` files reachable from the master file (master first)."""
    seen: list[Path] = [main]
    queue = [main]
    while queue:
        cur = queue.pop(0)
        for m in _INPUT_RE.finditer(cur.read_text(encoding="utf-8")):
            path = _resolve(m.group("path"))
            if path.exists() and path not in seen:
                seen.append(path)
                queue.append(path)
    return seen
