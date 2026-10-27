#!/usr/bin/env python3
"""Audit acronym definitions in the standalone manuscript (IEEE style).

IEEE convention: spell out a term at first use and put the acronym in
parentheses, then use the acronym alone. This script flags acronyms whose
first appearance (in reading order, after the title/abstract rules) is not
a definition of the form ``Full Name (ACR)`` or ``ACR, full name``.

Abstract is checked on its own: IEEE requires it to be self-contained, so
acronyms must be defined there too and again in the body.
"""
from __future__ import annotations

import re
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STANDALONE = ROOT / "paper" / "NCAD_CS" / "NCAD_CS_standalone.tex"

# Universally understood in IEEE venues; no definition required.
COMMON = {
    "IEEE", "TNNLS", "AUC", "ROC", "CPU", "GPU", "PDF", "MSE", "KL", "LP",
    "NASA", "TCN", "MLP", "RMS", "SVD", "PCA", "IID", "EMA", "FFT", "CNN",
    "LSTM", "GRU", "AdamW", "ReLU", "GELU", "SiLU", "DWT", "DB4", "PAC",
}
# Dataset / model proper names, not acronyms to expand.
PROPER = {
    "SMAP", "MSL", "SMD", "GHL", "GECCO", "SWAN", "TimesNet", "TranAD",
    "JEPA", "VICReg", "NCAD", "NCADCS", "OpEntropy", "TS", "CICIDS", "DCdetector",
    "TimeEval", "SF", "PyTorch", "CUDA", "NVIDIA", "RTX", "BFGS",
    # Page headers, address and keyword macros that survive tag stripping.
    "FL", "SUBMISSION", "FOR", "REVIEW", "IEEEkeywords",
}


def strip_tex(src: str) -> str:
    src = re.sub(r"(?<!\\)%.*", "", src)
    src = re.sub(r"\\begin\{(equation|align|aligned|eqnarray)\*?\}.*?\\end\{\1\*?\}", " ", src, flags=re.S)
    src = re.sub(r"\$[^$]*\$", " ", src)
    src = re.sub(r"\\(cite[a-z]*|ref|eqref|label|includegraphics|input|texttt|bibliographystyle|bibliography)\*?(\[[^\]]*\])?\{[^}]*\}", " ", src)
    src = re.sub(r"\\[a-zA-Z]+\*?", " ", src)
    return re.sub(r"[{}\\]", " ", src)


def split_regions(full: str):
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", full, flags=re.S)
    abstract = m.group(1) if m else ""
    body = full.replace(abstract, " ") if abstract else full
    body = body.split("\\begin{document}")[-1]
    return abstract, body


def first_use(text: str):
    seen: "OrderedDict[str, tuple[int, bool, str]]" = OrderedDict()
    for m in re.finditer(r"\b([A-Z][A-Za-z]*[A-Z][A-Za-z]*|[A-Z]{2,})s?\b", text):
        tok = m.group(1)
        # Plural forms (CIs, LSTMs, SDs) count as the singular acronym.
        if re.fullmatch(r"[A-Z]{2,}s", tok):
            tok = tok[:-1]
        if len(re.sub(r"[^A-Z]", "", tok)) < 2:
            continue
        if tok in seen:
            continue
        start = m.start()
        before = text[max(0, start - 140):start]
        after = text[m.end():m.end() + 140]
        # Defined as "Full Name (ACR)" or "ACR (full name)".
        defined = bool(re.search(r"\($", before.rstrip() + "")) and bool(re.search(r"[a-z]\s*\($", before.rstrip()))
        defined = defined or bool(re.match(r"\s*\(([a-z][^)]{6,})\)", after))
        # Caption style: "ER is event recall", "MAD denotes median ...".
        defined = defined or bool(re.match(r"\s+(is|are|denotes|stands for)\s+[a-z]", after))
        ctx = " ".join((before[-60:] + "[[" + tok + "]]" + after[:40]).split())
        seen[tok] = (start, defined, ctx)
    return seen


def run_audit(verbose: bool = True) -> bool:
    """Return True when every acronym is defined at first use."""
    full = STANDALONE.read_text(encoding="utf-8")
    abstract, body = split_regions(full)
    bad = 0
    for name, region in (("ABSTRACT", strip_tex(abstract)), ("BODY", strip_tex(body))):
        if verbose:
            print(f"\n=== {name} ===")
        for tok, (pos, defined, ctx) in first_use(region).items():
            if tok in COMMON or tok in PROPER:
                continue
            if not defined:
                bad += 1
                print(f"  UNDEFINED at first use: {tok:10s} ...{ctx}...")
    print(f"\nTotal acronyms without a definition at first use: {bad}")
    return bad == 0


def main() -> int:
    return 0 if run_audit() else 1


if __name__ == "__main__":
    sys.exit(main())
