"""
scan_ai_writing.py - Automated anti-AI writing pattern scanner based on docs/SKILL.md.

Enforces zero-tolerance for P0 credibility killers, P1 word-list violations,
formulaic transitions, copula avoidance, hedge stacking, and stylistic cadence
across manuscript prose.
"""

import math
import os
import re
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
SECTIONS_DIR = WORKSPACE_ROOT / "paper" / "NCAD_CS" / "sections"
SKILL_MD_PATH = WORKSPACE_ROOT / "docs" / "SKILL.md"

# P0: Credibility Killers (Zero tolerance)
P0_PATTERNS = {
    "Cutoff disclaimer": r"\b(?:as\s+of\s+(?:my|the)\s+(?:knowledge\s+cutoff|last\s+update)|knowledge\s+cutoff)\b",
    "Chatbot artifact": r"\b(?:i\s+hope\s+this\s+helps|great\s+question|as\s+an\s+ai|language\s+model|feel\s+free\s+to)\b",
    "Vague unsourced attribution": r"\b(?:experts\s+believe|researchers\s+have\s+shown|studies\s+suggest|widely\s+considered\s+to\s+be|experts\s+agree|it\s+is\s+believed\s+that)\b",
    "Codebase relative path": r"\b(?:reports|scripts|src|mTSBench_data|data|benchmarks|tests)[\/\\][a-zA-Z0-9_\-\.\/\\]+",
    "Raw artifact filename": r"\b[a-zA-Z0-9_\-]+\.(?:csv|tsv|py|sh|json|pt|pth)\b",
    "Significance inflation": r"\b(?:game[- ]changer|groundbreaking|revolutionary|unprecedented|paradigm\s+shift|transformative)\b",
}

# P0: Unanchored comparative assertions (comparatives without numerical evidence on the line)
UNANCHORED_COMPARATIVE_PATTERNS = [
    r"\b(?:significantly|drastically|substantially)\s+(?:outperforms?|improves?|reduces?|increases?|superior|better|higher|lower)\b",
    r"\b(?:vastly\s+superior|marginal\s+degradation)\b",
]

# Allowed statistical/mathematical compounds for "robust"
ROBUST_TECHNICAL_COMPOUND_REGEX = re.compile(
    r"\brobust\s+(?:covariance|estimation|estimators?|statistics|loss(?:es)?|scales?|dispersions?|regressions?|m-estimators?)\b",
    re.IGNORECASE
)

# P1: Obvious AI Smell & Blacklisted Vocabulary (GEMINI.md & docs/SKILL.md)
P1_WORD_LIST = {
    "delve": r"\bdelv(?:e|es|ed|ing)\b",
    "leverage": r"\bleverag(?:e|es|ed|ing)\b",
    "harness": r"\bharness(?:es|ed|ing)?\b",
    "robust": r"\brobust\b",
    "pivotal": r"\bpivotal\b",
    "multifaceted": r"\bmultifaceted\b",
    "testament": r"\btestaments?\s+to\b",
    "beacon": r"\bbeacons?\b",
    "tapestry": r"\btapestr(?:y|ies)\b",
    "foster": r"\bfoster(?:s|ed|ing)?\b",
    "elevate": r"\belevat(?:e|es|ed|ing)\b",
    "underscore": r"\bunderscore(?:s|d|ing)?\b",
    "nuanced": r"\bnuanced?\b",
    "intricate": r"\bintricate\b",
    "seamlessly": r"\bseamless(?:ly)?\b",
    "poised": r"\bpoised(?:\s+to)?\b",
    "holistic": r"\bholistic(?:ally)?\b",
    "paramount": r"\bparamount\b",
    "paradigm": r"\bparadigms?\b",
    "cutting-edge": r"\bcutting[- ]edge\b",
    "game-changer": r"\bgame[- ]chang(?:er|ing)\b",
    "deep dive": r"\bdeep[- ]dives?\b",
    "unpack": r"\bunpack(?:s|ed|ing)?\b",
    "ever-evolving": r"\bever[- ]evolving\b",
    "synergy": r"\bsynerg(?:y|ies)\b",
    "interplay": r"\binterplay\b",
    "at its core": r"\bat\s+its\s+core\b",
    "best practices": r"\bbest[- ]practices\b",
    "realm": r"\brealms?\b",
    "embark": r"\bembarks?\b",
    "meticulous": r"\bmeticulous(?:ly)?\b",
    "plethora": r"\bplethora\b",
    "myriad": r"\bmyriad\b",
    "symphony": r"\bsymphony\b",
    "treasure trove": r"\btreasure\s+trove\b",
    "linchpin": r"\blinchpin\b",
    "cornerstone": r"\bcornerstone\b",
}

# P1: Formulaic Openings & Transitions
P1_TRANSITIONS = {
    "moreover": r"\bmoreover\b",
    "furthermore": r"\bfurthermore\b",
    "additionally": r"\badditionally\b",
    "in conclusion": r"\bin\s+conclusion\b",
    "in summary": r"\bin\s+summary\b",
    "it is worth noting that": r"\bit(?:\'?s|\s+is)\s+worth\s+noting(?:\s+that)?\b",
    "it is important to note that": r"\bit(?:\'?s|\s+is)\s+important\s+to\s+note(?:\s+that)?\b",
    "not only X but also Y": r"\bnot\s+only\b.*?\bbut\s+also\b",
    "rapidly evolving world": r"\bin\s+the\s+rapidly\s+evolving\s+world\s+of\b",
    "at the end of the day": r"\bat\s+the\s+end\s+of\s+the\s+day\b",
    "crucially": r"\bcrucially\b",
    "key findings emerge": r"\bkey\s+findings?\s+emerge\b",
    "let us opener": r"\blet\'?s\b|\blet\s+us\b",
    "performative concession": r"\bwhile\s+promising,?\s+challenges\s+remain\b",
    "call to adventure": r"\b(?:paving\s+the\s+way|opens?\s+up\s+exciting\s+avenues|promising\s+avenues\s+for\s+future|a\s+promising\s+step\s+forward)\b",
    "dual mechanism reflex": r"\bdue\s+to\s+both\b.*?\band\b",
}

# P1: Copula Avoidance
P1_COPULA = {
    "serves as": r"\bserves\s+as\b",
    "boasts": r"\bboasts\b",
    "features (verb)": r"\b(?:model|pipeline|architecture|framework|method|baseline|approach|system)\s+features\b",
}

# P1: Hedge-Stacked Predictions
P1_HEDGES = {
    "could potentially": r"\bcould\s+potentially\b",
    "may eventually": r"\bmay\s+eventually\b",
    "might ultimately": r"\bmight\s+ultimately\b",
    "may potentially": r"\bmay\s+potentially\b",
    "could possibly": r"\bcould\s+possibly\b",
}

# P2: Stylistic Polish
EM_DASH_PATTERN = r"---|[\u2014]"

def extract_clean_sentences(text: str):
    """Segment prose into sentences, handling standard academic abbreviations."""
    # Remove LaTeX comments
    text = re.sub(r"%.*$", "", text, flags=re.MULTILINE)
    # Remove LaTeX display environments (tables, figures, equations, algorithms)
    text = re.sub(r"\\begin\{(?:table|figure|equation|align|gather|tabular|algorithm)\*?\}.*?\\end\{(?:table|figure|equation|align|gather|tabular|algorithm)\*?\}", " ", text, flags=re.DOTALL)
    # Remove math display and inline blocks
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\$.*?\$", " ", text)
    # Remove section/caption/label macros
    text = re.sub(r"\\(?:section|subsection|subsubsection|paragraph|caption|label)\*?\{.*?\}", " ", text)
    # Remove citation and cross-reference macros
    text = re.sub(r"\\[a-zA-Z]+\{.*?\}", " ", text)
    # Remove standalone LaTeX commands
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    # Protect common academic abbreviations
    abbreviations = [
        (r"\be\.g\.", "eg_placeholder"),
        (r"\bi\.e\.", "ie_placeholder"),
        (r"\bet\s+al\.", "etal_placeholder"),
        (r"\bFig\.", "fig_placeholder"),
        (r"\bTab\.", "tab_placeholder"),
        (r"\bEq\.", "eq_placeholder"),
        (r"\bSec\.", "sec_placeholder"),
        (r"\bvs\.", "vs_placeholder"),
        (r"\bref\.", "ref_placeholder"),
        (r"\bno\.", "no_placeholder"),
    ]
    for pattern, placeholder in abbreviations:
        text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE)

    # Split on sentence boundaries
    raw_sentences = re.split(r"[.!?]+(?:\s+|\n+|$)", text)
    sentences = []
    for s in raw_sentences:
        s_clean = s.strip()
        words = re.findall(r"\b[A-Za-z0-9_-]+\b", s_clean)
        # Consider complete sentences between 4 and 100 words
        if 4 <= len(words) <= 120:
            sentences.append((s_clean, len(words)))
    return sentences

def compute_cadence_cv(sentence_lengths):
    """Compute coefficient of variation (sigma / mu) for sentence lengths."""
    if not sentence_lengths or len(sentence_lengths) < 5:
        return None, 0.0, 0.0
    n = len(sentence_lengths)
    mu = sum(sentence_lengths) / n
    variance = sum((x - mu) ** 2 for x in sentence_lengths) / n
    sigma = math.sqrt(variance)
    cv = sigma / mu if mu > 0 else 0.0
    return cv, mu, sigma

def scan_file(file_path: Path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    lines = content.splitlines()
    prose_lines = []
    for line in lines:
        cleaned = re.sub(r"%.*$", "", line)
        prose_lines.append(cleaned)

    prose = "\n".join(prose_lines)
    word_count = len(re.findall(r"\b[A-Za-z0-9_-]+\b", prose))
    em_dash_matches = re.findall(EM_DASH_PATTERN, prose)
    em_dash_count = len(em_dash_matches)

    issues = []

    for line_idx, line in enumerate(prose_lines, start=1):
        if not line.strip():
            continue

        # Skip raw table data rows containing numbers/delimiters
        is_tabular_data = bool(re.search(r"^\s*[\w\.-]+\s*&", line))

        def check_category(category_name, patterns_dict):
            for label, pattern in patterns_dict.items():
                for m in re.finditer(pattern, line, re.IGNORECASE):
                    matched_span = m.group(0)
                    start, end = m.span()
                    prefix = line[max(0, start-15):start]
                    # Exclude citation keys, labels, or cross refs
                    if any(cmd in prefix for cmd in ["cite", "label", "ref", "eqref"]):
                        continue
                    # Carve out technical compound terms for "robust"
                    if label == "robust":
                        context_snippet = line[start:min(len(line), start+35)]
                        if ROBUST_TECHNICAL_COMPOUND_REGEX.search(context_snippet):
                            continue
                        if is_tabular_data:
                            continue
                    issues.append({
                        "file": file_path.name,
                        "line": line_idx,
                        "category": category_name,
                        "label": label,
                        "match": matched_span,
                        "context": line.strip()[:120]
                    })

        check_category("P0 Credibility Killer", P0_PATTERNS)
        check_category("P1 Banned Word", P1_WORD_LIST)
        check_category("P1 Formulaic Transition", P1_TRANSITIONS)
        check_category("P1 Copula Avoidance", P1_COPULA)
        check_category("P1 Hedge Stacking", P1_HEDGES)

        # Check P0 Unanchored Comparative Claims
        for pat in UNANCHORED_COMPARATIVE_PATTERNS:
            for m in re.finditer(pat, line, re.IGNORECASE):
                # Check whether there is an immediate numerical anchor in the line
                has_num = bool(re.search(r"(?:\d+|%|\$p\s*[<>=]|p\s*[<>=]|\\pm|\\times|\bCI\b|\bAUROC\b|\bF1\b)", line))
                if not has_num:
                    issues.append({
                        "file": file_path.name,
                        "line": line_idx,
                        "category": "P0 Credibility Killer",
                        "label": "Unanchored quantitative claim",
                        "match": m.group(0),
                        "context": line.strip()[:120]
                    })

    # Sentence cadence analysis
    sentences = extract_clean_sentences(prose)
    sentence_lengths = [length for _, length in sentences]
    cv, mu, sigma = compute_cadence_cv(sentence_lengths)
    if cv is not None and cv < 0.35 and len(sentence_lengths) >= 8:
        issues.append({
            "file": file_path.name,
            "line": 1,
            "category": "P2 Stylistic Polish",
            "label": "Low sentence length variance",
            "match": f"CV={cv:.2f} (< 0.35)",
            "context": f"Monotonous sentence lengths (mean: {mu:.1f} words, std: {sigma:.1f}, N={len(sentence_lengths)})"
        })

    return issues, word_count, em_dash_count, sentence_lengths

def run_scan():
    print("=" * 70)
    print(" NCAD-CS Manuscript AI Writing Pattern Scanner (docs/SKILL.md)")
    print("=" * 70)

    if not SKILL_MD_PATH.exists():
        print(f"[!] Warning: docs/SKILL.md not found at {SKILL_MD_PATH}")

    target_files = sorted(list(SECTIONS_DIR.glob("*.tex")))
    
    total_words = 0
    total_em_dashes = 0
    all_issues = []
    all_sentence_lengths = []

    for fpath in target_files:
        issues, words, em_dashes, s_lengths = scan_file(fpath)
        total_words += words
        total_em_dashes += em_dashes
        all_issues.extend(issues)
        all_sentence_lengths.extend(s_lengths)

    em_dash_rate = total_em_dashes / (total_words / 1000) if total_words > 0 else 0
    global_cv, global_mu, global_sigma = compute_cadence_cv(all_sentence_lengths)

    print(f"\n[+] Scanned {len(target_files)} section files.")
    print(f"    Total prose words: {total_words}")
    print(f"    Total em-dashes: {total_em_dashes} ({em_dash_rate:.2f} per 1,000 words; guideline <= 1.0/1,000)")
    if global_cv is not None:
        print(f"    Sentence Cadence (Burstiness): CV = {global_cv:.2f} (mean: {global_mu:.1f} words, std: {global_sigma:.1f}, N={len(all_sentence_lengths)})")

    p0_issues = [i for i in all_issues if i["category"].startswith("P0")]
    p1_issues = [i for i in all_issues if i["category"].startswith("P1")]
    p2_issues = [i for i in all_issues if i["category"].startswith("P2")]

    print(f"\n[+] Total AI Writing Violations Found: {len(all_issues)}")
    print(f"    - P0 Credibility Killers: {len(p0_issues)}")
    print(f"    - P1 Obvious AI Smells / Banned Words: {len(p1_issues)}")
    print(f"    - P2 Stylistic Polish Warnings: {len(p2_issues)}")

    if all_issues:
        print("\n" + "-" * 70)
        for issue in all_issues:
            print(f"[{issue['file']}:{issue['line']}] ({issue['category']}) \"{issue['match']}\" [{issue['label']}]")
            print(f"    Context: {issue['context']}")
        print("-" * 70)

    if em_dash_rate > 1.2:
        print(f"\n[!] Warning: Em-dash frequency ({em_dash_rate:.2f}/1k words) exceeds the recommended rate (<= 1.0/1k).")

    if p0_issues or p1_issues:
        print("\n[FAILED] Manuscript prose contains prohibited AI writing patterns.")
        return False
    else:
        print("\n[PASSED] Zero P0/P1 AI writing pattern violations found!")
        return True

if __name__ == "__main__":
    success = run_scan()
    sys.exit(0 if success else 1)
