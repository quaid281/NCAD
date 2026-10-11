"""
scan_ai_writing.py - Automated anti-AI writing pattern scanner based on docs/SKILL.md.

Enforces zero-tolerance for P0 credibility killers, P1 word-list violations,
formulaic transitions, copula avoidance, and hedge stacking across manuscript prose.
"""

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
    "Chatbot artifact": r"\b(?:i\s+hope\s+this\s+helps|great\s+question|as\s+an\s+ai|language\s+model)\b",
    "Vague unsourced attribution": r"\b(?:experts\s+believe|researchers\s+have\s+shown|studies\s+suggest|widely\s+considered\s+to\s+be)\b",
}

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
}

# P2: Stylistic Polish
EM_DASH_PATTERN = r"---|[\u2014]"

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

        # Skip raw table definitions when checking word list if they contain pure numbers/labels
        # but check table captions and notes
        is_tabular_data = bool(re.search(r"^\s*[\w\.-]+\s*&", line))

        def check_category(category_name, patterns_dict):
            for label, pattern in patterns_dict.items():
                for m in re.finditer(pattern, line, re.IGNORECASE):
                    # Exclude bibtex citation keys or commands like \cite{...}
                    matched_span = m.group(0)
                    start, end = m.span()
                    prefix = line[max(0, start-15):start]
                    if "cite" in prefix or "label" in prefix or "ref" in prefix:
                        continue
                    # Ignore pure table data lines for generic words unless it's a headline
                    if is_tabular_data and label in ["robust"]:
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

    return issues, word_count, em_dash_count

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

    for fpath in target_files:
        issues, words, em_dashes = scan_file(fpath)
        total_words += words
        total_em_dashes += em_dashes
        all_issues.extend(issues)

    em_dash_rate = total_em_dashes / (total_words / 1000) if total_words > 0 else 0

    print(f"\n[+] Scanned {len(target_files)} section files.")
    print(f"    Total prose words: {total_words}")
    print(f"    Total em-dashes: {total_em_dashes} ({em_dash_rate:.2f} per 1,000 words; guideline <= 1.0/1,000)")

    p0_issues = [i for i in all_issues if i["category"].startswith("P0")]
    p1_issues = [i for i in all_issues if i["category"].startswith("P1")]

    print(f"\n[+] Total AI Writing Violations Found: {len(all_issues)}")
    print(f"    - P0 Credibility Killers: {len(p0_issues)}")
    print(f"    - P1 Obvious AI Smells / Banned Words: {len(p1_issues)}")

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
