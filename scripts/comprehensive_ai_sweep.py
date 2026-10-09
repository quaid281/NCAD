#!/usr/bin/env python3
"""Comprehensive sweep using docs/SKILL.md and docs/SKILL.full.md rules."""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECTIONS_DIR = ROOT / "paper" / "NCAD_CS" / "sections"

# Parse tables from docs/SKILL.full.md
with open(ROOT / "docs" / "SKILL.full.md", "r", encoding="utf-8", errors="ignore") as f:
    skill_text = f.read()

# Pattern dictionaries
PATTERNS = {
    "P0: Chatbot / Meta Artifacts": {
        "as an ai": r"\bas an ai\b",
        "i hope this helps": r"\bi hope this helps\b",
        "great question": r"\bgreat question\b",
        "certainly / sure thing": r"\b(?:certainly|sure thing)!?\b",
        "let's dive/explore/unpack": r"\blet'?s\s+(?:dive|explore|examine|unpack|look at|delve)\b",
        "here is a breakdown": r"\bhere(?:'s| is)\s+a\s+breakdown\b",
        "as of my last update": r"\bas of my (?:last )?update\b",
        "in a world where": r"\bin a world where\b",
        "now more than ever": r"\bnow more than ever\b",
        "line i keep coming back to": r"\bthe line i keep coming back to\b",
        "can't stop thinking about": r"\bcan'?t stop thinking about\b",
        "thank me later": r"\bthank me later\b",
        "worth your time": r"\bworth your time\b",
    },
    "P1: Tier 1A Word List": {
        "delve": r"\bdelv(?:e|es|ed|ing)\b",
        "tapestry": r"\btapestr(?:y|ies)\b",
        "realm": r"\brealms?\b",
        "paradigm (metaphor)": r"\bparadigms?\b",
        "embark": r"\bembarks?\b",
        "beacon": r"\bbeacons?\b",
        "testament to": r"\btestaments?\s+to\b",
        "cutting-edge": r"\bcutting[- ]edge\b",
        "leverage (verb)": r"\bleverag(?:e|es|ed|ing)\b",
        "pivotal": r"\bpivotal\b",
        "underscores (verb)": r"\bunderscores?\b",
        "meticulous": r"\bmeticulous(?:ly)?\b",
        "seamless": r"\bseamless(?:ly)?\b",
        "game-changer": r"\bgame[- ]chang(?:er|ing)\b",
        "deep dive": r"\bdeep[- ]dives?\b",
        "unpack": r"\bunpack(?:s|ed|ing)?\b",
        "intricate": r"\bintricate(?:ly)?\b",
        "intricacies": r"\bintricacies\b",
        "ever-evolving": r"\bever[- ]evolving\b",
        "daunting": r"\bdaunting\b",
        "holistic": r"\bholistic(?:ally)?\b",
        "actionable": r"\bactionable\b",
        "impactful": r"\bimpactful\b",
        "learnings": r"\blearnings\b",
        "best practices": r"\bbest[- ]practices\b",
        "at its core": r"\bat\s+its\s+core\b",
        "synergy": r"\bsynerg(?:y|ies)\b",
        "interplay": r"\binterplay\b",
        "embrace": r"\bembrac(?:e|es|ed|ing)\b",
        "load-bearing": r"\bload[- ]bearing\b",
        "comprehensive": r"\bcomprehensive\b",
    },
    "P1: Tier 1B Wordiness / Inflated Formality": {
        "utilize": r"\butiliz(?:e|es|ed|ing)\b",
        "in order to": r"\bin\s+order\s+to\b",
        "due to the fact that": r"\bdue\s+to\s+the\s+fact\s+that\b",
        "serves as": r"\bserves\s+as\b",
        "features (verb)": r"\bfeatures\b",
        "boasts": r"\bboasts\b",
        "commence": r"\bcommenc(?:e|es|ed|ing)\b",
        "ascertain": r"\bascertain\b",
        "endeavor": r"\bendeavor\b",
    },
    "P1: Formulaic / Aphorism / False Contrast": {
        "not only X but also Y": r"\bnot\s+only\b.*?\bbut\s+(?:also\b)?",
        "isn't just X, it's Y": r"\bisn'?t\s+just\b.*?\bit'?s\b",
        "less about X and more about Y": r"\bless\s+about\b.*?\band\s+more\s+about\b",
        "double-edged sword": r"\bdouble[- ]edged\s+sword\b",
        "push the envelope": r"\bpush(?:ing)?\s+the\s+envelope\b",
        "at the end of the day": r"\bat\s+the\s+end\s+of\s+the\s+day\b",
        "when it comes to": r"\bwhen\s+it\s+comes\s+to\b",
        "it's important to remember": r"\bit'?s\s+important\s+to\s+remember\b",
        "only time will tell": r"\bonly\s+time\s+will\s+tell\b",
        "a force to be reckoned with": r"\ba\s+force\s+to\s+be\s+reckoned\s+with\b",
        "fast-paced world": r"\bfast[- ]paced\s+world\b",
    },
    "P1: Hollow Hedges / Significance Inflation": {
        "could potentially": r"\bcould\s+potentially\b",
        "may potentially": r"\bmay\s+potentially\b",
        "might potentially": r"\bmight\s+potentially\b",
        "it is worth noting that": r"\bit(?:\'?s|\s+is)\s+worth\s+noting\s+that\b",
        "it is important to note": r"\bit(?:\'?s|\s+is)\s+important\s+to\s+note\b",
        "to be clear / let's be clear": r"\b(?:to|let\'?s)\s+be\s+clear\b",
        "quite frankly / to be honest": r"\b(?:quite\s+frankly|to\s+be\s+honest)\b",
        "historic milestone": r"\bhistoric\s+milestone\b",
        "watershed moment": r"\bwatershed\s+moment\b",
        "groundbreaking": r"\bgroundbreaking\b",
        "breathtaking / astounding": r"\b(?:breathtaking|astounding)\b",
    },
    "P2: Tier 2 Words (High Density Alert)": {
        "harness": r"\bharness(?:es|ed|ing)?\b",
        "navigate (metaphor)": r"\bnavigat(?:e|es|ed|ing)\b",
        "foster": r"\bfoster(?:s|ed|ing)?\b",
        "elevate": r"\belevat(?:e|es|ed|ing)\b",
        "unleash": r"\bunleash(?:es|ed|ing)?\b",
        "streamline": r"\bstreamlin(?:e|es|ed|ing)\b",
        "empower": r"\bempower(?:s|ed|ing)?\b",
        "bolster": r"\bbolster(?:s|ed|ing)?\b",
        "spearhead": r"\bspearhead(?:s|ed|ing)?\b",
        "resonate": r"\bresonat(?:e|es|ed|ing)\b",
        "revolutionize": r"\brevolutioniz(?:e|es|ed|ing)\b",
        "facilitate": r"\bfacilitat(?:e|es|ed|ing)\b",
        "underpin": r"\bunderpin(?:s|ned|ning)?\b",
        "nuanced": r"\bnuanced?\b",
        "crucial": r"\bcrucial\b",
        "multifaceted": r"\bmultifaceted\b",
        "ecosystem (metaphor)": r"\becosystems?\b",
        "myriad": r"\bmyriad\b",
        "plethora": r"\bplethora\b",
        "encompass": r"\bencompass(?:es|ed|ing)?\b",
        "catalyze": r"\bcatalyz(?:e|es|ed|ing)\b",
        "reimagine": r"\breimagin(?:e|es|ed|ing)\b",
        "galvanize": r"\bgalvaniz(?:e|es|ed|ing)\b",
        "augment": r"\baugment(?:s|ed|ing)?\b",
        "cultivate": r"\bcultivat(?:e|es|ed|ing)\b",
        "illuminate": r"\billuminat(?:e|es|ed|ing)\b",
        "elucidate": r"\belucidat(?:e|es|ed|ing)\b",
        "juxtapose": r"\bjuxtapos(?:e|es|ed|ing)\b",
        "transformative": r"\btransformative\b",
        "cornerstone": r"\bcornerstone\b",
        "paramount": r"\bparamount\b",
        "poised to": r"\bpoised\s+to\b",
        "burgeoning": r"\bburgeoning\b",
        "nascent": r"\bnascent\b",
        "quintessential": r"\bquintessential\b",
        "overarching": r"\boverarching\b",
        "vital": r"\bvital\b",
        "essential": r"\bessential\b",
        "remarkable": r"\bremarkabl(?:e|y)\b",
    },
    "P2: Overused Transitions": {
        "moreover": r"\bmoreover\b",
        "furthermore": r"\bfurthermore\b",
        "additionally": r"\badditionally\b",
        "that being said": r"\bthat\s+being\s+said\b",
        "with that in mind": r"\bwith\s+that\s+in\s+mind\b",
        "in summary / to wrap up": r"\b(?:in\s+summary|to\s+wrap\s+things?\s+up)\b",
    },
    "P2: Copula Avoidance": {
        "acts as a": r"\bacts\s+as\s+(?:a|an)\b",
        "functions as a": r"\bfunctions\s+as\s+(?:a|an)\b",
        "stands as a": r"\bstands\s+as\s+(?:a|an)\b",
        "serves to": r"\bserves\s+to\b",
        "marks a": r"\bmarks\s+a\b",
    },
}

files = [f for f in sorted(os.listdir(SECTIONS_DIR)) if f.endswith(".tex") and not f.startswith("tab_")]

total_words = 0
total_em_dashes = 0
findings = []

for fname in files:
    fpath = SECTIONS_DIR / fname
    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    lines = content.split("\n")
    prose_lines = []
    for l in lines:
        cleaned = re.sub(r"%.*$", "", l)
        prose_lines.append(cleaned)
    prose = "\n".join(prose_lines)

    words = len(re.findall(r"\b\w+\b", prose))
    total_words += words
    dashes = len(re.findall(r"---", prose)) + len(re.findall(r"—", prose))
    total_em_dashes += dashes

    for line_no, line in enumerate(prose_lines, start=1):
        if not line.strip():
            continue
        # Skip pure LaTeX equation blocks
        if line.strip().startswith("\\begin{equation}") or line.strip().startswith("\\end{equation}"):
            continue
        
        for cat_name, pat_dict in PATTERNS.items():
            for label, pat in pat_dict.items():
                for m in re.finditer(pat, line, re.IGNORECASE):
                    findings.append({
                        "file": fname,
                        "line": line_no,
                        "category": cat_name,
                        "label": label,
                        "match": m.group(0),
                        "snippet": line.strip()[:140]
                    })

print(f"=== COMPREHENSIVE SKILL.MD AUDIT REPORT ===")
print(f"Total prose words: {total_words}")
print(f"Total em-dashes: {total_em_dashes} (rate: {total_em_dashes / (total_words/1000):.2f} / 1k words)")
print(f"Total pattern matches: {len(findings)}\n")

# Group by category
by_cat = {}
for item in findings:
    by_cat.setdefault(item["category"], []).append(item)

for cat, items in by_cat.items():
    print(f"### {cat} ({len(items)} hits)")
    for it in items:
        print(f"  [{it['file']}:{it['line']}] \"{it['match']}\" ({it['label']})")
        print(f"     -> {it['snippet']}")
    print()
