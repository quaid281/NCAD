#!/usr/bin/env python3
"""
deep_ai_audit.py - Exhaustive sweep against docs/SKILL.md and docs/SKILL.full.md
Checks all sections and tables in paper/NCAD_CS/
"""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECTIONS_DIR = ROOT / "paper" / "NCAD_CS" / "sections"

FILES_TO_CHECK = [
    "02_title.tex",
    "03_abstract.tex",
    "04_intro.tex",
    "05_related.tex",
    "06_dataset.tex",
    "07_method.tex",
    "08_experiments.tex",
    "09_variants.tex",
    "10_results.tex",
    "11_future.tex",
    "tab_all_datasets.tex",
    "tab_core_benchmark.tex",
    "tab_dataset_prauc.tex",
]

# Comprehensive rule definitions based on SKILL.full.md
RULES_P0 = {
    "Cutoff disclaimer": r"\bas of my (?:last )?update\b|\bi don'?t have access to real-time\b",
    "Chatbot artifact": r"\b(?:i hope this helps|great question|certainly!|sure thing!|feel free to reach out|let me know if you need)\b",
    "Vague attribution without source": r"\b(?:experts believe|studies show|research suggests|industry leaders agree|analysts agree|studies consistently show)\b(?!\s*\\cite)",
    "Significance inflation": r"\b(?:historic milestone|watershed moment|groundbreaking|breathtaking|astounding)\b",
    "Meta narration": r"\blet'?s\s+(?:dive|explore|examine|unpack|look at|delve)\b|\bin this article,? we will\b",
}

RULES_P1_VOCAB_1A = {
    "delve": r"\bdelv(?:e|es|ed|ing)\b",
    "landscape (metaphor)": r"\b(?:telemetry|anomaly detection|monitoring|operational|machine learning)\s+landscape\b|\bthe landscape of\b",
    "tapestry": r"\btapestr(?:y|ies)\b",
    "realm": r"\brealms?\b",
    "paradigm (metaphor)": r"\b(?:novel|new|unprecedented|shifting)\s+paradigms?\b|\bparadigm shift\b",
    "embark": r"\bembarks?\b",
    "beacon": r"\bbeacons?\b",
    "testament to": r"\btestaments?\s+to\b",
    "robust": r"\brobust\b",
    "comprehensive": r"\bcomprehensive\b",
    "cutting-edge": r"\bcutting[- ]edge\b",
    "leverage (verb)": r"\bleverag(?:e|es|ed|ing)\b",
    "pivotal": r"\bpivotal\b",
    "underscores (verb)": r"\bunderscores?\b",
    "meticulous": r"\bmeticulous(?:ly)?\b",
    "seamless": r"\bseamless(?:ly)?\b",
    "game-changer": r"\bgame[- ]chang(?:er|ing)\b",
    "hit differently": r"\bhits?\s+differently?\b",
    "marking a pivotal moment": r"\bmarking\s+a\s+pivotal\s+moment\b",
    "the future looks bright": r"\bthe\s+future\s+looks\s+bright\b",
    "only time will tell": r"\bonly\s+time\s+will\s+tell\b",
    "nestled": r"\bnestled\b",
    "vibrant": r"\bvibrant\b",
    "thriving": r"\bthriving\b",
    "showcasing": r"\bshowcas(?:ing|es|ed)\b",
    "deep dive": r"\bdeep[- ]dives?\b",
    "unpack": r"\bunpack(?:s|ed|ing)?\b",
    "bustling": r"\bbustling\b",
    "intricate / intricacies": r"\bintricate(?:ly)?\b|\bintricacies\b",
    "complexities": r"\bcomplexities\b",
    "ever-evolving": r"\bever[- ]evolving\b",
    "enduring": r"\benduring\b",
    "daunting": r"\bdaunting\b",
    "holistic": r"\bholistic(?:ally)?\b",
    "actionable": r"\bactionable\b",
    "impactful": r"\bimpactful\b",
    "learnings": r"\blearnings\b",
    "thought leader": r"\bthought\s+lead(?:er|ership)\b",
    "best practices": r"\bbest[- ]practices?\b",
    "at its core": r"\bat\s+its\s+core\b",
    "synergy": r"\bsynerg(?:y|ies)\b",
    "interplay": r"\binterplay\b",
    "keen (intensifier)": r"\bkeen(?:ly)?\b",
    "genuine / genuinely (intensifier)": r"\bgenuine(?:ly)?\b",
    "symphony (metaphor)": r"\bsymphon(?:y|ies)\b",
    "embrace (metaphor)": r"\bembrac(?:e|es|ed|ing)\b",
    "load-bearing (metaphor)": r"\bload[- ]bearing\b",
}

RULES_P1_VOCAB_1B = {
    "utilize": r"\butiliz(?:e|es|ed|ing)\b",
    "in order to": r"\bin\s+order\s+to\b",
    "due to the fact that": r"\bdue\s+to\s+the\s+fact\s+that\b",
    "serves as": r"\bserves\s+as\b",
    "features (verb)": r"\bfeatures\b",
    "boasts": r"\bboasts\b",
    "presents (inflated)": r"\bpresents\s+a\s+(?:unique|novel|comprehensive)\b",
    "commence": r"\bcommenc(?:e|es|ed|ing)\b",
    "ascertain": r"\bascertain\b",
    "endeavor": r"\bendeavor\b",
}

RULES_P1_PATTERNS = {
    "Not only X but also Y": r"\bnot\s+only\b.*?\bbut\s+(?:also\b)?",
    "Isn't just X, it's Y": r"\bisn'?t\s+just\b.*?\bit'?s\b|\bis\s+not\s+just\b.*?\bit\s+is\b",
    "Hedge-stacked prediction": r"\b(?:could\s+potentially|may\s+potentially|might\s+potentially|may\s+eventually|could\s+conceivably)\b",
    "Generic future-narrative closer": r"\b(?:may|could|will|is poised to)\s+become\s+(?:one\s+of\s+the\s+most\s+)?(?:important|defining)\s+(?:narratives|trends|chapters)\b",
    "Narrated candor": r"\b(?:i\s+would\s+rather\s+flag|in\s+the\s+interest\s+of\s+full\s+disclosure|to\s+be\s+fully\s+transparent|rather\s+than\s+bury\s+this)\b",
    "Real/actual adjective inflation": r"\b(?:real|actual|genuine|true)\s+(?:on-chain|tokenomics|reward|utility|sustainability)\b",
    "Moral-adjective category error": r"\b(?:honest|truthful|faithful)\s+(?:shape|number|accuracy|representation|curve|output)\b|\b(?:flagged|described)\s+honestly\b",
    "Unfilled placeholder": r"\[(?:Your|Insert|Add|Enter|Describe|Specify|Choose)[^\]]+\]|\b\d{4}-XX-XX\b",
    "Chatbot citation leak": r"\b(?:citeturn\d+|contentReference|oaicite|grok_card)\b",
    "Infomercial hook": r"\b(?:the\s+catch\?|the\s+kicker\?|here'?s\s+the\s+thing\.|plot\s+twist:)\b",
    "Social endorsement closer": r"\b(?:this\s+one\s+is\s+worth|thank\s+me\s+later|bookmark\s+this|save\s+this\s+for\s+later)\b",
    "Lingering attention claim": r"\b(?:the\s+line\s+i\s+keep\s+coming\s+back\s+to|can'?t\s+stop\s+thinking\s+about\s+this)\b",
}

RULES_P2_STYLISTIC = {
    "Copula avoidance": r"\b(?:acts\s+as\s+(?:a|an)|functions\s+as\s+(?:a|an)|stands\s+as\s+(?:a|an)|serves\s+to|marks\s+a)\b",
    "Overused transition": r"\b(?:moreover|furthermore|additionally|that\s+being\s+said|with\s+that\s+in\s+mind)\b",
    "Filler phrase": r"\b(?:it\s+is\s+worth\s+noting\s+that|it\s+is\s+important\s+to\s+note\b|at\s+the\s+end\s+of\s+the\s+day|when\s+it\s+comes\s+to)\b",
    "Hollow intensifier": r"\b(?:quite\s+frankly|to\s+be\s+honest|let'?s\s+be\s+clear)\b",
    "Confidence calibration cue": r"\b(?:interestingly,|surprisingly,|notably,|without\s+a\s+doubt)\b",
    "Unnecessary hyphenation": r"\bin\s+real-time\b|\bdata-set\b|\bcode-base\b|\btime-frame\b",
}

RULES_TIER2_INDIVIDUAL = {
    "harness": r"\bharness(?:es|ed|ing)?\b",
    "navigate": r"\bnavigat(?:e|es|ed|ing)\b",
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
    "ecosystem": r"\becosystems?\b",
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
}


def audit():
    total_words = 0
    total_em_dashes = 0
    findings = []

    for fname in FILES_TO_CHECK:
        fpath = SECTIONS_DIR / fname
        if not fpath.exists():
            print(f"Warning: {fname} does not exist")
            continue

        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Split into lines
        lines = content.split("\n")
        prose_lines = []
        for l in lines:
            # strip comments
            cleaned = re.sub(r"%.*$", "", l)
            prose_lines.append(cleaned)
        prose = "\n".join(prose_lines)

        words = len(re.findall(r"\b\w+\b", prose))
        total_words += words

        # Check em dashes (---, —, or -- used as punctuation dash)
        # Note: -- in LaTeX is an en-dash, used for ranges like 1--10. Check if used as em-dash (surrounded by spaces)
        em3 = len(re.findall(r"---", prose))
        em_unicode = len(re.findall(r"—", prose))
        # en-dash used as punctuation dash with spaces e.g. "word -- word"
        en_dash_punct = len(re.findall(r"\s--\s", prose))
        dashes = em3 + em_unicode + en_dash_punct
        total_em_dashes += dashes

        if dashes > 0:
            for line_no, line in enumerate(prose_lines, start=1):
                if re.search(r"---|—|\s--\s", line):
                    findings.append({
                        "file": fname,
                        "line": line_no,
                        "category": "P2: Formatting (Em-dash)",
                        "label": "em-dash",
                        "match": "em-dash",
                        "snippet": line.strip()[:140]
                    })

        # Check line-by-line against rules
        for line_no, line in enumerate(prose_lines, start=1):
            sline = line.strip()
            if not sline:
                continue
            # Skip pure LaTeX equation blocks or figure definitions
            if sline.startswith("\\begin{equation}") or sline.startswith("\\end{equation}"):
                continue
            if sline.startswith("\\begin{figure") or sline.startswith("\\end{figure"):
                continue
            if sline.startswith("\\includegraphics") or sline.startswith("\\label") or sline.startswith("\\cite"):
                continue

            # Check P0
            for label, pat in RULES_P0.items():
                for m in re.finditer(pat, line, re.IGNORECASE):
                    findings.append({
                        "file": fname,
                        "line": line_no,
                        "category": "P0: Credibility Killers",
                        "label": label,
                        "match": m.group(0),
                        "snippet": sline[:140]
                    })

            # Check P1 1A
            for label, pat in RULES_P1_VOCAB_1A.items():
                for m in re.finditer(pat, line, re.IGNORECASE):
                    findings.append({
                        "file": fname,
                        "line": line_no,
                        "category": "P1: Tier 1A Frequency Marker",
                        "label": label,
                        "match": m.group(0),
                        "snippet": sline[:140]
                    })

            # Check P1 1B
            for label, pat in RULES_P1_VOCAB_1B.items():
                for m in re.finditer(pat, line, re.IGNORECASE):
                    findings.append({
                        "file": fname,
                        "line": line_no,
                        "category": "P1: Tier 1B Wordiness / Inflated Formality",
                        "label": label,
                        "match": m.group(0),
                        "snippet": sline[:140]
                    })

            # Check P1 Patterns
            for label, pat in RULES_P1_PATTERNS.items():
                for m in re.finditer(pat, line, re.IGNORECASE):
                    findings.append({
                        "file": fname,
                        "line": line_no,
                        "category": "P1: Structural & Rhetorical Patterns",
                        "label": label,
                        "match": m.group(0),
                        "snippet": sline[:140]
                    })

            # Check P2 Stylistic
            for label, pat in RULES_P2_STYLISTIC.items():
                for m in re.finditer(pat, line, re.IGNORECASE):
                    findings.append({
                        "file": fname,
                        "line": line_no,
                        "category": "P2: Stylistic Polish",
                        "label": label,
                        "match": m.group(0),
                        "snippet": sline[:140]
                    })

            # Check Tier 2 individual (record separately to check cluster vs isolated)
            for label, pat in RULES_TIER2_INDIVIDUAL.items():
                for m in re.finditer(pat, line, re.IGNORECASE):
                    findings.append({
                        "file": fname,
                        "line": line_no,
                        "category": "Tier 2 Candidate (Legitimate in isolation, check cluster)",
                        "label": label,
                        "match": m.group(0),
                        "snippet": sline[:140]
                    })

    print("=================================================================")
    print("           DEEP SKILL.MD AUDIT REPORT FOR NCAD-CS PAPER          ")
    print("=================================================================")
    print(f"Total files audited: {len(FILES_TO_CHECK)}")
    print(f"Total prose words:   {total_words}")
    print(f"Total em-dashes:     {total_em_dashes} (rate: {total_em_dashes / (total_words/1000):.2f} / 1k words)")
    print()

    # Filter into strict violations (P0, P1, P2) vs Tier 2 candidates
    strict_violations = [f for f in findings if not f["category"].startswith("Tier 2 Candidate")]
    tier2_hits = [f for f in findings if f["category"].startswith("Tier 2 Candidate")]

    print(f"STRICT VIOLATIONS (P0, P1, P2): {len(strict_violations)}")
    if strict_violations:
        by_cat = {}
        for it in strict_violations:
            by_cat.setdefault(it["category"], []).append(it)
        for cat, items in by_cat.items():
            print(f"\n--- {cat} ({len(items)} hits) ---")
            for it in items:
                print(f"  [{it['file']}:{it['line']}] \"{it['match']}\" ({it['label']})")
                print(f"     -> {it['snippet']}")
    else:
        print("  -> None found! Clean sweep for P0, P1, P2!")

    print(f"\nTIER 2 CANDIDATE SCAN (Evaluated for clusters within paragraphs): {len(tier2_hits)}")
    if tier2_hits:
        # Group by file and approximate paragraph
        print(f"  Total individual Tier 2 occurrences: {len(tier2_hits)}")
        by_label = {}
        for it in tier2_hits:
            by_label.setdefault(it["label"], []).append(it)
        for lbl, hits in sorted(by_label.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"    - '{lbl}': {len(hits)} occurrence(s)")
            for h in hits[:2]:
                print(f"        [{h['file']}:{h['line']}] \"{h['snippet'][:100]}...\"")

    return strict_violations


if __name__ == "__main__":
    audit()
