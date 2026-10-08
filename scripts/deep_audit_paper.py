"""
deep_audit_paper.py - Exhaustive audit of all paper sections against avoid-ai-writing SKILL.full.md catalog
"""
import os
import re

SECTIONS_DIR = 'paper/NCAD_CS/sections'
TARGETS = [
    os.path.join(SECTIONS_DIR, f)
    for f in sorted(os.listdir(SECTIONS_DIR))
    if f.endswith('.tex')
]
TARGETS.append('paper/NCAD_CS/NCAD_CS_standalone.tex')

TIER_1A = [
    r'\bdelve\w*\b', r'\blandscape\b', r'\btapestry\b', r'\brealm\b',
    r'\bparadigm\w*\b', r'\bembark\w*\b', r'\bbeacon\b', r'\btestament\s+to\b',
    r'\brobust\b', r'\bcomprehensive\b', r'\bcutting-edge\b', r'\bleverage\w*\b',
    r'\bpivotal\b', r'\bunderscore\w*\b', r'\bmeticulous\w*\b', r'\bseamless\w*\b',
    r'\bgame-changer\b', r'\bwatershed\s+moment\b', r'\bnestled\b', r'\bvibrant\b',
    r'\bthriving\b', r'\bshowcasing\b', r'\bdeep\s+dive\b', r'\bunpack\w*\b',
    r'\bbustling\b', r'\bintricate\b', r'\bintricacies\b', r'\bcomplexities\b',
    r'\bever-evolving\b', r'\benduring\b', r'\bdaunting\b', r'\bholistic\w*\b',
    r'\bactionable\b', r'\bimpactful\b', r'\blearnings\b', r'\bthought\s+leader\w*\b',
    r'\bbest\s+practices\b', r'\bat\s+its\s+core\b', r'\bsynergy\b', r'\bsynergies\b',
    r'\binterplay\b', r'\bkeen\b', r'\bgenuinely\b', r'\bsymphony\b', r'\bembrace\w*\b',
    r'\bload-bearing\b'
]

TIER_1B = [
    r'\butilize\w*\b', r'\bin\s+order\s+to\b', r'\bdue\s+to\s+the\s+fact\s+that\b',
    r'\bserves\s+as\b', r'\bboasts\b', r'\bpresents\b', r'\bcommence\w*\b',
    r'\bascertain\w*\b', r'\bendeavor\w*\b'
]

TIER_2 = [
    r'\bharness\w*\b', r'\bnavigate\w*\b', r'\bfoster\w*\b', r'\belevate\w*\b',
    r'\bunleash\w*\b', r'\bstreamline\w*\b', r'\bempower\w*\b', r'\bbolster\w*\b',
    r'\bspearhead\w*\b', r'\bresonate\w*\b', r'\brevolutionize\w*\b',
    r'\bfacilitate\w*\b', r'\bunderpin\w*\b', r'\bnuanced\b', r'\bcrucial\b',
    r'\bmultifaceted\b', r'\becosystem\b', r'\bmyriad\b', r'\bplethora\b',
    r'\bencompass\w*\b', r'\bcatalyze\w*\b', r'\breimagine\w*\b', r'\bgalvanize\w*\b',
    r'\baugment\w*\b', r'\bcultivate\w*\b', r'\billuminate\w*\b', r'\belucidate\w*\b',
    r'\bjuxtapose\w*\b', r'\btransformative\b', r'\bcornerstone\b', r'\bparamount\b',
    r'\bpoised\b', r'\bburgeoning\b', r'\bnascent\b', r'\bquintessential\b',
    r'\boverarching\b', r'\bunderpinning\w*\b'
]

TRANSITIONS = [
    r'\bmoreover\b', r'\bfurthermore\b', r'\badditionally\b', r'\bnotably\b',
    r'\bit\s+is\s+worth\s+noting\b', r'\bit\s+is\s+important\s+to\s+note\b',
    r'\bin\s+conclusion\b', r'\bin\s+summary\b', r'\bto\s+summarize\b',
    r'\bwhen\s+it\s+comes\s+to\b', r'\bat\s+the\s+end\s+of\s+the\s+day\b',
    r'\bthat\s+said\b', r'\bthat\s+being\s+said\b'
]

HEDGES = [
    r'\bcould\s+potentially\b', r'\bmay\s+eventually\b', r'\bmight\s+ultimately\b'
]

MORAL_ADJ = [
    r'\bhonest\s+(?:shape|representation|curve|number|result|evaluation)\b',
    r'\bflagged\s+honestly\b', r'\bdescribed\s+honestly\b'
]

def audit():
    total_findings = 0
    print("================== DEEP AUDIT OF MANUSCRIPT ==================")
    for path in TARGETS:
        fname = os.path.basename(path)
        with open(path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        # Strip LaTeX comments
        lines = [l.split('%')[0] for l in raw_text.splitlines()]
        clean_text = '\n'.join(lines)

        findings = []

        # Check prose em-dashes (excluding tables)
        if not fname.startswith('tab_'):
            # Look for --- or \u2014 in prose
            prose_em_dashes = re.findall(r'---|[\u2014]', clean_text)
            if prose_em_dashes:
                findings.append(('Prose Em-Dash', len(prose_em_dashes), prose_em_dashes))

        for pat in TIER_1A:
            matches = re.findall(pat, clean_text, re.IGNORECASE)
            if matches:
                findings.append(('Tier 1A', len(matches), list(set(matches))))

        for pat in TIER_1B:
            # carve out \cite{carmona2021neural} or proper nouns if any
            matches = re.findall(pat, clean_text, re.IGNORECASE)
            if matches:
                findings.append(('Tier 1B', len(matches), list(set(matches))))

        for pat in TIER_2:
            matches = re.findall(pat, clean_text, re.IGNORECASE)
            if matches:
                findings.append(('Tier 2', len(matches), list(set(matches))))

        for pat in TRANSITIONS:
            matches = re.findall(pat, clean_text, re.IGNORECASE)
            if matches:
                findings.append(('Transition', len(matches), list(set(matches))))

        for pat in HEDGES:
            matches = re.findall(pat, clean_text, re.IGNORECASE)
            if matches:
                findings.append(('Hedge Stack', len(matches), list(set(matches))))

        for pat in MORAL_ADJ:
            matches = re.findall(pat, clean_text, re.IGNORECASE)
            if matches:
                findings.append(('Moral Adjective', len(matches), list(set(matches))))

        if findings:
            total_findings += len(findings)
            print(f"\n[!] File: {fname}")
            for cat, count, words in findings:
                print(f"    - {cat} ({count}): {words}")

    print("\n==============================================================")
    if total_findings == 0:
        print("[SUCCESS] Manuscript is 100% clean. Zero AI-isms detected across all files!")
    else:
        print(f"[ACTION REQUIRED] Found {total_findings} items to clean.")

if __name__ == '__main__':
    audit()
