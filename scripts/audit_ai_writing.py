import os
import re

files = [f for f in sorted(os.listdir('paper/NCAD_CS/sections')) if f.endswith('.tex') and not f.startswith('tab_')]

# Patterns from docs/SKILL.full.md
TIER_1A = {
    'delve': r'\bdelv(?:e|es|ed|ing)\b',
    'landscape (metaphor)': r'\blandscapes?\b',
    'tapestry': r'\btapestr(?:y|ies)\b',
    'realm': r'\brealms?\b',
    'paradigm': r'\bparadigms?\b',
    'embark': r'\bembarks?\b',
    'beacon': r'\bbeacons?\b',
    'testament to': r'\btestaments?\s+to\b',
    'cutting-edge': r'\bcutting[- ]edge\b',
    'leverage (verb)': r'\bleverag(?:e|es|ed|ing)\b',
    'pivotal': r'\bpivotal\b',
    'underscores': r'\bunderscores?\b',
    'meticulous': r'\bmeticulous(?:ly)?\b',
    'seamless': r'\bseamless(?:ly)?\b',
    'game-changer': r'\bgame[- ]chang(?:er|ing)\b',
    'deep dive': r'\bdeep[- ]dives?\b',
    'unpack': r'\bunpack(?:s|ed|ing)?\b',
    'intricate': r'\bintricate\b',
    'intricacies': r'\bintricacies\b',
    'ever-evolving': r'\bever[- ]evolving\b',
    'daunting': r'\bdaunting\b',
    'holistic': r'\bholistic(?:ally)?\b',
    'actionable': r'\bactionable\b',
    'impactful': r'\bimpactful\b',
    'learnings': r'\blearnings\b',
    'best practices': r'\bbest[- ]practices\b',
    'at its core': r'\bat\s+its\s+core\b',
    'synergy': r'\bsynerg(?:y|ies)\b',
    'interplay': r'\binterplay\b',
    'embrace': r'\bembrac(?:e|es|ed|ing)\b',
    'load-bearing': r'\bload[- ]bearing\b',
    'comprehensive': r'\bcomprehensive\b',
    'robust': r'\brobust\b',  # note: check if statistical context
}

TIER_1B = {
    'utilize': r'\butiliz(?:e|es|ed|ing)\b',
    'in order to': r'\bin\s+order\s+to\b',
    'due to the fact that': r'\bdue\s+to\s+the\s+fact\s+that\b',
    'serves as': r'\bserves\s+as\b',
    'features (verb)': r'\bfeatures\b',
    'boasts': r'\bboasts\b',
    'presents (inflated)': r'\bpresents\b',
    'commence': r'\bcommenc(?:e|es|ed|ing)\b',
    'ascertain': r'\bascertain\b',
    'endeavor': r'\bendeavor\b',
}

HOLLOW_HEDGES = {
    'genuine/genuinely': r'\bgenuinel?y?\b',
    'truly': r'\btruly\b',
    'quite frankly': r'\bquite\s+frankly\b',
    'to be honest': r'\bto\s+be\s+honest\b',
    'let\'s be clear': r'\blet\'?s\s+be\s+clear\b',
    'it is worth noting that': r'\bit(?:\'?s|\s+is)\s+worth\s+noting\s+that\b',
    'worth exploring': r'\bworth\s+exploring\b',
    'could potentially': r'\bcould\s+potentially\b',
    'it is important to note': r'\bit(?:\'?s|\s+is)\s+important\s+to\s+note\b',
    'to be clear': r'\bto\s+be\s+clear\b',
    'actually (intensifier)': r'\bactually\b',
}

TIER_2 = {
    'harness': r'\bharness(?:es|ed|ing)?\b',
    'navigate': r'\bnavigat(?:e|es|ed|ing)\b',
    'foster': r'\bfoster(?:s|ed|ing)?\b',
    'elevate': r'\belevat(?:e|es|ed|ing)\b',
    'unleash': r'\bunleash(?:es|ed|ing)?\b',
    'streamline': r'\bstreamlin(?:e|es|ed|ing)\b',
    'empower': r'\bempower(?:s|ed|ing)?\b',
    'bolster': r'\bbolster(?:s|ed|ing)?\b',
    'spearhead': r'\bspearhead(?:s|ed|ing)?\b',
    'resonate': r'\bresonat(?:e|es|ed|ing)\b',
    'revolutionize': r'\brevolutioniz(?:e|es|ed|ing)\b',
    'facilitate': r'\bfacilitat(?:e|es|ed|ing)\b',
    'underpin': r'\bunderpin(?:s|ned|ning)?\b',
    'nuanced': r'\bnuanced?\b',
    'crucial': r'\bcrucial\b',
    'multifaceted': r'\bmultifaceted\b',
    'ecosystem': r'\becosystems?\b',
    'myriad': r'\bmyriad\b',
    'plethora': r'\bplethora\b',
    'encompass': r'\bencompass(?:es|ed|ing)?\b',
    'catalyze': r'\bcatalyz(?:e|es|ed|ing)\b',
    'reimagine': r'\breimagin(?:e|es|ed|ing)\b',
    'galvanize': r'\bgalvaniz(?:e|es|ed|ing)\b',
    'augment': r'\baugment(?:s|ed|ing)?\b',
    'cultivate': r'\bcultivat(?:e|es|ed|ing)\b',
    'illuminate': r'\billuminat(?:e|es|ed|ing)\b',
    'elucidate': r'\belucidat(?:e|es|ed|ing)\b',
    'juxtapose': r'\bjuxtapos(?:e|es|ed|ing)\b',
    'transformative': r'\btransformative\b',
    'cornerstone': r'\bcornerstone\b',
    'paramount': r'\bparamount\b',
    'poised to': r'\bpoised\s+to\b',
    'burgeoning': r'\bburgeoning\b',
    'nascent': r'\bnascent\b',
    'quintessential': r'\bquintessential\b',
    'overarching': r'\boverarching\b',
}

TRANSITIONS = {
    'moreover': r'\bmoreover\b',
    'furthermore': r'\bfurthermore\b',
    'additionally': r'\badditionally\b',
}

EM_DASH = r'---'

print("=== AUDIT SUMMARY ===")
total_words = 0
total_em_dashes = 0
all_matches = []

for fname in files:
    fpath = os.path.join('paper/NCAD_CS/sections', fname)
    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # strip latex comments
    lines = content.split('\n')
    prose_lines = []
    for l in lines:
        cleaned = re.sub(r'%.*$', '', l)
        prose_lines.append(cleaned)
    prose = '\n'.join(prose_lines)
    
    words = len(re.findall(r'\b\w+\b', prose))
    total_words += words
    dashes = len(re.findall(EM_DASH, prose))
    total_em_dashes += dashes

    for line_no, line in enumerate(prose_lines, start=1):
        if not line.strip():
            continue
        
        # Check patterns
        for cat_name, pat_dict in [('Tier 1A', TIER_1A), ('Tier 1B', TIER_1B), ('Hollow/Hedge', HOLLOW_HEDGES), ('Tier 2', TIER_2), ('Transitions', TRANSITIONS)]:
            for label, pattern in pat_dict.items():
                for m in re.finditer(pattern, line, re.IGNORECASE):
                    all_matches.append({
                        'file': fname,
                        'line': line_no,
                        'category': cat_name,
                        'label': label,
                        'match': m.group(0),
                        'context': line.strip()[:140]
                    })

print(f"Total prose words across paper: {total_words}")
print(f"Total em-dashes (---): {total_em_dashes} (Rate: {total_em_dashes / (total_words/1000):.2f} per 1,000 words)")
print(f"Total pattern matches: {len(all_matches)}\n")

for m in all_matches:
    print(f"[{m['file']}:{m['line']}] ({m['category']}) \"{m['match']}\" [{m['label']}]")
    print(f"   -> {m['context']}\n")
