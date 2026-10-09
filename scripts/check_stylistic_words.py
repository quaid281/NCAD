import os
import re

files = [f for f in sorted(os.listdir('paper/NCAD_CS/sections')) if f.endswith('.tex') and not f.startswith('tab_')]

check_words = {
    'crucial': r'\bcrucial(?:ly)?\b',
    'furthermore': r'\bfurthermore\b',
    'moreover': r'\bmoreover\b',
    'additionally': r'\badditionally\b',
    'in order to': r'\bin\s+order\s+to\b',
    'not only': r'\bnot\s+only\b',
    'remarkable': r'\bremarkabl(?:e|y)\b',
    'significant': r'\bsignificant(?:ly)?\b',
    'vital': r'\bvital\b',
    'essential': r'\bessential\b',
    'intricate': r'\bintricate\b',
    'seamless': r'\bseamless\b',
    'cornerstone': r'\bcornerstone\b',
    'paramount': r'\bparamount\b'
}

for fname in files:
    fpath = os.path.join('paper/NCAD_CS/sections', fname)
    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    for i, line in enumerate(lines, 1):
        clean = re.sub(r'%.*$', '', line).strip()
        for label, pat in check_words.items():
            for m in re.finditer(pat, clean, re.IGNORECASE):
                print(f"{fname}:{i} [{label}] \"{m.group(0)}\" -> {clean[:120]}")
