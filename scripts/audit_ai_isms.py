import os
import re

sections_dir = 'paper/NCAD_CS/sections'

tier1a = [
    'delve', 'landscape', 'tapestry', 'realm', 'paradigm', 'embark', 'beacon',
    'testament to', 'robust', 'comprehensive', 'cutting-edge', 'leverage',
    'pivotal', 'underscores', 'meticulous', 'seamless', 'game-changer',
    'watershed moment', 'nestled', 'vibrant', 'thriving', 'showcasing',
    'deep dive', 'unpack', 'bustling', 'intricate', 'complexities',
    'ever-evolving', 'enduring', 'daunting', 'holistic', 'actionable',
    'impactful', 'learnings', 'thought leader', 'best practices', 'at its core',
    'synergy', 'interplay', 'keen', 'genuinely', 'symphony', 'embrace'
]

tier1b = [
    'utilize', 'in order to', 'due to the fact that', 'serves as',
    'boasts', 'presents', 'commence', 'ascertain', 'endeavor'
]

tier2 = [
    'harness', 'navigate', 'foster', 'elevate', 'unleash', 'streamline',
    'empower', 'bolster', 'spearhead', 'resonate', 'revolutionize',
    'facilitate', 'underpin', 'nuanced', 'crucial', 'multifaceted',
    'ecosystem', 'myriad', 'plethora', 'encompass', 'catalyze',
    'reimagine', 'galvanize', 'augment', 'cultivate', 'illuminate',
    'elucidate', 'juxtapose', 'transformative', 'cornerstone',
    'paramount', 'poised', 'burgeoning', 'nascent', 'quintessential',
    'overarching', 'underpinning'
]

transitions = [
    'moreover', 'furthermore', 'additionally', 'notably',
    'it is worth noting', 'it is important to note', 'in conclusion', 'in summary'
]

files = [os.path.join(sections_dir, f) for f in sorted(os.listdir(sections_dir)) if f.endswith('.tex')]
files.append('paper/NCAD_CS/NCAD_CS_standalone.tex')

print('=== SCANNING FILES FOR AI-ISMS ===')
for path in files:
    f = os.path.basename(path)
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Exclude latex comments
    lines = [line.split('%')[0] for line in content.splitlines()]
    clean_content = '\n'.join(lines)
    
    em_dashes = len(re.findall(r'---|[\u2014]', clean_content))
    # text en dashes that are used like em-dashes (not between numbers)
    text_en_dashes = re.findall(r'(?<![0-9\\])--(?![-0-9])', clean_content)
    
    hits = []
    for w in tier1a:
        m = re.findall(rf'\b{w}\w*\b', clean_content, re.IGNORECASE)
        if m:
            hits.append((f'Tier 1A: {w}', len(m), list(set(m))))
            
    for w in tier1b:
        m = re.findall(rf'\b{w}\w*\b', clean_content, re.IGNORECASE)
        if m:
            hits.append((f'Tier 1B: {w}', len(m), list(set(m))))
            
    for w in tier2:
        m = re.findall(rf'\b{w}\w*\b', clean_content, re.IGNORECASE)
        if m:
            hits.append((f'Tier 2: {w}', len(m), list(set(m))))
            
    for w in transitions:
        m = re.findall(rf'\b{w}\b', clean_content, re.IGNORECASE)
        if m:
            hits.append((f'Transition: {w}', len(m), list(set(m))))
            
    if hits or em_dashes > 0 or len(text_en_dashes) > 0:
        print(f'\nFile: {f} (Em-dashes/---: {em_dashes}, text --: {len(text_en_dashes)})')
        for cat, count, words in hits:
            print(f'  [{cat}]: {count} occurrences -> {words}')
