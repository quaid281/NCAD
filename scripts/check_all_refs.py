import re
import glob
from pathlib import Path

def main():
    aux_files = glob.glob('paper/NCAD_CS/*.aux')
    print(f"Found aux files: {aux_files}")
    defined_labels = set()
    for aux in aux_files:
        with open(aux, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            labels = re.findall(r'\\newlabel\{([^}]+)\}', content)
            defined_labels.update(labels)
            print(f"  {aux}: {len(labels)} labels")

    print(f"\nTotal defined labels: {len(defined_labels)}")

    tex_files = sorted(glob.glob('paper/NCAD_CS/sections/*.tex'))
    all_refs = {}
    for sec in tex_files:
        with open(sec, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        for idx, l in enumerate(lines, 1):
            clean = l.split('%')[0]
            refs = re.findall(r'\\(?:ref|pageref|autoref|eqref|Cref|cref)\{([^}]+)\}', clean)
            for r in refs:
                all_refs.setdefault(r, []).append((sec, idx, l.strip()))

    print(f"Total unique referenced labels: {len(all_refs)}")

    missing = {}
    for r, occurrences in all_refs.items():
        if r not in defined_labels:
            missing[r] = occurrences

    print(f"\nMISSING LABELS ({len(missing)}):")
    for r, occurrences in sorted(missing.items()):
        print(f"\n[!] Missing Label: '{r}'")
        for sec, idx, line in occurrences:
            print(f"    {sec}:{idx} -> {line[:120]}")

if __name__ == '__main__':
    main()
