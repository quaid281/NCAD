"""
build_standalone.py - Reconstruct NCAD_CS_standalone.tex from sections/
"""
import os

SECTIONS_DIR = os.path.join("paper", "NCAD_CS", "sections")
OUTPUT_PATH = os.path.join("paper", "NCAD_CS", "NCAD_CS_standalone.tex")

SECTION_FILES = [
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
]

def build():
    with open(os.path.join(SECTIONS_DIR, "01_preamble.tex"), "r", encoding="utf-8") as f:
        preamble = f.read()

    lines = [
        "% Standalone document for NCAD-CS",
        "\\documentclass[journal]{IEEEtran}",
        "",
        preamble.strip(),
        "",
        "\\begin{document}",
        ""
    ]

    for sec in SECTION_FILES:
        sec_path = os.path.join(SECTIONS_DIR, sec)
        lines.append(f"% --- Input from {sec} ---")
        with open(sec_path, "r", encoding="utf-8") as f:
            lines.append(f.read().strip())
        lines.append("")

    lines.extend([
        "% Bibliography",
        "\\def\\IEEEbibitemsep{0.5pt plus 0.5pt}",
        "{\\scriptsize",
        "\\bibliographystyle{IEEEtran}",
        "\\bibliography{export}}",
        "",
        "\\end{document}",
        ""
    ])

    content = "\n".join(lines)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Successfully generated {OUTPUT_PATH} ({len(content)} bytes)")

if __name__ == "__main__":
    build()
