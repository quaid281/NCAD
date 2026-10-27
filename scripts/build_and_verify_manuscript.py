"""
build_and_verify_manuscript.py - Build standalone manuscript, compile with tectonic,
and exhaustively scan for unresolved cross-references (??), missing citations ([?]),
and page-budget violations.
"""

import os
import re
import sys
import subprocess
import difflib
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = WORKSPACE_ROOT / "paper" / "NCAD_CS"
SECTIONS_DIR = PAPER_DIR / "sections"
STANDALONE_TEX = PAPER_DIR / "NCAD_CS_standalone.tex"
SUPPLEMENT_TEX = PAPER_DIR / "NCAD_CS_supplement.tex"
STANDALONE_PDF = PAPER_DIR / "NCAD_CS_standalone.pdf"
SUPPLEMENT_PDF = PAPER_DIR / "NCAD_CS_supplement.pdf"
BIB_FILE = PAPER_DIR / "export.bib"

MAX_PAGE_BUDGET = 13

def step(msg):
    print(f"\n[+] {msg}")

def warn(msg):
    print(f"  [!] WARNING: {msg}")

def err(msg):
    print(f"  [x] ERROR: {msg}")

def build_standalone():
    step("1. Reconstructing NCAD_CS_standalone.tex from sections/...")
    from build_standalone import build as build_latex
    build_latex()

def run_tectonic(target_file):
    import shutil
    tectonic_bin = shutil.which("tectonic") or "tectonic"
    cmd = [tectonic_bin, "--keep-intermediates", str(target_file)]
    res = subprocess.run(cmd, cwd=str(PAPER_DIR), capture_output=True, text=True, shell=True)
    if res.returncode != 0:
        err(f"Tectonic compilation failed for {target_file.name if hasattr(target_file, 'name') else target_file}:")
        print(res.stderr or res.stdout)
        return False
    return True

def compile_documents():
    step("2. Compiling the self-contained standalone manuscript via Tectonic (--keep-intermediates)...")
    if not run_tectonic(STANDALONE_TEX.name):
        return False
    return True

def scan_pdf_for_markers(pdf_path):
    import pypdf
    issues = []
    if not pdf_path.exists():
        return [f"PDF not found: {pdf_path}"]
        
    reader = pypdf.PdfReader(str(pdf_path))
    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        text = page.extract_text() or ""
        lines = text.split("\n")
        for line in lines:
            if "??" in line:
                issues.append(f"Page {page_num} [unresolved reference '??']: {line.strip()}")
            # Match unresolved citations like [?], [?, ?]
            if re.search(r"\[\?+\]|\[\?+,\s*\?+\]", line):
                issues.append(f"Page {page_num} [unresolved citation '[?]']: {line.strip()}")
    return issues

def scan_bib_keys():
    keys = set()
    if not BIB_FILE.exists():
        return keys
    content = BIB_FILE.read_text(encoding="utf-8")
    for match in re.finditer(r"@\w+\s*\{\s*([^,\s]+),", content):
        keys.add(match.group(1).strip())
    return keys

def _is_manuscript_section(path):
    """True for section files that belong to the self-contained manuscript (not legacy sup_*/unused tab_* files)."""
    if path.name.startswith("sup_"):
        return False
    if path.name.startswith("tab_"):
        used = "".join(p.read_text(encoding="utf-8") for p in SECTIONS_DIR.glob("[0-9]*.tex"))
        return f"sections/{path.name}" in used
    return True

def scan_tex_crossrefs():
    step("3. Scanning .tex files for unresolved labels, citations, and placeholders...")
    
    # 1. Collect all labels
    labels = set()
    tex_files = [f for f in SECTIONS_DIR.glob("*.tex") if _is_manuscript_section(f)] + [STANDALONE_TEX]
    
    label_def_pattern = re.compile(r"\\label\{([^}]+)\}")
    ref_pattern = re.compile(r"\\(ref|eqref|cref|autoref)\{([^}]+)\}")
    cite_pattern = re.compile(r"\\cite[a-zA-Z]*\{([^}]+)\}")
    placeholder_pattern = re.compile(r"(?<![\w\\])\?\?(?![\w])")
    
    file_contents = {}
    for tf in tex_files:
        if tf.exists():
            text = tf.read_text(encoding="utf-8", errors="ignore")
            file_contents[tf] = text
            for m in label_def_pattern.finditer(text):
                labels.add(m.group(1).strip())

    bib_keys = scan_bib_keys()
    
    errors = []
    warnings = []
    
    for tf, text in file_contents.items():
        rel_name = tf.relative_to(WORKSPACE_ROOT) if tf.is_relative_to(WORKSPACE_ROOT) else tf.name
        lines = text.splitlines()
        for line_no, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("%"):
                continue  # skip full comment lines
            
            # Remove trailing comments
            code_line = re.sub(r"(?<!\\)%.*$", "", line)
            
            # Check for literal ?? placeholder
            if placeholder_pattern.search(code_line):
                warnings.append(f"{rel_name}:{line_no}: Contains literal '??' placeholder: '{code_line.strip()}'")
                
            # Check references
            for rm in ref_pattern.finditer(code_line):
                ref_key = rm.group(2).strip()
                if ref_key not in labels:
                    matches = difflib.get_close_matches(ref_key, list(labels), n=3, cutoff=0.6)
                    hint = f" (did you mean: {', '.join(matches)}?)" if matches else ""
                    errors.append(f"{rel_name}:{line_no}: Undefined reference '\\{rm.group(1)}{{{ref_key}}}'{hint}")
                    
            # Check citations
            for cm in cite_pattern.finditer(code_line):
                cite_entries = [c.strip() for c in cm.group(1).split(",")]
                for ck in cite_entries:
                    if ck and ck not in bib_keys:
                        matches = difflib.get_close_matches(ck, list(bib_keys), n=3, cutoff=0.6)
                        hint = f" (did you mean: {', '.join(matches)}?)" if matches else ""
                        errors.append(f"{rel_name}:{line_no}: Undefined citation '\\cite{{{ck}}}'{hint}")

    return errors, warnings

def check_page_budget():
    step("4. Checking page count against IEEE TNNLS budget...")
    import pypdf
    reader = pypdf.PdfReader(str(STANDALONE_PDF))
    page_count = len(reader.pages)
    print(f"  -> Manuscript page count: {page_count} pages (Hard ceiling: {MAX_PAGE_BUDGET} pages)")
    
    if page_count > MAX_PAGE_BUDGET:
        warn(f"Page count ({page_count}) exceeds hard budget of {MAX_PAGE_BUDGET} pages! Over by {page_count - MAX_PAGE_BUDGET} page(s).")
        return False
    else:
        print(f"  [OK] Page count is within budget ({page_count}/{MAX_PAGE_BUDGET} pages).")
        return True

def main():
    print("=" * 70)
    print(" NCAD-CS Manuscript Build, Reference Scanner & Quality Verifier")
    print("=" * 70)
    
    build_standalone()
    
    # Pre-compilation static check
    tex_errors, tex_warnings = scan_tex_crossrefs()
    
    if tex_warnings:
        print(f"\nFound {len(tex_warnings)} placeholder warnings in .tex source:")
        for w in tex_warnings[:10]:
            warn(w)
        if len(tex_warnings) > 10:
            warn(f"... and {len(tex_warnings) - 10} more.")
            
    if tex_errors:
        print(f"\nFound {len(tex_errors)} undefined reference/citation errors in .tex source:")
        for e in tex_errors:
            err(e)
            
    # Compile
    comp_ok = compile_documents()
    if not comp_ok:
        err("Compilation failed. Fix LaTeX syntax errors above.")
        sys.exit(1)
        
    # Post-compilation PDF scan
    step("5. Scanning generated PDFs for rendered '??' and '[?]' markers...")
    standalone_issues = scan_pdf_for_markers(STANDALONE_PDF)
    
    has_pdf_issues = False
    if standalone_issues:
        has_pdf_issues = True
        err(f"Found {len(standalone_issues)} rendered reference issues in {STANDALONE_PDF.name}:")
        for issue in standalone_issues:
            print(f"    {issue}")
    else:
        print(f"  [OK] {STANDALONE_PDF.name}: 0 rendered '??' or '[?]' issues.")
        
    # Check page budget
    budget_ok = check_page_budget()
    
    # 6. Anti-AI writing pattern audit (docs/SKILL.md)
    step("6. Auditing manuscript prose against anti-AI writing rules (docs/SKILL.md)...")
    from scan_ai_writing import run_scan
    ai_audit_ok = run_scan()

    # 7. IEEE acronym audit: define each acronym at first use
    step("7. Auditing acronym definitions at first use (IEEE style)...")
    from audit_acronyms import run_audit
    acronym_ok = run_audit()
    
    print("\n" + "=" * 70)
    if not tex_errors and not has_pdf_issues and budget_ok and ai_audit_ok and acronym_ok:
        print(" [SUCCESS] All cross-references, page budget, and SKILL.md anti-AI rules verified cleanly!")
        print("=" * 70)
        sys.exit(0)
    else:
        if not budget_ok:
            print(f" [ACTION REQUIRED] Reduce text/figures to satisfy the <= {MAX_PAGE_BUDGET} page ceiling.")
        if not ai_audit_ok:
            print(" [ACTION REQUIRED] Eliminate identified AI writing patterns per docs/SKILL.md.")
        if not acronym_ok:
            print(" [ACTION REQUIRED] Define each acronym as Full Name (ACR) at first use.")
        if tex_errors or has_pdf_issues:
            print(" [ACTION REQUIRED] Resolve broken cross-references or citations.")
        print("=" * 70)
        sys.exit(1)

if __name__ == "__main__":
    main()
