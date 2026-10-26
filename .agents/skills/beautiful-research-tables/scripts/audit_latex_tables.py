#!/usr/bin/env python3
"""
Automated LaTeX Table Typographic & Structural Auditor
(`beautiful-research-tables` skill)

Audits .tex files against the publication-grade table rules from TeX.SE #112343
("Beautiful table samples"), TeX.SE #681429 ("How to make better looking tables
for a publication"), Simon Fear's `booktabs` guidelines, and `IEEEtran`
two-column standards.

Checks performed:
  [P0] RESIZEBOX_TABLE     - \resizebox or \scalebox wrapping a tabular environment
  [P0] VERTICAL_RULE       - Vertical pipes `|` in tabular colspec or \vline in body
  [P0] HLINE_OR_CLINE      - Raw \hline or \cline used instead of booktabs rules
  [P1] UNTRIMMED_CMIDRULE  - \cmidrule{a-b} missing (lr), (l), or (r) trim spec
  [P1] REPEATED_BODY_UNITS - Repetitive inline `%` (`\%`) or unit suffixes in rows
  [P1] CRAMPED_STRETCH     - \arraystretch set below 0.90 (crushes booktabs padding)
  [P2] MISSING_OUTER_TRIM  - Tabular colspec missing leading/trailing `@{}`
  [P2] CAPTION_BELOW       - \caption placed after \end{tabular} instead of above
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Finding:
    severity: str  # "P0", "P1", "P2"
    code: str
    line: int
    message: str
    suggestion: str


@dataclass
class TableReport:
    file: str
    env_type: str
    label: str
    start_line: int
    end_line: int
    colspec: str
    findings: List[Finding] = field(default_factory=list)


TABLE_ENV_RE = re.compile(
    r"\\begin\{(table\*?|wraptable)\}(.*?)\\end\{\1\}",
    re.DOTALL,
)

TABULAR_ENV_NAME_RE = re.compile(r"\\begin\{(tabular\*?|tabularx|tblr)\}")


def extract_brace_group(text: str, pos: int) -> tuple[str, int]:
    """Extract a balanced {...} group starting at or after pos."""
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if pos < len(text) and text[pos] == "[":
        # Skip optional [pos] argument such as [!t] or [t]
        depth_sq = 0
        while pos < len(text):
            if text[pos] == "[":
                depth_sq += 1
            elif text[pos] == "]":
                depth_sq -= 1
                if depth_sq == 0:
                    pos += 1
                    break
            pos += 1
        while pos < len(text) and text[pos].isspace():
            pos += 1
    if pos >= len(text) or text[pos] != "{":
        return "", pos
    depth = 0
    start = pos + 1
    while pos < len(text):
        if text[pos] == "{" and (pos == 0 or text[pos - 1] != "\\"):
            depth += 1
        elif text[pos] == "}" and (pos == 0 or text[pos - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return text[start:pos], pos + 1
        pos += 1
    return text[start:], len(text)


def extract_tabular_colspec(block_text: str) -> str:
    """Extract the column specification of tabular, tabular*, tabularx, or tblr."""
    m = TABULAR_ENV_NAME_RE.search(block_text)
    if not m:
        return ""
    env_name = m.group(1)
    pos = m.end()
    first_arg, pos = extract_brace_group(block_text, pos)
    if env_name in ("tabular*", "tabularx"):
        second_arg, _ = extract_brace_group(block_text, pos)
        return " ".join(second_arg.split())
    return " ".join(first_arg.split())

LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
RESIZEBOX_RE = re.compile(r"\\(?:resizebox|scalebox)\b")
HLINE_RE = re.compile(r"\\(?:hline|cline)\b")
VLINE_RE = re.compile(r"\\vline\b")
CMIDRULE_UNTRIMMED_RE = re.compile(r"\\cmidrule\s*\{(\d+-\d+)\}")
ARRAYSTRETCH_RE = re.compile(r"\\renewcommand\{\\arraystretch\}\{([0-9.]+)\}")


def strip_comments(tex: str) -> str:
    """Remove LaTeX comments while preserving line count."""
    cleaned_lines = []
    for line in tex.splitlines():
        # Match first unescaped %
        m = re.search(r"(?<!\\)%", line)
        if m:
            cleaned_lines.append(line[: m.start()])
        else:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def has_vertical_pipe_in_colspec(colspec: str) -> bool:
    """Check if a tabular column specification contains unescaped vertical pipe |."""
    # Ignore @{...} expressions that don't use |
    stripped = re.sub(r"@\{[^}]*\}", "", colspec)
    return "|" in stripped


def audit_table_block(
    file_path: Path,
    env_type: str,
    block_text: str,
    start_line: int,
    end_line: int,
) -> TableReport:
    label_match = LABEL_RE.search(block_text)
    label = label_match.group(1) if label_match else "(unlabeled)"

    colspec = extract_tabular_colspec(block_text)

    report = TableReport(
        file=str(file_path),
        env_type=env_type,
        label=label,
        start_line=start_line,
        end_line=end_line,
        colspec=colspec,
    )

    lines = block_text.splitlines()

    # 1. [P0] Check for \resizebox or \scalebox
    for idx, line in enumerate(lines):
        if RESIZEBOX_RE.search(line):
            report.findings.append(
                Finding(
                    severity="P0",
                    code="RESIZEBOX_TABLE",
                    line=start_line + idx,
                    message="Table uses \\resizebox or \\scalebox, distorting glyph sizes and booktabs rule weights.",
                    suggestion="Remove \\resizebox; use \\footnotesize or \\scriptsize, tune \\tabcolsep, or use tabular* with @{\\extracolsep{\\fill}}.",
                )
            )

    # 2. [P0] Check for vertical pipes in colspec or \vline
    if colspec and has_vertical_pipe_in_colspec(colspec):
        report.findings.append(
            Finding(
                severity="P0",
                code="VERTICAL_RULE",
                line=start_line,
                message=f"Tabular column spec '{colspec}' contains vertical rules '|'.",
                suggestion="Remove all '|' from column specification; separate columns with whitespace.",
            )
        )
    for idx, line in enumerate(lines):
        if VLINE_RE.search(line):
            report.findings.append(
                Finding(
                    severity="P0",
                    code="VERTICAL_RULE",
                    line=start_line + idx,
                    message="Table body uses \\vline.",
                    suggestion="Remove \\vline; never use vertical lines in publication tables.",
                )
            )

    # 3. [P0] Check for \hline or \cline
    for idx, line in enumerate(lines):
        if HLINE_RE.search(line):
            report.findings.append(
                Finding(
                    severity="P0",
                    code="HLINE_OR_CLINE",
                    line=start_line + idx,
                    message="Table uses raw \\hline or \\cline instead of booktabs rules.",
                    suggestion="Replace with \\toprule, \\midrule, \\cmidrule(lr){a-b}, and \\bottomrule.",
                )
            )

    # 4. [P1] Check for untrimmed \cmidrule{a-b}
    for idx, line in enumerate(lines):
        for m in CMIDRULE_UNTRIMMED_RE.finditer(line):
            report.findings.append(
                Finding(
                    severity="P1",
                    code="UNTRIMMED_CMIDRULE",
                    line=start_line + idx,
                    message=f"\\cmidrule{{{m.group(1)}}} is not trimmed.",
                    suggestion=f"Use \\cmidrule(lr){{{m.group(1)}}} so adjacent sub-rules do not touch.",
                )
            )

    # 5. [P1] Check for overly cramped \arraystretch (< 0.90)
    for idx, line in enumerate(lines):
        m = ARRAYSTRETCH_RE.search(line)
        if m:
            try:
                val = float(m.group(1))
                if val < 0.90:
                    report.findings.append(
                        Finding(
                            severity="P1",
                            code="CRAMPED_STRETCH",
                            line=start_line + idx,
                            message=f"\\arraystretch is set to {val} (< 0.90), compressing vertical row breathing room.",
                            suggestion="Use \\arraystretch >= 0.95 (or tune \\aboverulesep and \\belowrulesep locally).",
                        )
                    )
            except ValueError:
                pass

    # 6. [P1] Check for repetitive inline `%` (`\%`) in body rows
    in_body = False
    rows_with_percent = 0
    data_rows = 0
    for idx, line in enumerate(lines):
        if "\\midrule" in line:
            in_body = True
            continue
        if "\\bottomrule" in line or "\\end{tabular" in line or "\\end{tblr" in line:
            in_body = False
            continue
        if in_body and "&" in line and "\\\\" in line:
            data_rows += 1
            if r"\%" in line:
                rows_with_percent += 1

    if data_rows >= 3 and rows_with_percent >= max(3, int(0.6 * data_rows)):
        report.findings.append(
            Finding(
                severity="P1",
                code="REPEATED_BODY_UNITS",
                line=start_line,
                message=f"Inline '\\%' repeated in {rows_with_percent}/{data_rows} data rows.",
                suggestion="Hoist '(%%)' into the column header or spanner and omit '\\%' from body cells.",
            )
        )

    # 7. [P2] Check caption placement (caption should precede tabular)
    caption_pos = block_text.find("\\caption")
    tabular_pos = block_text.find("\\begin{tabular")
    if tabular_pos == -1:
        tabular_pos = block_text.find("\\begin{tblr")
    if caption_pos != -1 and tabular_pos != -1 and caption_pos > tabular_pos:
        report.findings.append(
            Finding(
                severity="P2",
                code="CAPTION_BELOW",
                line=start_line,
                message="\\caption is placed below the tabular environment.",
                suggestion="Place table captions above \\toprule (standard IEEE / scientific convention).",
            )
        )

    # 8. [P2] Check outer `@{}` trim on classic tabular/tabular*
    if colspec and "tblr" not in block_text:
        if not colspec.lstrip().startswith("@{") or not colspec.rstrip().endswith("}"):
            report.findings.append(
                Finding(
                    severity="P2",
                    code="MISSING_OUTER_TRIM",
                    line=start_line,
                    message="Tabular column spec lacks outer '@{}' padding trim at left/right edges.",
                    suggestion="Start and end colspec with '@{}' (e.g. '@{} l c c @{}') so text aligns flush with booktabs rule ends.",
                )
            )

    return report


def scan_tex_file(file_path: Path) -> List[TableReport]:
    raw_text = file_path.read_text(encoding="utf-8", errors="replace")
    cleaned = strip_comments(raw_text)
    reports: List[TableReport] = []

    for match in TABLE_ENV_RE.finditer(cleaned):
        env_type = match.group(1)
        full_block = match.group(0)
        start_line = cleaned.count("\n", 0, match.start()) + 1
        end_line = cleaned.count("\n", 0, match.end()) + 1
        reports.append(
            audit_table_block(file_path, env_type, full_block, start_line, end_line)
        )
    return reports


def collect_tex_files(targets: List[str]) -> List[Path]:
    files: List[Path] = []
    for target in targets:
        p = Path(target)
        if p.is_file() and p.suffix.lower() == ".tex":
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.rglob("*.tex")))
    return files


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit LaTeX tables against TeX.SE #112343 / #681429 and IEEEtran standards."
    )
    parser.add_argument(
        "targets",
        nargs="+",
        help="Path(s) to .tex files or directories containing .tex files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 1 if any P0 or P1 findings are detected.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit audit report as JSON.",
    )
    args = parser.parse_args(argv)

    tex_files = collect_tex_files(args.targets)
    all_reports: List[TableReport] = []
    for tf in tex_files:
        all_reports.extend(scan_tex_file(tf))

    if args.json:
        print(json.dumps([asdict(r) for r in all_reports], indent=2))
    else:
        p0_count = sum(
            1 for r in all_reports for f in r.findings if f.severity == "P0"
        )
        p1_count = sum(
            1 for r in all_reports for f in r.findings if f.severity == "P1"
        )
        p2_count = sum(
            1 for r in all_reports for f in r.findings if f.severity == "P2"
        )
        print(
            f"=== LaTeX Table Audit Summary: {len(all_reports)} table(s) across {len(tex_files)} .tex file(s) ==="
        )
        print(f"Findings: P0={p0_count} | P1={p1_count} | P2={p2_count}\n")

        for r in all_reports:
            status = "PASS" if not r.findings else f"FINDINGS ({len(r.findings)})"
            print(
                f"[{status}] {r.file}:{r.start_line}-{r.end_line} "
                f"({r.env_type}, label={r.label}, colspec={{{r.colspec}}})"
            )
            for f in r.findings:
                print(f"  - [{f.severity}] {f.code} (line {f.line}): {f.message}")
                print(f"    Fix: {f.suggestion}")

    has_p0_or_p1 = any(
        f.severity in ("P0", "P1") for r in all_reports for f in r.findings
    )
    if args.strict and has_p0_or_p1:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
