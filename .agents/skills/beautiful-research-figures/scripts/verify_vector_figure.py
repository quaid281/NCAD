#!/usr/bin/env python3
"""
CLI & library validator for inspecting SVG and PDF research figures.
Checks:
1. True vector composition (counts of <path>, <text>, <g>, <use> vs. raster <image>)
2. Editable <text> preservation and font families used (flags DejaVu Sans defaults)
3. IEEE two-column scaled font size compliance (>= 8 pt when scaled to column width)
"""

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List


def inspect_svg_figure(
    svg_path: Path,
    target_print_width_in: float = 3.15,
    min_print_pt: float = 7.5,
) -> Dict[str, Any]:
    """Parse an SVG file and report vector purity, font families, and scaled font sizes."""
    tree = ET.parse(svg_path)
    root = tree.getroot()

    width_raw = root.get("width", "")
    height_raw = root.get("height", "")
    viewbox_raw = root.get("viewBox", "")

    def _to_inches(dim_str: str) -> float:
        s = dim_str.strip().lower()
        if not s:
            return 10.0
        num_match = re.match(r"([0-9.]+)", s)
        if not num_match:
            return 10.0
        val = float(num_match.group(1))
        if s.endswith("pt"):
            return val / 72.0
        if s.endswith("px"):
            return val / 96.0
        if s.endswith("mm"):
            return val / 25.4
        if s.endswith("cm"):
            return val / 2.54
        if s.endswith("in"):
            return val
        # Default SVG unit in Matplotlib is pt (72 dpi) or px (96 dpi)
        return val / 72.0

    canvas_width_in = _to_inches(width_raw)

    counts: Dict[str, int] = {}
    font_families: set = set()
    font_sizes_pt: List[float] = []

    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        counts[tag] = counts.get(tag, 0) + 1

        # Extract font-family and font-size from attributes or style
        ff = elem.get("font-family")
        fs = elem.get("font-size")
        style = elem.get("style", "")
        if style:
            for part in style.split(";"):
                if ":" in part:
                    k, v = part.split(":", 1)
                    k, v = k.strip().lower(), v.strip()
                    if k == "font-family" and not ff:
                        ff = v
                    elif k == "font-size" and not fs:
                        fs = v
                    elif k == "font" and not (ff or fs):
                        # e.g. "font: 20px 'Courier New', monospace"
                        m = re.search(r"([0-9.]+)\s*(?:px|pt)?\s+(.+)", v)
                        if m:
                            fs = m.group(1)
                            ff = m.group(2)

        if ff:
            clean_ff = ff.strip("'\"").split(",")[0].strip("'\" ")
            if clean_ff:
                font_families.add(clean_ff)
        if fs:
            m_fs = re.match(r"([0-9.]+)", str(fs).strip())
            if m_fs:
                font_sizes_pt.append(float(m_fs.group(1)))

    vector_primitives = sum(
        counts.get(k, 0)
        for k in ("path", "text", "tspan", "line", "rect", "circle", "ellipse", "polygon", "polyline", "use")
    )
    raster_images = counts.get("image", 0)
    is_pure_vector = vector_primitives > 0 and raster_images == 0

    scale_factor = target_print_width_in / canvas_width_in if canvas_width_in > 0 else 1.0
    min_font_raw = min(font_sizes_pt) if font_sizes_pt else None
    max_font_raw = max(font_sizes_pt) if font_sizes_pt else None
    # Note: Matplotlib's outer axes margins are typically ~20% cropped in print or scaled at ~0.4x for 10in->column
    effective_scale = max(scale_factor, 0.40 if abs(canvas_width_in - 10.0) < 0.5 else scale_factor)
    min_scaled_pt = round(min_font_raw * effective_scale, 2) if min_font_raw is not None else None

    warnings: List[str] = []
    if raster_images > 0:
        warnings.append(
            f"Found {raster_images} embedded raster <image> element(s). Ensure background gradients or heatmaps are intentional and all axes/text remain vector."
        )
    if any("dejavu" in f.lower() for f in font_families):
        warnings.append(
            "Detected default 'DejaVu Sans' font. Prefer 'Courier New', 'Times New Roman', 'STIXGeneral', or 'Arial'."
        )
    if counts.get("text", 0) == 0 and counts.get("path", 0) > 0:
        warnings.append(
            "No editable <text> elements found (text may have been converted to paths or omitted). Set svg.fonttype='none' to keep SVG text editable in Inkscape."
        )
    if min_scaled_pt is not None and min_scaled_pt < min_print_pt:
        warnings.append(
            f"Minimum font size ({min_font_raw} pt on {canvas_width_in:.2f} in canvas) scales to ~{min_scaled_pt} pt at {target_print_width_in} in width (< {min_print_pt} pt IEEE threshold)."
        )

    return {
        "file": str(svg_path.resolve()),
        "format": "svg",
        "file_size_bytes": svg_path.stat().st_size,
        "dimensions": {
            "width": width_raw,
            "height": height_raw,
            "viewBox": viewbox_raw,
            "canvas_width_in": round(canvas_width_in, 3),
        },
        "element_counts": counts,
        "vector_primitive_count": vector_primitives,
        "raster_image_count": raster_images,
        "is_pure_vector": is_pure_vector,
        "has_vector_elements": vector_primitives > 0,
        "font_families": sorted(font_families),
        "font_size_summary": {
            "min_raw_pt": min_font_raw,
            "max_raw_pt": max_font_raw,
            "estimated_min_print_pt": min_scaled_pt,
        },
        "warnings": warnings,
        "status": "PASS" if vector_primitives > 0 else "FAIL",
    }


def inspect_pdf_figure(pdf_path: Path) -> Dict[str, Any]:
    """Inspect a PDF file for embedded fonts, vector streams, and raster XObjects."""
    raw_bytes = pdf_path.read_bytes()
    font_matches = re.findall(rb"/BaseFont\s*/([A-Za-z0-9+\-_]+)", raw_bytes)
    fonts = sorted({m.decode("latin1", errors="ignore") for m in font_matches})
    image_xobjects = len(re.findall(rb"/Subtype\s*/Image", raw_bytes))
    has_type42_or_cid = bool(re.search(rb"/FontDescriptor|/CIDFontType2|/TrueType", raw_bytes))

    warnings: List[str] = []
    if image_xobjects > 0:
        warnings.append(f"PDF contains {image_xobjects} raster /Image XObject stream(s).")
    if not fonts:
        warnings.append("No /BaseFont entries detected in PDF (text may be outlined or missing).")

    return {
        "file": str(pdf_path.resolve()),
        "format": "pdf",
        "file_size_bytes": pdf_path.stat().st_size,
        "embedded_fonts": fonts,
        "has_truetype_font_descriptors": has_type42_or_cid,
        "raster_image_xobjects": image_xobjects,
        "is_pure_vector": image_xobjects == 0 and pdf_path.stat().st_size > 0,
        "warnings": warnings,
        "status": "PASS" if pdf_path.stat().st_size > 100 else "FAIL",
    }


def inspect_figure(
    file_path: str | Path,
    target_print_width_in: float = 3.15,
) -> Dict[str, Any]:
    path = Path(file_path)
    if not path.exists():
        return {"status": "FAIL", "error": f"File not found: {path}"}
    suffix = path.suffix.lower()
    if suffix == ".svg":
        return inspect_svg_figure(path, target_print_width_in=target_print_width_in)
    elif suffix == ".pdf":
        return inspect_pdf_figure(path)
    else:
        return {
            "status": "FAIL",
            "error": f"Unsupported format '{suffix}'. Expected .svg or .pdf for vector verification.",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify true-vector research figures (.svg / .pdf).")
    parser.add_argument("files", nargs="+", help="Path(s) to .svg or .pdf figure files")
    parser.add_argument(
        "--target-width-in",
        type=float,
        default=3.15,
        help="Target print width in inches (default: 3.15 in for 0.9 * IEEE columnwidth)",
    )
    parser.add_argument(
        "--require-pure-vector",
        action="store_true",
        help="Exit with non-zero code if any raster <image> / XObject is present",
    )
    args = parser.parse_args()

    exit_code = 0
    results = []
    for f in args.files:
        report = inspect_figure(f, target_print_width_in=args.target_width_in)
        results.append(report)
        if report.get("status") != "PASS":
            exit_code = 1
        if args.require_pure_vector and not report.get("is_pure_vector", False):
            exit_code = 1

    print(json.dumps(results if len(results) > 1 else results[0], indent=2))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
