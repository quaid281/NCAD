"""Export and verify the IEEE TNNLS Figure 1 architecture diagram (Laminar-JEPA & Laminar-JEPA-Lite).

Uses the publication-grade dual-track Stage 1-4 figure (`opentropy_jepa_architecture.png`)
and exports the lossless print-ready PDF (`opentropy_jepa_architecture.pdf`) at 300 DPI so
re-running figure scripts never reverts Figure 1 to an earlier matplotlib box sketch.
"""
from __future__ import annotations

from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "NCAD_CS" / "figures"
PNG_PATH = FIG_DIR / "opentropy_jepa_architecture.png"
PDF_PATH = FIG_DIR / "opentropy_jepa_architecture.pdf"


def export_architecture_pdf(dpi: float = 300.0) -> None:
    if not PNG_PATH.exists():
        raise FileNotFoundError(f"Missing Figure 1 master PNG: {PNG_PATH}")
    im = Image.open(PNG_PATH).convert("RGB")
    im.save(PDF_PATH, "PDF", resolution=dpi)
    print(f"[OK] Exported {PDF_PATH} ({im.size[0]}x{im.size[1]} @ {dpi:.0f} DPI)")


if __name__ == "__main__":
    export_architecture_pdf()
