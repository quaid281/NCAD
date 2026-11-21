"""FastAPI server for the local GPU-powered Laminar-JEPA-Lite interactive explorer."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from app.gpu_pipeline import (
    PROJECT_ROOT,
    get_device_info,
    list_available_streams,
    run_laminar_lite_on_gpu,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
SITE_EXPORT_DIR = PROJECT_ROOT / "site"

app = FastAPI(title="Laminar-JEPA-Lite GPU Explorer", version="1.0.0")


class RunRequest(BaseModel):
    dataset: str = Field(..., min_length=1)
    stream: str = Field(..., min_length=1)
    epochs: int = Field(default=10, ge=2, le=25)
    seed: int = Field(default=42, ge=0, le=99999)


@app.get("/", response_class=HTMLResponse)
def serve_index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Frontend index.html missing.")
    return FileResponse(index_path)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", **get_device_info()}


@app.get("/api/streams")
def api_streams() -> dict[str, Any]:
    return list_available_streams()


@app.post("/api/run")
def api_run(req: RunRequest) -> dict[str, Any]:
    try:
        return run_laminar_lite_on_gpu(
            dataset=req.dataset,
            stream=req.stream,
            epochs=req.epochs,
            seed=req.seed,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/export-cloudflare")
def api_export_cloudflare(req: RunRequest) -> dict[str, Any]:
    """Export a self-contained static site/ folder with real GPU-computed data for Cloudflare Pages."""
    payload = run_laminar_lite_on_gpu(
        dataset=req.dataset,
        stream=req.stream,
        epochs=req.epochs,
        seed=req.seed,
    )
    SITE_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(STATIC_DIR / "index.html", SITE_EXPORT_DIR / "index.html")

    # Atomic JSON write per SOP 3
    target_json = SITE_EXPORT_DIR / "demo_trace.json"
    tmp_json = SITE_EXPORT_DIR / f"demo_trace.{os.getpid()}.tmp"
    with open(tmp_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_json, target_json)

    size_kb = round(target_json.stat().st_size / 1024.0, 1)
    return {
        "status": "exported",
        "site_dir": str(SITE_EXPORT_DIR),
        "trace_file": str(target_json),
        "size_kb": size_kb,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.server:app", host="127.0.0.1", port=8000, reload=False)
