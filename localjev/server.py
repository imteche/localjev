"""FastAPI server exposing the Jev-compatible endpoint plus the demo dashboard.

    POST /v1/systemone   -> Jev-shaped typed decisions (Choice/Score/Noul)
    GET  /health         -> LM Studio connectivity + resolved model
    GET  /               -> the support-triage demo dashboard
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import __version__, engine, lmstudio

app = FastAPI(title="LocalJev", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class SystemOneRequest(BaseModel):
    state: str
    questions: Dict[str, Any]
    model: Optional[str] = None  # optional; defaults to LM Studio's loaded model


@app.get("/health")
def health() -> JSONResponse:
    try:
        model = lmstudio.resolve_model()
        return JSONResponse(
            {"status": "ok", "model": model, "lmstudio_url": lmstudio.BASE_URL}
        )
    except lmstudio.LMStudioError as exc:
        return JSONResponse({"status": "unavailable", "detail": str(exc)}, status_code=503)


@app.post("/v1/systemone")
def systemone(req: SystemOneRequest) -> Dict[str, Any]:
    if not req.state or not req.questions:
        raise HTTPException(400, "Both 'state' and 'questions' are required.")
    try:
        return engine.evaluate(req.state, req.questions, model=req.model)
    except lmstudio.LMStudioError as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


def main() -> None:
    import uvicorn

    port = int(os.environ.get("LOCALJEV_PORT", "8000"))
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
