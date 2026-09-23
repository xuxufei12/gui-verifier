import io
import json
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from .schema import VerifyRequest

MAX_BYTES = 12 * 1024 * 1024
MAX_PIXELS = 16_000_000


def load_image(content):
    if len(content) > MAX_BYTES:
        raise ValueError("Screenshot exceeds 12 MiB")
    try:
        with Image.open(io.BytesIO(content)) as img:
            if img.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("Use PNG, JPEG, or WebP screenshots")
            if img.width * img.height > MAX_PIXELS:
                raise ValueError("Screenshot exceeds 16 megapixels")
            return img.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Invalid or oversized screenshot") from exc


def create_app(verifier=None):
    app = FastAPI(title="GUI Verifier", version="0.1.0")
    app.state.verifier = verifier
    app.state.init_lock = threading.Lock()

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": app.state.verifier is not None,
                "model": os.getenv("GUI_MODEL", "Qwen/Qwen3.5-0.8B")}

    @app.post("/v1/verify")
    def verify(image: UploadFile = File(...), request: str = Form(...)):
        started = time.perf_counter()
        try:
            task = VerifyRequest.model_validate_json(request)
            screenshot = load_image(image.file.read(MAX_BYTES + 1))
        except (ValidationError, ValueError) as exc:
            raise HTTPException(422, detail=str(exc)) from exc
        with app.state.init_lock:
            if app.state.verifier is None:
                from .backend import QwenVerifier
                if os.getenv("GUI_BACKEND") == "fast":
                    from .fast_backend import FastQwenVerifier
                    QwenVerifier = FastQwenVerifier
                app.state.verifier = QwenVerifier(
                    model=os.getenv("GUI_MODEL", "Qwen/Qwen3.5-0.8B"),
                    device=os.getenv("GUI_DEVICE", "auto"),
                    batch_size=int(os.getenv("GUI_BATCH_SIZE", "1")),
                    adapter=os.getenv("GUI_ADAPTER") or None,
                    max_pixels=int(os.getenv("GUI_MAX_PIXELS", "262144")),
                )
        try:
            result = app.state.verifier.verify(screenshot, task)
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc)) from exc
        result.timing_ms["server_total"] = (time.perf_counter() - started) * 1000
        return result

    return app


app = create_app()
