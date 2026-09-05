"""FastAPI application exposing CaptionForge to a browser on this machine."""

from __future__ import annotations

import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.adapters.whisper_adapter import WhisperAdapter
from app.core.config import Config
from app.core.constants import APP_NAME, SUPPORTED_OUTPUT_FORMATS, VERSION
from app.core.exceptions import CaptionForgeError
from app.core.logging_config import get_logger
from app.interfaces.web import errors
from app.interfaces.web.jobs import JobRegistry, JobRequest, MediaJobRequest
from app.interfaces.web.preferences import (
    WebPreferences,
    load_preferences,
    resolve,
    save_preferences,
)
from app.interfaces.web.schemas import (
    InspectRequest,
    JobRequestBody,
    MediaJobRequestBody,
)
from app.interfaces.web.security import LocalOnlyMiddleware
from app.services.factory import create_video_service

HOST = "127.0.0.1"
STATIC_DIRECTORY = Path(__file__).parent / "static"


def find_free_port() -> int:
    """Ask the operating system for an unused local port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((HOST, 0))
        return int(probe.getsockname()[1])


def create_app(
    config: Config,
    *,
    token: str,
    port: int,
    preferences_file: Path | None = None,
) -> FastAPI:
    """Build the local application with its job registry and guards."""
    registry = JobRegistry(config)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        registry.shutdown()

    application = FastAPI(
        title=f"{APP_NAME} local interface",
        version=VERSION,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    application.add_middleware(LocalOnlyMiddleware, token=token, port=port)
    application.state.registry = registry
    application.state.config = config

    @application.exception_handler(CaptionForgeError)
    async def handle_application_error(
        _: Request, exc: CaptionForgeError
    ) -> JSONResponse:
        """Return the short message and keep the technical cause in the log."""
        get_logger().error(
            "Web request failed user_message={} technical_cause={}",
            exc.message,
            exc.details or type(exc).__name__,
        )
        return JSONResponse(errors.body_for(exc), status_code=errors.status_for(exc))

    @application.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        """Serve the single page."""
        return HTMLResponse((STATIC_DIRECTORY / "index.html").read_text("utf-8"))

    @application.get("/api/health")
    async def health() -> dict[str, object]:
        """Report the running version and the defaults the page starts from."""
        return {
            "app": APP_NAME,
            "version": VERSION,
            "default_language": config.default_language,
            "default_formats": list(config.default_output_formats),
            # What the page starts with: last session's choices where they
            # exist, otherwise one configured format. The CLI is unaffected.
            "preferences": resolve(load_preferences(preferences_file), config),
            "supported_formats": sorted(SUPPORTED_OUTPUT_FORMATS),
            "default_model": config.default_whisper_model,
            "default_device": config.whisper_device,
            **_cuda_status(),
            "output_directory": str(config.default_output_folder.resolve()),
        }

    @application.put("/api/preferences")
    async def write_preferences(body: WebPreferences) -> dict[str, object]:
        """Remember the current choices for the next session."""
        save_preferences(body, preferences_file)
        return resolve(body, config)

    @application.post("/api/inspect")
    async def inspect(body: InspectRequest) -> dict[str, object]:
        """Return metadata, caption tracks, and downloadable qualities."""
        language = body.language or config.default_language
        result = create_video_service(config).inspect_all(
            body.url, language, allow_translated=body.allow_translated
        )
        # One lookup answers both questions the page asks, so choosing a
        # quality never costs a second wait on YouTube.
        return {
            **result.discovery.model_dump(mode="json"),
            "media": result.media.model_dump(mode="json"),
        }

    @application.post("/api/jobs", status_code=202)
    async def create_job(body: JobRequestBody) -> dict[str, object]:
        """Queue a caption export that falls back to local transcription."""
        record = registry.submit(
            JobRequest(
                url=body.url,
                language=body.language,
                formats=body.formats,
                model=body.model,
                device=body.device,
                compute_type=body.compute_type,
                prompt=body.prompt,
                force=body.force,
                overwrite=body.overwrite,
                keep_audio=body.keep_audio,
                timestamped_txt=body.timestamped_txt,
                postprocess=body.postprocess,
                allow_translated=body.allow_translated,
            )
        )
        return record.snapshot()

    @application.post("/api/media", status_code=202)
    async def create_media_job(body: MediaJobRequestBody) -> dict[str, object]:
        """Queue a whole-file download of the video or of its audio alone."""
        record = registry.submit_media(
            MediaJobRequest(
                url=body.url, quality=body.quality, overwrite=body.overwrite
            )
        )
        return record.snapshot()

    @application.get("/api/jobs/{job_id}")
    async def read_job(job_id: str) -> JSONResponse:
        """Report the current stage, percentage, and produced files."""
        record = registry.get(job_id)
        if record is None:
            return _unknown_job()
        return JSONResponse(record.snapshot())

    @application.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> JSONResponse:
        """Ask a running job to stop at its next checkpoint."""
        if not registry.cancel(job_id):
            return _unknown_job()
        record = registry.get(job_id)
        assert record is not None
        return JSONResponse(record.snapshot())

    @application.get("/api/jobs/{job_id}/files/{name}", response_model=None)
    async def download(job_id: str, name: str) -> FileResponse | JSONResponse:
        """Stream one produced file, matched by name against what the job wrote."""
        record = registry.get(job_id)
        if record is None:
            return _unknown_job()
        produced = record.file_named(name)
        if produced is None or not produced.path.is_file():
            return JSONResponse(
                {
                    "error": "That file is no longer available.",
                    "code": "FileNotFound",
                    "retryable": False,
                },
                status_code=404,
            )
        return FileResponse(
            produced.path, filename=produced.name, media_type="application/octet-stream"
        )

    application.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")
    return application


def _cuda_status() -> dict[str, object]:
    """Separate "no GPU" from "GPU present but its libraries will not load"."""
    adapter = WhisperAdapter()
    missing = WhisperAdapter.missing_cuda_libraries()
    return {
        "cuda_available": adapter.cuda_available(),
        "cuda_device_present": WhisperAdapter.cuda_device_present(),
        "cuda_missing_libraries": list(missing),
    }


def _unknown_job() -> JSONResponse:
    """Response for a job identifier the registry no longer knows."""
    return JSONResponse(
        {
            "error": "That job is no longer available.",
            "code": "JobNotFound",
            "retryable": False,
        },
        status_code=404,
    )
