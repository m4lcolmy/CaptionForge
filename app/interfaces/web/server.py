"""FastAPI application exposing CaptionForge to a browser on this machine."""

from __future__ import annotations

import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import unquote

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import ClientDisconnect

from app.adapters import package_updater
from app.adapters.deepgram_adapter import DeepgramAdapter, check_and_save_key
from app.adapters.whisper_adapter import CudaStatus
from app.core.config import Config
from app.core.constants import (
    APP_NAME,
    LOCAL_MEDIA_EXTENSIONS,
    SUPPORTED_OUTPUT_FORMATS,
    VERSION,
)
from app.core.deepgram_key import describe_key, forget_key, key_path, load_key
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
    DeepgramKeyBody,
    InspectRequest,
    JobRequestBody,
    MediaJobRequestBody,
    UpdateRequestBody,
)
from app.interfaces.web.security import LocalOnlyMiddleware
from app.interfaces.web.uploads import UploadStore
from app.models.job import JobStatus
from app.services.factory import create_video_service
from app.utils.local_media import local_media_path

HOST = "127.0.0.1"
STATIC_DIRECTORY = Path(__file__).parent / "static"
# The chosen file's name, percent-encoded: a header carries only ASCII.
FILENAME_HEADER = "X-CaptionForge-Filename"
FINISHED = frozenset({JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED})


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
    deepgram: DeepgramAdapter | None = None,
) -> FastAPI:
    """Build the local application with its job registry and guards."""
    deepgram_adapter = deepgram or DeepgramAdapter()
    registry = JobRegistry(config)
    uploads = UploadStore(config.temp_directory, config.minimum_free_disk_bytes)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        uploads.sweep_stale()
        yield
        registry.shutdown()
        uploads.discard_all()

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
            # What the file chooser lists, beside anything the system calls
            # audio or video.
            "local_media_extensions": list(LOCAL_MEDIA_EXTENSIONS),
            "default_engine": config.transcription_engine,
            "default_model": config.default_whisper_model,
            "default_deepgram_model": config.deepgram_model,
            "default_device": config.whisper_device,
            # Whether a key is there, where from, and its last four characters.
            # The key itself never leaves the server.
            "deepgram": _deepgram_status(),
            **_cuda_status(),
            "output_directory": str(config.default_output_folder.resolve()),
        }

    @application.put("/api/preferences")
    async def write_preferences(body: WebPreferences) -> dict[str, object]:
        """Remember the current choices for the next session."""
        save_preferences(body, preferences_file)
        return resolve(body, config)

    @application.put("/api/deepgram-key")
    def write_deepgram_key(body: DeepgramKeyBody) -> dict[str, object]:
        """Check a pasted key with Deepgram, then keep it where only this user reads.

        A plain ``def``: the check is a network round trip, which must not hold
        up the event loop. Offline, the key is saved unchecked and the reply
        says so, because the person may well be about to go back online.
        """
        _, verdict = check_and_save_key(body.key, deepgram_adapter)
        return {**_deepgram_status(), "checked": verdict is True}

    @application.delete("/api/deepgram-key")
    async def delete_deepgram_key() -> dict[str, object]:
        """Forget the saved key. One set in the environment stays in charge."""
        forget_key()
        return _deepgram_status()

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

    @application.put("/api/uploads", status_code=201, response_model=None)
    async def upload(request: Request) -> dict[str, object] | JSONResponse:
        """Take a copy of a file the page was handed.

        A browser never tells a page where a file lives, so this is the only
        way a chosen or dropped file reaches CaptionForge. The reply is the
        copy's path, which the page then looks up like a pasted one.
        """
        length = request.headers.get("content-length", "")
        try:
            stored = await uploads.receive(
                unquote(request.headers.get(FILENAME_HEADER, "")),
                int(length) if length.isdigit() else None,
                request.stream(),
            )
        except ClientDisconnect:
            get_logger().info("The page stopped sending a file")
            return JSONResponse(
                {
                    "error": "The copy was cancelled.",
                    "code": "UploadCancelled",
                    "retryable": False,
                },
                status_code=400,
            )
        # One copy at a time is enough: earlier ones go, unless a job that has
        # not finished yet still has to read one.
        uploads.prune(keep={stored, *_sources_in_use(registry)})
        return {
            "path": str(stored),
            "name": stored.name,
            "size_bytes": stored.stat().st_size,
        }

    @application.post("/api/jobs", status_code=202)
    async def create_job(body: JobRequestBody) -> dict[str, object]:
        """Queue a caption export that falls back to local transcription."""
        record = registry.submit(
            JobRequest(
                url=body.url,
                language=body.language,
                formats=body.formats,
                engine=body.engine,
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

    @application.get("/api/updates")
    def list_updates(fresh: bool = False) -> dict[str, object]:
        """Newer releases the page can offer. Nothing is installed here.

        A plain ``def``: pip takes seconds, and FastAPI runs these on a worker
        thread instead of holding up every other request meanwhile.
        """
        updater = package_updater.UPDATER
        if not config.check_for_updates or not updater.can_update():
            return {"checked": False, "updates": []}
        return {
            "checked": True,
            "updates": [update.as_dict() for update in updater.check(fresh=fresh)],
        }

    @application.post("/api/updates")
    def install_updates(body: UpdateRequestBody) -> dict[str, object]:
        """Install exactly the packages the person ticked, and nothing else."""
        installed = package_updater.UPDATER.install(body.packages)
        return {"updated": [item.as_dict() for item in installed]}

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


def _sources_in_use(registry: JobRegistry) -> set[Path]:
    """Files that a job which has not finished yet will still read."""
    sources: set[Path] = set()
    for record in registry.recent():
        if record.status in FINISHED:
            continue
        source = local_media_path(record.request.url)
        if source is not None:
            sources.add(source)
    return sources


def _deepgram_status() -> dict[str, object]:
    """What the page may know about the Deepgram key, and where it would live."""
    return {
        **describe_key(load_key()),
        "location": str(key_path().parent),
    }


def _cuda_status() -> dict[str, object]:
    """Separate "no GPU" from "GPU present but its libraries will not load"."""
    status = CudaStatus.probe()
    return {
        "cuda_available": status.available,
        "cuda_device_present": status.device_present,
        "cuda_missing_libraries": list(status.missing_libraries),
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
