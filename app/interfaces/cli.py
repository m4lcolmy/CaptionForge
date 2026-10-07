"""Typer command-line interface for CaptionForge."""

import os
import platform
import shutil
import sys
import webbrowser
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from app.adapters.deepgram_adapter import DeepgramAdapter, check_and_save_key
from app.adapters.ffmpeg_adapter import FFmpegAdapter
from app.adapters.package_updater import UPDATER, AvailableUpdate, InstalledUpdate
from app.adapters.whisper_adapter import WhisperAdapter
from app.adapters.ytdlp_adapter import YtDlpAdapter
from app.core.config import Config
from app.core.constants import APP_NAME, VERSION, ExitCode
from app.core.deepgram_key import forget_key, key_path, load_key
from app.core.exceptions import (
    CaptionForgeError,
    ConfigurationError,
    ExtractorRefusedError,
    InvalidYouTubeUrlError,
    LiveStreamNotSupportedError,
    LocalMediaError,
    MetadataRetrievalError,
    SubtitleDiscoveryError,
    UnsupportedYouTubeUrlError,
    VideoUnavailableError,
)
from app.core.logging_config import configure_logging, get_logger
from app.models.subtitle import (
    RawSubtitle,
    SubtitleDiscoveryResult,
    SubtitleSourceType,
    SubtitleTrack,
)
from app.models.video import VideoMetadata
from app.services.audio_service import AudioService
from app.services.export_service import ExportService
from app.services.factory import create_media_service
from app.services.subtitle_service import SubtitleService
from app.services.transcription_plan import TranscriptionPlan
from app.services.transcription_service import TranscriptionService
from app.services.video_service import VideoService

app = typer.Typer(
    name="captionforge",
    help="CaptionForge subtitle workflow foundation.",
    no_args_is_help=False,
    invoke_without_command=True,
    add_completion=False,
)
config_app = typer.Typer(help="Show or update persistent user configuration.")
app.add_typer(config_app, name="config")
console = Console()
error_console = Console(stderr=True)


def _run_with_config(action: Callable[[Config], None]) -> None:
    """Load configuration, initialize logging, and run a CLI action."""
    offered = False
    try:
        config = Config.load()
        configure_logging(config)
        try:
            action(config)
        except ExtractorRefusedError:
            offered = _can_offer_updates(config)
            if not offered or not _update_after_refusal():
                raise
            # The approved yt-dlp is loaded now, so the command gets one more go.
            action(config)
    except CaptionForgeError as exc:
        get_logger().error(
            "Command failed user_message={} technical_cause={}",
            exc.message,
            exc.details or type(exc).__name__,
        )
        error_console.print(f"[bold red]Error:[/bold red] {exc.message}")
        if isinstance(exc, ExtractorRefusedError) and not offered:
            error_console.print(
                "Run 'captionforge update' to check for a newer yt-dlp."
            )
        raise typer.Exit(code=_exit_code_for(exc)) from exc
    except (KeyboardInterrupt, typer.Abort) as exc:
        # Abort is Ctrl+D at one of the update questions: a cancel, like Ctrl+C.
        get_logger().warning("Command cancelled by user")
        error_console.print(
            "[yellow]Cancelled:[/yellow] No incomplete output was kept."
        )
        raise typer.Exit(code=130) from exc
    except Exception as exc:
        get_logger().exception("Unexpected command failure technical_cause={}", exc)
        error_console.print(
            "[bold red]Error:[/bold red] CaptionForge could not complete the command. "
            "See the log for technical details."
        )
        raise typer.Exit(code=ExitCode.FAILURE) from exc


@app.callback()
def root(ctx: typer.Context) -> None:
    """CaptionForge metadata and subtitle discovery commands."""
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


@app.command()
def version() -> None:
    """Print the installed CaptionForge version."""
    console.print(f"{APP_NAME} {VERSION}")


@config_app.command(name="show")
def show_config() -> None:
    """Display the effective, validated configuration."""

    def render(config: Config) -> None:
        table = Table(title="CaptionForge Configuration", show_header=False)
        table.add_column("Setting", style="cyan")
        table.add_column("Value")
        for name, value in config.model_dump(mode="json").items():
            display = ", ".join(value) if isinstance(value, list) else str(value)
            table.add_row(name, display)
        console.print(table)

    _run_with_config(render)


@config_app.command(name="set")
def set_config(
    key: str = typer.Argument(..., metavar="KEY"),
    value: str = typer.Argument(..., metavar="VALUE"),
) -> None:
    """Validate and persist one configuration setting."""

    def update(config: Config) -> None:
        normalized_key = key.strip().lower()
        parsed = Config.parse_setting(normalized_key, value)
        data = config.model_dump(mode="json")
        data[normalized_key] = parsed
        updated = Config.model_validate(data)
        path = updated.persist()
        console.print(f"Saved {normalized_key} in {path}", markup=False)

    _run_with_config(update)


@config_app.command(name="reset")
def reset_config() -> None:
    """Remove persisted settings and return to defaults plus environment overrides."""
    try:
        path = Config.user_config_path()
        path.unlink(missing_ok=True)
        console.print(f"Reset configuration: {path}", markup=False)
    except OSError as exc:
        error = ConfigurationError(
            "The user configuration could not be reset.", details=str(exc)
        )
        error_console.print(f"[bold red]Error:[/bold red] {error.message}")
        raise typer.Exit(code=ExitCode.FAILURE) from exc


@app.command()
def doctor() -> None:
    """Report local environment and dependency diagnostics without network access."""

    def render(config: Config) -> None:
        output_folder = config.default_output_folder
        writable = _is_writable_directory(output_folder)
        temp_writable = _is_writable_directory(config.temp_directory)
        table = Table(title="CaptionForge Doctor")
        table.add_column("Check", style="cyan")
        table.add_column("Result")
        table.add_row("Python version", platform.python_version())
        table.add_row("Operating system", platform.platform())
        table.add_row("Working directory", str(Path.cwd()))
        table.add_row(
            "Writable output folder",
            f"{'Yes' if writable else 'No'} ({output_folder})",
        )
        ffmpeg_path = shutil.which(str(config.ffmpeg_executable))
        table.add_row("FFmpeg installed", "Yes" if ffmpeg_path else "No")
        if ffmpeg_path:
            try:
                ffmpeg_version = FFmpegAdapter(config.ffmpeg_executable).version()
            except CaptionForgeError:
                ffmpeg_version = "Unable to read version"
        else:
            ffmpeg_version = "Not available"
        table.add_row("FFmpeg version", ffmpeg_version)
        ffprobe = FFmpegAdapter(config.ffmpeg_executable).probe_executable
        table.add_row(
            "FFprobe installed",
            "Yes" if shutil.which(ffprobe) else "No - needed to read files",
        )
        try:
            ytdlp_version = package_version("yt-dlp")
            table.add_row("yt-dlp installed", "Yes")
            table.add_row("yt-dlp version", ytdlp_version)
        except PackageNotFoundError:
            table.add_row("yt-dlp installed", "No")
            table.add_row("yt-dlp version", "Not available")
        table.add_row("Package updates", _updates_status(config))
        try:
            whisper_version = package_version("faster-whisper")
            table.add_row("faster-whisper installed", "Yes")
            table.add_row("faster-whisper version", whisper_version)
        except PackageNotFoundError:
            table.add_row("faster-whisper installed", "No")
            table.add_row("faster-whisper version", "Not available")
        try:
            docx_version = package_version("python-docx")
            table.add_row("python-docx installed", "Yes")
            table.add_row("python-docx version", docx_version)
        except PackageNotFoundError:
            table.add_row("python-docx installed", "No")
            table.add_row("python-docx version", "Not available")
        table.add_row("JavaScript runtime", _javascript_runtime_status())
        whisper = WhisperAdapter()
        cuda_available = whisper.cuda_available()
        recommended_device = "cuda" if cuda_available else "cpu"
        recommended_compute = WhisperAdapter.select_compute_type(
            "auto", recommended_device
        )
        missing_cuda = WhisperAdapter.missing_cuda_libraries()
        gpu_present = WhisperAdapter.cuda_device_present()
        if cuda_available:
            cuda_status = "Yes"
        elif gpu_present and missing_cuda:
            # A GPU the app cannot use is worth naming, not hiding behind "No".
            cuda_status = f"No - {' and '.join(missing_cuda)} could not be loaded"
        else:
            cuda_status = "No"
        table.add_row("CUDA available", cuda_status)
        table.add_row(
            "Detected GPU",
            _detected_gpu_name() if gpu_present else "Not available",
        )
        table.add_row("Recommended device", recommended_device)
        table.add_row("Recommended compute type", recommended_compute)
        table.add_row("GPU memory", _gpu_memory_status())
        table.add_row("Planned configuration", _planned_configuration(config))
        table.add_row("Transcription engine", config.transcription_engine)
        table.add_row("Deepgram key", _deepgram_key_status())
        table.add_row(
            "Writable temporary folder",
            f"{'Yes' if temp_writable else 'No'} ({config.temp_directory})",
        )
        console.print(table)
        get_logger().info("Doctor diagnostics completed")

    _run_with_config(render)


@app.command()
def inspect(
    video_url: str = typer.Argument(
        ...,
        metavar="SOURCE",
        help="A YouTube video URL, or a video or audio file on this computer.",
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        "-l",
        help="Preferred subtitle language (defaults to configuration).",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit a stable JSON document only.",
    ),
    allow_translated: bool = typer.Option(
        False,
        "--allow-translated",
        help="Also consider YouTube machine-translated caption tracks.",
    ),
) -> None:
    """Inspect video metadata and caption availability without downloading files."""

    def run(config: Config) -> None:
        preferred_language = language or config.default_language
        result = _create_video_service(config).inspect(
            video_url, preferred_language, allow_translated=allow_translated
        )
        if json_output:
            console.print(result.model_dump_json(indent=2), markup=False)
        else:
            _render_inspection(result)

    _run_with_config(run)


@app.command()
def transcribe(
    video_url: str = typer.Argument(
        ...,
        metavar="SOURCE",
        help="A YouTube video URL, or a video or audio file on this computer.",
    ),
    language: str | None = typer.Option(
        None, "--language", "-l", help="Caption/transcription language."
    ),
    engine: str | None = typer.Option(
        None,
        "--engine",
        help="whisper (on this computer, the default) or deepgram (uploads "
        "the audio to Deepgram).",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Whisper model name or folder, or a Deepgram model such as nova-3.",
    ),
    device: str | None = typer.Option(
        None, "--device", help="Device: auto, cpu, or cuda."
    ),
    compute_type: str | None = typer.Option(
        None, "--compute-type", help="Compute type such as auto, int8, or float16."
    ),
    prompt: str | None = typer.Option(
        None,
        "--prompt",
        help="Names and terms in the video, comma-separated. Whisper reads them "
        "as a hint; Deepgram takes them as key terms.",
    ),
    formats: Annotated[
        list[str] | None,
        typer.Option(
            "--format",
            "-f",
            help="Output format; repeat for srt, vtt, txt, json, or docx.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Directory for generated files."),
    ] = None,
    keep_audio: bool = typer.Option(
        False, "--keep-audio", help="Keep the prepared audio and job workspace."
    ),
    force: bool = typer.Option(
        False, "--force", help="Transcribe even when suitable captions exist."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Replace existing output files."
    ),
    timestamped_txt: bool = typer.Option(
        False, "--timestamped-txt", help="Include timestamps in TXT output."
    ),
    no_postprocess: bool = typer.Option(
        False,
        "--no-postprocess",
        help="Export source segments without Phase 6 cleanup.",
    ),
    allow_translated: bool = typer.Option(
        False,
        "--allow-translated",
        help="Also consider YouTube machine-translated caption tracks.",
    ),
) -> None:
    """Export existing captions, or transcribe with Whisper or Deepgram.

    A file on this computer has no captions to reuse, so it is always transcribed.
    Whisper runs here; --engine deepgram uploads a compressed copy of the audio.
    """

    def run(config: Config) -> None:
        adapter = YtDlpAdapter()
        subtitles = SubtitleService(config)
        video_service = VideoService(adapter, subtitles, config)
        service = TranscriptionService(
            video_service,
            adapter,
            subtitles,
            AudioService(
                video_service,
                adapter,
                FFmpegAdapter(config.ffmpeg_executable),
                config,
            ),
            WhisperAdapter(),
            ExportService(),
            config,
            deepgram=DeepgramAdapter(),
        )

        def report(message: str, percent: float | None) -> None:
            suffix = f" ({percent:.0f}%)" if percent is not None else ""
            console.print(f"[cyan]{message}{suffix}[/cyan]")

        result = service.process(
            video_url,
            language=language,
            engine=engine,
            model_name=model,
            device=device,
            compute_type=compute_type,
            formats=formats,
            output_directory=output,
            keep_audio=keep_audio,
            force=force,
            overwrite=overwrite,
            timestamped_txt=timestamped_txt,
            postprocess=not no_postprocess,
            allow_translated=allow_translated,
            initial_prompt=prompt,
            progress=report,
        )
        info = result.transcription
        if result.used_existing_captions or info is None:
            source = "existing captions"
        elif info.engine == "deepgram":
            source = "Deepgram transcription"
        else:
            source = "local Whisper transcription"
        console.print(f"[bold green]Completed using {source}.[/bold green]")
        if info is not None:
            probability = (
                f", probability {info.language_probability:.1%}"
                if info.language_probability is not None
                else ""
            )
            # Device and compute type describe this computer, so a Deepgram
            # result has neither worth printing.
            hardware = (
                ""
                if info.engine == "deepgram"
                else f"; device: {info.device}; compute type: {info.compute_type}"
            )
            console.print(
                f"Language: {info.detected_language}{probability}; "
                f"model: {info.model_name}{hardware}"
            )
        if result.prepared_audio_path is not None:
            console.print(
                f"Prepared audio kept at: {result.prepared_audio_path.resolve()}",
                markup=False,
            )
        for path in result.paths:
            console.print(str(path.resolve()), markup=False)

    _run_with_config(run)


@app.command(name="deepgram-key")
def deepgram_key(
    forget: bool = typer.Option(
        False, "--forget", help="Remove the saved key instead of saving one."
    ),
) -> None:
    """Save the Deepgram API key that --engine deepgram uses, or remove it.

    The key is checked with Deepgram, then kept in a file only you can read.
    CAPTIONFORGE_DEEPGRAM_API_KEY or DEEPGRAM_API_KEY, when set, take priority.
    """

    def run(_config: Config) -> None:
        if forget:
            removed = forget_key()
            console.print(
                "Removed the saved Deepgram key."
                if removed
                else "No Deepgram key was saved."
            )
            remaining = load_key()
            if remaining is not None and remaining.source == "environment":
                console.print(
                    f"A key ({remaining.hint}) is still set in the environment."
                )
            return
        saved, verdict = check_and_save_key(
            typer.prompt("Deepgram API key", hide_input=True)
        )
        console.print(
            f"[bold green]Saved[/bold green] the Deepgram key {saved.hint} in "
            f"{escape(str(key_path()))}"
        )
        if verdict is None:
            console.print(
                "[yellow]Deepgram could not be reached to check it now; it is "
                "checked again the first time it is used.[/yellow]"
            )

    _run_with_config(run)


@app.command(name="prepare-audio")
def prepare_audio(
    video_url: str = typer.Argument(
        ...,
        metavar="SOURCE",
        help="A YouTube video URL, or a video or audio file on this computer.",
    ),
    language: str | None = typer.Option(
        None, "--language", "-l", help="Preferred caption language."
    ),
    output_temp: Annotated[
        Path | None,
        typer.Option("--output-temp", help="Temporary workspace root."),
    ] = None,
    keep_temp: bool = typer.Option(
        False, "--keep-temp", help="Preserve downloaded and converted job files."
    ),
    force: bool = typer.Option(
        False, "--force", help="Prepare audio even when matching captions exist."
    ),
) -> None:
    """Prepare mono 16 kHz PCM WAV audio; transcription arrives in Phase 5."""

    def run(config: Config) -> None:
        adapter = YtDlpAdapter()
        service = AudioService(
            VideoService(adapter, SubtitleService(config), config),
            adapter,
            FFmpegAdapter(config.ffmpeg_executable),
            config,
        )

        def report(message: str, percent: float | None) -> None:
            suffix = f" ({percent:.0f}%)" if percent is not None else ""
            console.print(f"[cyan]{message}{suffix}[/cyan]")

        path = service.prepare(
            video_url,
            language or config.default_language,
            temporary_directory=output_temp,
            keep_temp=True if keep_temp else None,
            force=force,
            progress=report,
        )
        console.print(f"[bold green]Prepared audio:[/bold green] {path}", markup=False)
        console.print("Transcription will be implemented in Phase 5.")

    _run_with_config(run)


@app.command()
def extract(
    video_url: str = typer.Argument(
        ..., metavar="VIDEO_URL", help="An individual YouTube video URL."
    ),
    language: str | None = typer.Option(
        None, "--language", "-l", help="Preferred caption language."
    ),
    formats: Annotated[
        list[str] | None,
        typer.Option(
            "--format",
            "-f",
            help=(
                "Output format; repeat for multiple formats "
                "(srt, vtt, txt, json, docx)."
            ),
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Directory for generated files."),
    ] = None,
    timestamped_txt: bool = typer.Option(
        False, "--timestamped-txt", help="Include start timestamps in TXT output."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Replace existing output files."
    ),
    no_postprocess: bool = typer.Option(
        False,
        "--no-postprocess",
        help="Export source captions without Phase 6 cleanup.",
    ),
    allow_translated: bool = typer.Option(
        False,
        "--allow-translated",
        help="Also consider YouTube machine-translated caption tracks.",
    ),
) -> None:
    """Download and export an existing YouTube caption track without media."""

    def run(config: Config) -> None:
        preferred = language or config.default_language
        requested_formats = formats or list(config.default_output_formats)
        output_directory = output or config.default_output_folder
        adapter = YtDlpAdapter()
        subtitle_service = SubtitleService(config)
        discovery = VideoService(adapter, subtitle_service, config).inspect(
            video_url, preferred, allow_translated=allow_translated
        )
        video_id = discovery.video.video_id
        if video_id is None:
            raise SubtitleDiscoveryError(
                "A file on this computer has no caption track to export. Run "
                "'captionforge transcribe' to transcribe it locally."
            )
        track = discovery.selected_track
        if track is None:
            translated = SubtitleService.translated_matches(discovery)
            if translated:
                raise SubtitleDiscoveryError(
                    f"Only machine-translated '{discovery.preferred_language}' "
                    "captions are available, which are translations of an "
                    "automatic transcription. Run 'captionforge transcribe' for "
                    "better quality, or pass --allow-translated to export them."
                )
            raise SubtitleDiscoveryError(
                f"No captions matching '{discovery.preferred_language}' were found. "
                "Run 'captionforge transcribe' to transcribe the audio locally."
            )
        segments = subtitle_service.retrieve_and_parse(
            adapter,
            video_id,
            track,
            postprocess=not no_postprocess,
        )
        paths = ExportService().export(
            discovery.video,
            track,
            segments,
            requested_formats,
            output_directory,
            timestamped_txt=timestamped_txt,
            overwrite=overwrite,
        )
        console.print(
            f"[bold green]Success:[/bold green] exported {len(paths)} caption file(s)."
        )
        for path in paths:
            console.print(str(path.resolve()), markup=False)

    _run_with_config(run)


@app.command()
def download(
    video_url: str = typer.Argument(
        ..., metavar="VIDEO_URL", help="An individual YouTube video URL."
    ),
    quality: str = typer.Option(
        "best",
        "--quality",
        "-q",
        help="mp3 for audio only, best, or a height such as 1080.",
    ),
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Directory to write the file into."),
    ] = None,
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Replace an existing file instead of numbering it."
    ),
    show: bool = typer.Option(
        False, "--list", help="Print what this video offers and download nothing."
    ),
) -> None:
    """Download the whole video as MP4, or its audio alone as MP3."""

    def run(config: Config) -> None:
        service = create_media_service(config)
        if show:
            video, options = service.options(video_url)
            table = Table(title=video.title, show_header=True)
            table.add_column("Quality", style="cyan")
            table.add_column("File")
            table.add_column("Approximate size", justify="right")
            for variant in options.variants:
                table.add_row(
                    variant.key,
                    f"{variant.label} · {variant.extension}",
                    _format_bytes(variant.estimated_bytes),
                )
            console.print(table)
            return

        def report(message: str, percent: float | None) -> None:
            suffix = f" ({percent:.0f}%)" if percent is not None else ""
            console.print(f"[cyan]{message}{suffix}[/cyan]")

        result = service.download(
            video_url,
            quality,
            output_directory=output,
            overwrite=overwrite,
            progress=report,
        )
        # Escape rather than disable markup: a video title can contain square
        # brackets, and the label beside it should still be green.
        console.print(
            f"[bold green]Saved {result.variant.label}:[/bold green] "
            f"{escape(str(result.path))}"
        )

    _run_with_config(run)


@app.command()
def clean(
    input_file: Annotated[
        Path,
        typer.Argument(metavar="INPUT_FILE", help="Existing SRT or VTT subtitle file."),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output file or destination directory."),
    ] = None,
    language: Annotated[
        str | None, typer.Option("--language", "-l", help="Subtitle language code.")
    ] = None,
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Replace an existing cleaned file."
    ),
) -> None:
    """Clean an existing SRT or VTT file and preserve its format."""

    def run(config: Config) -> None:
        extension = input_file.suffix.lower().lstrip(".")
        if extension not in {"srt", "vtt"}:
            raise SubtitleDiscoveryError("The clean command accepts only SRT or VTT.")
        try:
            content = input_file.read_text(encoding="utf-8-sig")
        except OSError as exc:
            raise SubtitleDiscoveryError(
                f"Unable to read subtitle file: {input_file}", details=str(exc)
            ) from exc
        selected_language = language or config.default_language
        track = SubtitleTrack(
            language_code=selected_language,
            normalized_language_code=selected_language,
            source_type=SubtitleSourceType.MANUAL,
            is_automatic=False,
        )
        segments = SubtitleService(config).parse_and_clean(
            RawSubtitle(content=content, format=extension), track
        )
        destination = _clean_destination(input_file, output)
        source = input_file.resolve()
        video = VideoMetadata(
            title=destination.stem,
            webpage_url=source.as_uri(),
            original_url=str(input_file),
            local_path=source,
        )
        paths = ExportService().export(
            video,
            track,
            segments,
            (extension,),
            destination.parent,
            overwrite=overwrite,
        )
        generated = paths[0]
        if generated != destination and (overwrite or not destination.exists()):
            generated.replace(destination)
            generated = destination
        console.print(f"[bold green]Cleaned:[/bold green] {generated.resolve()}")

    _run_with_config(run)


@app.command()
def web(
    port: int = typer.Option(
        0, "--port", help="Port to bind on 127.0.0.1; 0 chooses a free one."
    ),
    open_browser: bool = typer.Option(
        True, "--open/--no-open", help="Open the page in your default browser."
    ),
) -> None:
    """Serve the CaptionForge interface to a browser on this computer."""

    def run(config: Config) -> None:
        try:
            import uvicorn

            from app.interfaces.web.security import generate_token
            from app.interfaces.web.server import HOST, create_app, find_free_port
        except ImportError as exc:
            raise ConfigurationError(
                "The web interface needs extra packages. Install them with "
                'pip install "captionforge[web]"',
                details=str(exc),
            ) from exc

        selected_port = port or find_free_port()
        token = generate_token()
        url = f"http://{HOST}:{selected_port}/?t={token}"
        application = create_app(config, token=token, port=selected_port)
        console.print(
            Panel(
                f"[bold green]{url}[/bold green]\n\n"
                "Downloading, conversion and transcription all run on this "
                "computer,\nunless you pick Deepgram, which is sent the audio.\n"
                "Keep this terminal open; press Ctrl+C to stop.",
                title=f"{APP_NAME} is serving",
                border_style="green",
            )
        )
        if open_browser:
            webbrowser.open(url)
        uvicorn.run(application, host=HOST, port=selected_port, log_level="warning")

    _run_with_config(run)


@app.command()
def desktop() -> None:
    """Open CaptionForge in a window of its own, with no terminal to keep open."""

    def run(config: Config) -> None:
        from app.interfaces.desktop import launch

        if launch(config):
            console.print(
                f"{APP_NAME} is already open; its window has been brought forward."
            )

    _run_with_config(run)


@app.command()
def update(
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Install every available update without asking."
    ),
) -> None:
    """Check CaptionForge's packages for updates, and install the ones you pick."""

    def run(config: Config) -> None:
        with console.status("Checking for updates…"):
            updates = UPDATER.check(fresh=True)
        if not updates:
            console.print("Everything is up to date.")
            return
        if yes:
            for update in updates:
                _show_update(update)
            console.print()
            chosen = [update.name for update in updates]
        else:
            chosen = _choose_updates(updates, suggested=lambda _update: True)
        _install_updates(chosen)

    _run_with_config(run)


@app.command(name="install-desktop")
def install_desktop(
    remove: bool = typer.Option(
        False, "--remove", help="Remove the entry instead of adding it."
    ),
    workdir: Annotated[
        Path | None,
        typer.Option(
            "--workdir",
            help="Folder the app runs in. Output, temp, and logs land here. "
            "Defaults to the current folder.",
        ),
    ] = None,
) -> None:
    """Add CaptionForge to this computer's applications, or take it away."""

    def run(config: Config) -> None:
        from app.interfaces.launcher import install, uninstall

        if remove:
            removed = uninstall()
            if not removed:
                console.print("[yellow]Nothing to remove:[/yellow] no entry was found.")
                return
            for path in removed:
                console.print(f"[bold green]Removed:[/bold green] {path}")
            return

        installed = install(workdir)
        written = "\n".join(str(path) for path in installed.paths)
        console.print(
            Panel(
                f"[bold green]{APP_NAME}[/bold green] is now in your "
                "applications.\nSearch for it by name and start it like any "
                "other app.\n\n"
                f"Runs in: {installed.workdir}\n"
                f"Command: {' '.join(installed.command)}\n\n"
                f"{written}",
                title="Installed",
                border_style="green",
            )
        )

    _run_with_config(run)


def _clean_destination(input_file: Path, output: Path | None) -> Path:
    """Resolve a predictable cleaned filename without changing formats."""
    suffix = input_file.suffix.lower()
    if output is None:
        return input_file.with_name(f"{input_file.stem}.cleaned{suffix}")
    if output.suffix.lower() in {".srt", ".vtt"}:
        if output.suffix.lower() != suffix:
            raise SubtitleDiscoveryError(
                "Clean output extension must match the input format."
            )
        return output
    return output / f"{input_file.stem}.cleaned{suffix}"


def _create_video_service(config: Config | None = None) -> VideoService:
    """Construct the inspection service graph."""
    return VideoService(YtDlpAdapter(), SubtitleService(config), config)


def _render_inspection(result: SubtitleDiscoveryResult) -> None:
    """Render an inspection result for a human reader."""
    video = result.video
    details = Table(show_header=False, box=None)
    details.add_column("Field", style="cyan")
    details.add_column("Value")
    if video.local_path is not None:
        details.add_row("File", str(video.local_path))
        details.add_row("Title", video.title)
        details.add_row("Duration", _format_duration(video.duration_seconds))
        details.add_row("Preferred language", result.preferred_language)
        console.print(Panel(details, title="File Metadata"))
        # A file carries no YouTube tracks, so there are no tables to fill.
        console.print(
            "\nA file on this computer has no caption tracks to reuse; "
            "'captionforge transcribe' transcribes its audio locally."
        )
        return
    details.add_row("Video ID", video.video_id or "")
    details.add_row("Title", video.title)
    details.add_row("Channel", video.channel_name or "Unknown")
    details.add_row("Duration", _format_duration(video.duration_seconds))
    details.add_row("URL", video.webpage_url)
    details.add_row("Live status", video.live_status or "not live")
    details.add_row("Preferred language", result.preferred_language)
    console.print(Panel(details, title="Video Metadata"))
    _render_tracks("Manual Subtitles", result.manual_tracks, result.selected_track)
    _render_tracks("Automatic Captions", result.automatic_tracks, result.selected_track)
    if result.selected_track:
        track = result.selected_track
        label = track.language_name or track.normalized_language_code
        console.print(
            f"\n[bold green]Selected subtitle:[/bold green] {label} "
            f"({track.normalized_language_code}) — {track.source_type.value.title()}"
        )
        if result.selection_reason:
            console.print(result.selection_reason)
    else:
        console.print(
            "\n[yellow]No subtitle track matched the preferred language.[/yellow]"
        )
        if SubtitleService.translated_matches(result):
            console.print(
                "[yellow]Machine-translated tracks were found but skipped; "
                "use --allow-translated to include them.[/yellow]"
            )


def _render_tracks(
    title: str,
    tracks: tuple[SubtitleTrack, ...],
    selected: SubtitleTrack | None,
) -> None:
    """Render one category of subtitle tracks."""
    table = Table(title=title)
    table.add_column("Language")
    table.add_column("Normalized Code")
    table.add_column("Source")
    table.add_column("Translated")
    table.add_column("Formats")
    table.add_column("Selected")
    for track in tracks:
        table.add_row(
            track.language_name or "Unknown",
            track.normalized_language_code,
            track.source_type.value.title(),
            "Yes" if track.is_translated else "No",
            ", ".join(track.available_formats) or "Unknown",
            "✓" if track == selected else "",
        )
    if not tracks:
        table.add_row("None", "—", "—", "—", "—", "")
    console.print(table)


def _format_duration(seconds: int | None) -> str:
    """Format a numeric duration for terminal display."""
    if seconds is None:
        return "Unknown"
    hours, remainder = divmod(seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes:d}:{remaining_seconds:02d}"


def _format_bytes(size: int | None) -> str:
    """Render an estimated download size, or say plainly that it is unknown."""
    if not size:
        return "unknown"
    if size < 1024 * 1024:
        return f"{size / 1024:.0f} KB"
    if size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.0f} MB"
    return f"{size / (1024 * 1024 * 1024):.1f} GB"


def _exit_code_for(exc: CaptionForgeError) -> int:
    """Map application errors to stable process exit codes."""
    if isinstance(
        exc, (InvalidYouTubeUrlError, UnsupportedYouTubeUrlError, LocalMediaError)
    ):
        return ExitCode.INVALID_INPUT
    if isinstance(exc, (VideoUnavailableError, LiveStreamNotSupportedError)):
        return ExitCode.VIDEO_UNAVAILABLE
    if isinstance(exc, MetadataRetrievalError):
        return ExitCode.METADATA_FAILURE
    return ExitCode.FAILURE


def _is_writable_directory(directory: Path) -> bool:
    """Return whether a directory can be created and written to."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        return directory.is_dir() and os.access(directory, os.W_OK)
    except OSError:
        return False


def _gpu_memory_status() -> str:
    """Report what the driver has free, since that decides which plans fit."""
    from app.adapters.gpu_memory import format_bytes, read_vram

    snapshot = read_vram()
    if snapshot is None:
        return "Not available"
    return (
        f"{format_bytes(snapshot.free_bytes)} free of "
        f"{format_bytes(snapshot.total_bytes)}"
    )


def _planned_configuration(config: Config) -> str:
    """Show the plan a job would start on, after the pre-flight memory check."""
    from app.adapters.gpu_memory import read_vram
    from app.services.transcription_plan import build_ladder, select_plan

    whisper = WhisperAdapter()
    device = "cuda" if whisper.cuda_available() else "cpu"
    requested = TranscriptionPlan(
        model_name=config.default_whisper_model,
        device=device,
        compute_type=WhisperAdapter.select_compute_type(
            config.whisper_compute_type, device
        ),
        beam_size=config.whisper_beam_size,
        word_timestamps=config.whisper_word_timestamps,
    )
    if device != "cuda":
        return requested.describe()
    snapshot = read_vram()
    ladder = build_ladder(requested)
    planned = select_plan(ladder, getattr(snapshot, "free_bytes", None))
    if planned == requested:
        return planned.describe()
    return f"{planned.describe()} - reduced from {requested.model_name} to fit"


def _deepgram_key_status() -> str:
    """Whether a key is there, and from where, without asking Deepgram."""
    key = load_key()
    if key is None:
        return "Not set - run 'captionforge deepgram-key' to add one"
    where = "environment" if key.source == "environment" else str(key_path())
    return f"Set ({key.hint}, from {where})"


def _updates_status(config: Config) -> str:
    """Say when CaptionForge looks for updates; it always asks before installing."""
    if not UPDATER.can_update():
        return "Unavailable - this Python has no pip"
    if not config.check_for_updates:
        return "Asks first; checks only when you run 'captionforge update'"
    return "Asks first; checks when the page opens or YouTube refuses yt-dlp"


def _interactive() -> bool:
    """Whether someone is at the terminal to answer a question."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def _can_offer_updates(config: Config) -> bool:
    return config.check_for_updates and UPDATER.can_update() and _interactive()


def _update_after_refusal() -> bool:
    """Offer a newer yt-dlp after YouTube refused this one; True once it is loaded."""
    console.print(
        "[yellow]YouTube refused the installed yt-dlp.[/yellow] Checking for updates…"
    )
    try:
        updates = UPDATER.check(fresh=True)
    except CaptionForgeError as exc:
        error_console.print(f"[yellow]{exc.message}[/yellow]")
        return False
    if not any(update.name == "yt-dlp" for update in updates):
        version = UPDATER.installed_version("yt-dlp") or "installed"
        console.print(
            f"yt-dlp {version} is already the newest release, so YouTube may be "
            "limiting this connection. Try again later."
        )
        return False
    # yt-dlp is the fix, so it is offered first and suggested; anything else is
    # there to pick, but not pushed.
    chosen = _choose_updates(updates, suggested=lambda update: update.name == "yt-dlp")
    installed = _install_updates(chosen)
    ytdlp = next((item for item in installed if item.name == "yt-dlp"), None)
    if ytdlp is None or ytdlp.restart_needed:
        return False
    console.print(f"Trying again with yt-dlp {ytdlp.version}.")
    return True


def _show_update(update: AvailableUpdate) -> None:
    console.print(
        f"\n[bold]{update.name}[/bold]  {update.installed} → {update.latest}\n"
        f"  [dim]{escape(update.package.mission)}[/dim]"
    )


def _choose_updates(
    updates: tuple[AvailableUpdate, ...],
    *,
    suggested: Callable[[AvailableUpdate], bool],
) -> list[str]:
    """Ask about each update in turn, saying what the package is for."""
    count = len(updates)
    console.print(f"{count} update{'s' if count != 1 else ''} available:")
    chosen: list[str] = []
    for update in updates:
        _show_update(update)
        if typer.confirm(f"  Update {update.name}?", default=suggested(update)):
            chosen.append(update.name)
    return chosen


def _install_updates(names: list[str]) -> tuple[InstalledUpdate, ...]:
    """Install exactly what was chosen and report each result."""
    if not names:
        console.print("Nothing was updated.")
        return ()
    with console.status(f"Updating {', '.join(names)}…"):
        installed = UPDATER.install(names)
    for item in installed:
        later = (
            "; it takes effect the next time CaptionForge starts"
            if item.restart_needed
            else ""
        )
        console.print(
            f"[bold green]Updated[/bold green] {item.name} to {item.version}{later}"
        )
    for name in sorted(set(names) - {item.name for item in installed}):
        console.print(f"{name} was already up to date.")
    return installed


def _javascript_runtime_status() -> str:
    """Report the runtime yt-dlp needs for YouTube signature extraction."""
    if shutil.which("deno"):
        return "Yes (deno)"
    for alternative in ("node", "bun"):
        if shutil.which(alternative):
            return (
                f"{alternative} found, but yt-dlp enables only deno by default; "
                "install deno if extraction starts failing"
            )
    return "No - install deno; yt-dlp has deprecated extraction without one"


def _detected_gpu_name() -> str:
    """Best-effort local GPU name without importing heavyweight torch."""
    try:
        import subprocess

        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
        name = completed.stdout.strip().splitlines()
        return name[0] if name else "Compatible NVIDIA GPU"
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return "Compatible NVIDIA GPU"
