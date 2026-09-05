"""Typer command-line interface for CaptionForge."""

import os
import platform
import shutil
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

from app.adapters.ffmpeg_adapter import FFmpegAdapter
from app.adapters.whisper_adapter import WhisperAdapter
from app.adapters.ytdlp_adapter import YtDlpAdapter
from app.core.config import Config
from app.core.constants import APP_NAME, VERSION, ExitCode
from app.core.exceptions import (
    CaptionForgeError,
    ConfigurationError,
    InvalidYouTubeUrlError,
    LiveStreamNotSupportedError,
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
    try:
        config = Config.load()
        configure_logging(config)
        action(config)
    except CaptionForgeError as exc:
        get_logger().error(
            "Command failed user_message={} technical_cause={}",
            exc.message,
            exc.details or type(exc).__name__,
        )
        error_console.print(f"[bold red]Error:[/bold red] {exc.message}")
        raise typer.Exit(code=_exit_code_for(exc)) from exc
    except KeyboardInterrupt as exc:
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
        try:
            ytdlp_version = package_version("yt-dlp")
            table.add_row("yt-dlp installed", "Yes")
            table.add_row("yt-dlp version", ytdlp_version)
        except PackageNotFoundError:
            table.add_row("yt-dlp installed", "No")
            table.add_row("yt-dlp version", "Not available")
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
        metavar="VIDEO_URL",
        help="An individual YouTube video URL.",
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
        result = _create_video_service().inspect(
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
        ..., metavar="VIDEO_URL", help="An individual YouTube video URL."
    ),
    language: str | None = typer.Option(
        None, "--language", "-l", help="Caption/transcription language."
    ),
    model: str | None = typer.Option(
        None, "--model", help="faster-whisper model name or local model path."
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
        help="Vocabulary hint for Whisper, e.g. names and terms in the video.",
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
    """Export existing captions or fall back to local faster-whisper."""

    def run(config: Config) -> None:
        adapter = YtDlpAdapter()
        subtitles = SubtitleService(config)
        video_service = VideoService(adapter, subtitles)
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
        )

        def report(message: str, percent: float | None) -> None:
            suffix = f" ({percent:.0f}%)" if percent is not None else ""
            console.print(f"[cyan]{message}{suffix}[/cyan]")

        result = service.process(
            video_url,
            language=language,
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
        source = (
            "existing captions"
            if result.used_existing_captions
            else "local Whisper transcription"
        )
        console.print(f"[bold green]Completed using {source}.[/bold green]")
        if result.transcription is not None:
            info = result.transcription
            probability = (
                f", probability {info.language_probability:.1%}"
                if info.language_probability is not None
                else ""
            )
            console.print(
                f"Language: {info.detected_language}{probability}; "
                f"model: {info.model_name}; device: {info.device}; "
                f"compute type: {info.compute_type}"
            )
        if result.prepared_audio_path is not None:
            console.print(
                f"Prepared audio kept at: {result.prepared_audio_path.resolve()}",
                markup=False,
            )
        for path in result.paths:
            console.print(str(path.resolve()), markup=False)

    _run_with_config(run)


@app.command(name="prepare-audio")
def prepare_audio(
    video_url: str = typer.Argument(
        ..., metavar="VIDEO_URL", help="An individual YouTube video URL."
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
            VideoService(adapter, SubtitleService(config)),
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
        discovery = VideoService(adapter, subtitle_service).inspect(
            video_url, preferred, allow_translated=allow_translated
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
            discovery.video.video_id,
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
        video = VideoMetadata(
            video_id="localclean1",
            title=destination.stem,
            webpage_url="file://local",
            original_url=str(input_file),
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
                "computer.\nKeep this terminal open; press Ctrl+C to stop.",
                title=f"{APP_NAME} is serving",
                border_style="green",
            )
        )
        if open_browser:
            webbrowser.open(url)
        uvicorn.run(application, host=HOST, port=selected_port, log_level="warning")

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
    details.add_row("Video ID", video.video_id)
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
    if isinstance(exc, (InvalidYouTubeUrlError, UnsupportedYouTubeUrlError)):
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
