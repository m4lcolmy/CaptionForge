"""Updates are listed with what each package is for, and installed only if picked."""

import json
import subprocess
import sys
import tomllib
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner
from yt_dlp.utils import DownloadError

from app.adapters import package_updater
from app.adapters.package_updater import (
    PACKAGES,
    AvailableUpdate,
    InstalledUpdate,
    Package,
    PackageUpdater,
)
from app.adapters.ytdlp_adapter import YtDlpAdapter
from app.core.config import Config
from app.core.exceptions import (
    AudioStreamForbiddenError,
    MediaStreamForbiddenError,
    MetadataRefusedError,
    PackageUpdateError,
    ValidationError,
    VideoUnavailableError,
)
from app.interfaces import cli
from app.interfaces.web import server as web_server
from tests.conftest import VIDEO_ID, VIDEO_URL, FakeExtractor, extractor_factory

PROBE = "captionforge_update_probe"
REFUSED = "ERROR: [youtube] abc: HTTP Error 403: Forbidden"
YTDLP = next(package for package in PACKAGES if package.name == "yt-dlp")
WHISPER = next(package for package in PACKAGES if package.name == "faster-whisper")


# ---------- the updater ----------


class FakePip:
    """Stands in for pip: answers dry runs, and 'installs' by changing versions."""

    def __init__(self, installed: dict[str, str], newest: dict[str, str]) -> None:
        self.installed = installed
        self.newest = newest
        self.commands: list[Sequence[str]] = []
        self.returncode = 0
        self.on_install: Callable[[], None] = lambda: None

    def run(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        if self.returncode:
            return subprocess.CompletedProcess(command, 1, "", "network is down")
        named = [spec.split(">")[0].split("<")[0] for spec in _specs_in(command)]
        if "--dry-run" in command:
            report = {
                "install": [
                    {"metadata": {"name": name, "version": self.newest[name]}}
                    | {"requested": True}
                    for name in named
                    if self.newest.get(name, self.installed[name])
                    != self.installed[name]
                ]
            }
            return subprocess.CompletedProcess(command, 0, json.dumps(report), "")
        for name in named:
            self.installed[name] = self.newest.get(name, self.installed[name])
        self.on_install()
        return subprocess.CompletedProcess(command, 0, "", "")


def _specs_in(command: Sequence[str]) -> list[str]:
    after = list(command)[list(command).index("install") + 1 :]
    return [part for part in after if not part.startswith("-") and not part.isdigit()]


@pytest.fixture
def pip(monkeypatch: pytest.MonkeyPatch) -> FakePip:
    fake = FakePip(
        installed={"yt-dlp": "2026.8.19", "faster-whisper": "1.1.0"},
        newest={"yt-dlp": "2026.9.20", "faster-whisper": "1.2.0"},
    )
    monkeypatch.setattr(package_updater, "package_version", _versions(fake))
    return fake


def _versions(fake: FakePip) -> Callable[[str], str]:
    def version(name: str) -> str:
        if name not in fake.installed:
            raise package_updater.PackageNotFoundError(name)
        return fake.installed[name]

    return version


def updater_for(
    fake: FakePip,
    packages: Sequence[Package] = (YTDLP, WHISPER),
    clock: Callable[[], float] | None = None,
) -> PackageUpdater:
    return PackageUpdater(packages, runner=fake.run, clock=clock or (lambda: 0.0))


def test_a_check_lists_what_is_newer_and_what_each_package_is_for(
    pip: FakePip,
) -> None:
    updates = updater_for(pip).check()

    assert [update.as_dict() for update in updates] == [
        {
            "name": "yt-dlp",
            "installed": "2026.8.19",
            "latest": "2026.9.20",
            "mission": YTDLP.mission,
        },
        {
            "name": "faster-whisper",
            "installed": "1.1.0",
            "latest": "1.2.0",
            "mission": WHISPER.mission,
        },
    ]
    (command,) = pip.commands
    assert "--dry-run" in command
    # Upgrades stay inside the range CaptionForge was written against.
    assert "faster-whisper>=1.1,<2.0" in command
    assert pip.installed == {"yt-dlp": "2026.8.19", "faster-whisper": "1.1.0"}


def test_packages_that_are_not_installed_are_not_offered(pip: FakePip) -> None:
    del pip.installed["faster-whisper"]

    updates = updater_for(pip).check()

    assert [update.name for update in updates] == ["yt-dlp"]
    assert not any("faster-whisper" in part for part in pip.commands[0])


def test_the_page_does_not_run_pip_every_time_it_opens(pip: FakePip) -> None:
    now = [0.0]
    updater = updater_for(pip, clock=lambda: now[0])

    updater.check()
    now[0] = package_updater.CHECK_MAX_AGE_SECONDS - 1
    updater.check()
    assert len(pip.commands) == 1

    updater.check(fresh=True)
    assert len(pip.commands) == 2


def test_a_failed_check_says_so(pip: FakePip) -> None:
    pip.returncode = 1

    with pytest.raises(PackageUpdateError, match="could not check for updates"):
        updater_for(pip).check()


def test_only_the_picked_packages_are_installed(pip: FakePip) -> None:
    updater = updater_for(pip)
    updater.check()

    installed = updater.install(["faster-whisper"])

    assert installed == (InstalledUpdate("faster-whisper", "1.2.0", False),)
    install_command = pip.commands[-1]
    assert "--dry-run" not in install_command
    assert not any(part.startswith("yt-dlp") for part in install_command)
    assert pip.installed["yt-dlp"] == "2026.8.19"
    # What was installed is no longer offered; what was not still is.
    assert [update.name for update in updater.check()] == ["yt-dlp"]


def test_names_from_outside_the_list_never_reach_pip(pip: FakePip) -> None:
    with pytest.raises(ValidationError):
        updater_for(pip).install(["yt-dlp", "requests; rm -rf ~"])

    assert pip.commands == []


def test_a_failed_install_says_so(pip: FakePip) -> None:
    pip.returncode = 1

    with pytest.raises(PackageUpdateError, match="could not be installed"):
        updater_for(pip).install(["yt-dlp"])


def test_without_pip_nothing_runs(
    pip: FakePip, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    updater = updater_for(pip)

    assert updater.can_update() is False
    with pytest.raises(PackageUpdateError):
        updater.check()
    assert pip.commands == []


@pytest.fixture
def probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A tiny importable module playing the part of an installed package."""
    source = tmp_path / f"{PROBE}.py"
    source.write_text("VALUE = 'old'\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    __import__(PROBE)
    yield source
    sys.modules.pop(PROBE, None)


def test_a_new_yt_dlp_is_loaded_into_the_running_process(
    pip: FakePip, probe: Path
) -> None:
    package = Package("yt-dlp", PROBE, "", "test", reloads=True)
    pip.on_install = lambda: probe.write_text("VALUE = 'new'\n", encoding="utf-8")

    installed = updater_for(pip, (package,)).install(["yt-dlp"])

    assert installed == (InstalledUpdate("yt-dlp", "2026.9.20", False),)
    assert sys.modules[PROBE].VALUE == "new"


def test_a_release_that_will_not_load_keeps_the_old_one_running(
    pip: FakePip, probe: Path
) -> None:
    package = Package("yt-dlp", PROBE, "", "test", reloads=True)
    pip.on_install = lambda: probe.write_text("not python\n", encoding="utf-8")
    loaded = sys.modules[PROBE]

    installed = updater_for(pip, (package,)).install(["yt-dlp"])

    assert installed == (InstalledUpdate("yt-dlp", "2026.9.20", True),)
    assert sys.modules[PROBE] is loaded


def test_a_package_already_in_use_says_it_needs_a_restart(
    pip: FakePip, probe: Path
) -> None:
    package = Package("faster-whisper", PROBE, "", "test")

    installed = updater_for(pip, (package,)).install(["faster-whisper"])

    assert installed == (InstalledUpdate("faster-whisper", "1.2.0", True),)


def test_the_list_mirrors_the_versions_in_pyproject() -> None:
    project = tomllib.loads(
        (Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    declared = list(project["dependencies"])
    for extra, requirements in project["optional-dependencies"].items():
        if extra != "dev":
            declared.extend(requirements)
    specs = {
        package_updater._normalize(r.split(">")[0].split("<")[0]): r for r in declared
    }

    # Names are compared the way pip compares them: PySide6-Essentials is
    # spelled with capitals in both places, but it need not be.
    listed = {
        package_updater._normalize(package.name): package.name + package.requirement
        for package in PACKAGES
    }

    assert set(listed) == set(specs)
    for name, requirement in listed.items():
        assert sorted(requirement[len(name) :].split(",")) == sorted(
            specs[name][len(name) :].split(",")
        ), name
    assert all(package.mission for package in PACKAGES)


# ---------- the adapter only reports ----------


def test_a_refusal_is_reported_and_nothing_is_installed() -> None:
    adapter = YtDlpAdapter(
        extractor_factory(FakeExtractor(error=DownloadError(REFUSED)))
    )

    with pytest.raises(MetadataRefusedError) as caught:
        adapter.inspect(VIDEO_ID, VIDEO_URL)

    assert "too old" in caught.value.message
    assert caught.value.retryable is False


def test_errors_from_a_freshly_loaded_yt_dlp_are_still_recognised(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Loading a new yt-dlp brings a new DownloadError class with it."""

    class ReloadedDownloadError(Exception):
        pass

    monkeypatch.setitem(
        sys.modules,
        "yt_dlp.utils",
        SimpleNamespace(DownloadError=ReloadedDownloadError),
    )

    class Refused:
        def __call__(self, _options: dict[str, Any]) -> "Refused":
            return self

        def __enter__(self) -> "Refused":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def extract_info(self, url: str, *, download: bool) -> Any:
            raise ReloadedDownloadError(REFUSED)

    with pytest.raises(AudioStreamForbiddenError):
        YtDlpAdapter(Refused()).download_audio(VIDEO_ID, tmp_path)  # type: ignore[arg-type]


def test_streams_vanishing_during_inspection_blames_the_old_yt_dlp() -> None:
    """When yt-dlp cannot read YouTube's player, every format disappears."""
    error = DownloadError("ERROR: [youtube] abc: Requested format is not available")
    adapter = YtDlpAdapter(extractor_factory(FakeExtractor(error=error)))

    with pytest.raises(MetadataRefusedError) as caught:
        adapter.inspect(VIDEO_ID, VIDEO_URL)

    assert not isinstance(caught.value, VideoUnavailableError)


# ---------- the terminal asks ----------


class FakeUpdater:
    """Updater double for the interfaces: records what was checked and installed."""

    def __init__(
        self,
        updates: Sequence[Package] = (),
        installed: Sequence[InstalledUpdate] = (),
    ) -> None:
        self.updates = tuple(
            AvailableUpdate(package, "1.0", "2.0") for package in updates
        )
        self.installed = tuple(installed)
        self.checks: list[bool] = []
        self.installs: list[list[str]] = []

    @staticmethod
    def can_update() -> bool:
        return True

    @staticmethod
    def installed_version(name: str) -> str:
        return "2026.8.19"

    def check(self, *, fresh: bool = False) -> tuple[AvailableUpdate, ...]:
        self.checks.append(fresh)
        return self.updates

    def install(self, names: Sequence[str]) -> tuple[InstalledUpdate, ...]:
        self.installs.append(list(names))
        return tuple(item for item in self.installed if item.name in names)


runner = CliRunner()


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> Callable[[FakeUpdater], None]:
    """Pretend someone is at the terminal, with updates turned on."""
    monkeypatch.setenv("CAPTIONFORGE_CHECK_FOR_UPDATES", "true")
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    return lambda fake: monkeypatch.setattr(cli, "UPDATER", fake)


class RefusedOnce:
    """Video service that YouTube refuses until yt-dlp has been updated."""

    def __init__(self, fake: FakeUpdater, result: Any) -> None:
        self.fake = fake
        self.result = result
        self.calls = 0

    def inspect(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if not self.fake.installs:
            raise MetadataRefusedError("YouTube refused to describe this video.")
        return self.result


def test_the_terminal_asks_before_updating_yt_dlp_and_then_tries_again(
    monkeypatch: pytest.MonkeyPatch,
    terminal: Callable[[FakeUpdater], None],
    video_metadata: Any,
) -> None:
    from app.models.subtitle import SubtitleDiscoveryResult

    fake = FakeUpdater(
        updates=(YTDLP, WHISPER),
        installed=(InstalledUpdate("yt-dlp", "2026.9.20", False),),
    )
    terminal(fake)
    service = RefusedOnce(
        fake, SubtitleDiscoveryResult(video=video_metadata, preferred_language="ar")
    )
    monkeypatch.setattr(cli, "_create_video_service", lambda: service)

    result = runner.invoke(cli.app, ["inspect", VIDEO_URL], input="y\nn\n")

    assert result.exit_code == 0, result.output
    assert "Reads YouTube" in result.stdout
    assert "Transcribes the audio" in result.stdout
    assert fake.checks == [True]
    assert fake.installs == [["yt-dlp"]]
    assert "Trying again with yt-dlp 2026.9.20" in result.stdout
    assert service.calls == 2


def test_saying_no_installs_nothing(
    monkeypatch: pytest.MonkeyPatch, terminal: Callable[[FakeUpdater], None]
) -> None:
    fake = FakeUpdater(updates=(YTDLP,))
    terminal(fake)
    service = RefusedOnce(fake, None)
    monkeypatch.setattr(cli, "_create_video_service", lambda: service)

    result = runner.invoke(cli.app, ["inspect", VIDEO_URL], input="n\n")

    assert result.exit_code != 0
    assert fake.installs == []
    assert "Nothing was updated" in result.stdout
    assert "YouTube refused" in result.stderr
    assert service.calls == 1


def test_a_script_is_never_asked_and_gets_the_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeUpdater(updates=(YTDLP,))
    monkeypatch.setattr(cli, "UPDATER", fake)
    monkeypatch.setattr(cli, "_interactive", lambda: False)
    monkeypatch.setattr(cli, "_create_video_service", lambda: RefusedOnce(fake, None))

    result = runner.invoke(cli.app, ["inspect", VIDEO_URL])

    assert result.exit_code != 0
    assert fake.checks == []
    assert "captionforge update" in result.stderr


def test_the_update_command_lists_missions_and_installs_what_was_picked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeUpdater(
        updates=(YTDLP, WHISPER),
        installed=(InstalledUpdate("faster-whisper", "2.0", True),),
    )
    monkeypatch.setattr(cli, "UPDATER", fake)

    result = runner.invoke(cli.app, ["update"], input="n\ny\n")

    assert result.exit_code == 0, result.output
    assert "Reads YouTube" in result.stdout
    assert "Transcribes the audio" in result.stdout
    assert fake.installs == [["faster-whisper"]]
    assert "takes effect" in result.stdout


def test_the_update_command_with_nothing_to_do(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "UPDATER", FakeUpdater())

    result = runner.invoke(cli.app, ["update"])

    assert result.exit_code == 0
    assert "Everything is up to date" in result.stdout


# ---------- the page asks ----------

TOKEN = "test-token-value"
PORT = 8765
HEADERS = {"X-CaptionForge-Token": TOKEN, "Host": f"127.0.0.1:{PORT}"}


def page(tmp_path: Path, **settings: Any) -> TestClient:
    application = web_server.create_app(
        Config(default_output_folder=tmp_path / "output", **settings),
        token=TOKEN,
        port=PORT,
        preferences_file=tmp_path / "web-preferences.json",
    )
    return TestClient(application, base_url=f"http://127.0.0.1:{PORT}")


def test_the_page_lists_updates_with_missions_and_installs_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeUpdater(updates=(YTDLP,))
    monkeypatch.setattr(package_updater, "UPDATER", fake)

    with page(tmp_path) as client:
        response = client.get("/api/updates?fresh=1", headers=HEADERS)

    assert response.status_code == 200
    assert response.json() == {
        "checked": True,
        "updates": [
            {
                "name": "yt-dlp",
                "installed": "1.0",
                "latest": "2.0",
                "mission": YTDLP.mission,
            }
        ],
    }
    assert fake.checks == [True]
    assert fake.installs == []


def test_the_page_does_not_check_when_checks_are_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeUpdater(updates=(YTDLP,))
    monkeypatch.setattr(package_updater, "UPDATER", fake)

    with page(tmp_path, check_for_updates=False) as client:
        response = client.get("/api/updates", headers=HEADERS)

    assert response.json() == {"checked": False, "updates": []}
    assert fake.checks == []


def test_the_page_installs_exactly_what_was_ticked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeUpdater(
        updates=(YTDLP, WHISPER),
        installed=(InstalledUpdate("yt-dlp", "2.0", False),),
    )
    monkeypatch.setattr(package_updater, "UPDATER", fake)

    with page(tmp_path) as client:
        response = client.post(
            "/api/updates", json={"packages": ["yt-dlp"]}, headers=HEADERS
        )

    assert response.status_code == 200
    assert response.json() == {
        "updated": [{"name": "yt-dlp", "version": "2.0", "restart_needed": False}]
    }
    assert fake.installs == [["yt-dlp"]]


def test_the_page_cannot_install_something_off_the_list(
    pip: FakePip, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(package_updater, "UPDATER", updater_for(pip))

    with page(tmp_path) as client:
        response = client.post(
            "/api/updates", json={"packages": ["requests"]}, headers=HEADERS
        )

    assert response.status_code == 400
    assert pip.commands == []


def test_a_refused_lookup_tells_the_page_an_update_may_help(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Refusing:
        def inspect_all(self, *args: Any, **kwargs: Any) -> Any:
            raise MetadataRefusedError("YouTube refused to describe this video.")

    monkeypatch.setattr(web_server, "create_video_service", lambda config: Refusing())

    with page(tmp_path) as client:
        response = client.post("/api/inspect", json={"url": VIDEO_URL}, headers=HEADERS)

    assert response.json()["update_may_help"] is True


def test_a_refused_download_job_tells_the_page_an_update_may_help(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from app.interfaces.web import jobs as jobs_module

    class Refusing:
        def download(self, *args: Any, **kwargs: Any) -> Any:
            raise MediaStreamForbiddenError("YouTube refused the media stream.")

    monkeypatch.setattr(jobs_module, "create_media_service", lambda config: Refusing())

    with page(tmp_path) as client:
        created = client.post(
            "/api/media", json={"url": VIDEO_URL, "quality": "720"}, headers=HEADERS
        )
        for _ in range(200):
            job = client.get(f"/api/jobs/{created.json()['id']}", headers=HEADERS)
            if job.json()["status"] == "failed":
                break

    assert job.json()["update_may_help"] is True
