"""Offline tests for remembered web interface choices."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Config
from app.interfaces.web import server as web_server
from app.interfaces.web.preferences import (
    WebPreferences,
    load_preferences,
    resolve,
    save_preferences,
)

TOKEN = "test-token-value"
PORT = 8765
HEADERS = {"X-CaptionForge-Token": TOKEN, "Host": f"127.0.0.1:{PORT}"}


@pytest.fixture
def store(tmp_path: Path) -> Path:
    return tmp_path / "web-preferences.json"


def client_for(store: Path, config: Config | None = None) -> TestClient:
    application = web_server.create_app(
        config or Config(), token=TOKEN, port=PORT, preferences_file=store
    )
    return TestClient(application, base_url=f"http://127.0.0.1:{PORT}")


def test_missing_file_yields_defaults(store: Path) -> None:
    assert load_preferences(store) == WebPreferences()


def test_round_trip(store: Path) -> None:
    saved = WebPreferences(
        language="tr", formats=("docx",), model="medium", device="cpu", force=True
    )
    save_preferences(saved, store)
    assert load_preferences(store) == saved


def test_unreadable_file_does_not_raise(store: Path) -> None:
    store.write_text("{ not json", encoding="utf-8")
    assert load_preferences(store) == WebPreferences()


def test_invalid_fields_fall_back_individually(store: Path) -> None:
    """A bad device must not cost the user their other choices."""
    store.write_text(
        json.dumps({"language": "ar", "device": "quantum", "formats": ["srt", "mp3"]}),
        encoding="utf-8",
    )
    loaded = load_preferences(store)
    assert loaded.language == "ar"
    assert loaded.device is None
    assert loaded.formats == ("srt",)


def test_unknown_keys_from_another_version_are_ignored(store: Path) -> None:
    store.write_text(json.dumps({"language": "ar", "future": 1}), encoding="utf-8")
    assert load_preferences(store).language == "ar"


def test_resolve_prefers_saved_choices_over_configuration() -> None:
    config = Config(default_language="ar", default_output_formats=("srt", "vtt"))
    saved = WebPreferences(language="tr", formats=("docx", "txt"))
    merged = resolve(saved, config)
    assert merged["language"] == "tr"
    assert merged["formats"] == ["docx", "txt"]


def test_resolve_falls_back_to_one_configured_format() -> None:
    config = Config(default_output_formats=("srt", "vtt"))
    merged = resolve(WebPreferences(), config)
    assert merged["formats"] == ["srt"]
    assert merged["postprocess"] is True


def test_health_reports_the_starting_state(store: Path) -> None:
    with client_for(store) as client:
        payload = client.get("/api/health", headers=HEADERS).json()
    assert payload["preferences"]["formats"] == ["srt"]
    assert payload["preferences"]["language"] == "ar"


def test_saved_choices_come_back_on_the_next_session(store: Path) -> None:
    """The whole point: close the page, reopen it, find the same options."""
    with client_for(store) as client:
        written = client.put(
            "/api/preferences",
            json={
                "language": "tr",
                "formats": ["docx", "txt"],
                "model": "medium",
                "device": "cpu",
                "prompt": "Ibn Taymiyyah",
                "timestamped_txt": True,
                "postprocess": False,
            },
            headers=HEADERS,
        )
        assert written.status_code == 200

    assert store.is_file()

    # A brand new server process, as if the app were restarted.
    with client_for(store) as client:
        payload = client.get("/api/health", headers=HEADERS).json()
    starting = payload["preferences"]
    assert starting["language"] == "tr"
    assert starting["formats"] == ["docx", "txt"]
    assert starting["model"] == "medium"
    assert starting["device"] == "cpu"
    assert starting["prompt"] == "Ibn Taymiyyah"
    assert starting["timestamped_txt"] is True
    assert starting["postprocess"] is False


def test_rejected_device_never_reaches_disk(store: Path) -> None:
    with client_for(store) as client:
        response = client.put(
            "/api/preferences", json={"device": "quantum"}, headers=HEADERS
        )
    assert response.status_code == 422
    assert not store.exists()


def test_preferences_require_the_token(store: Path) -> None:
    with client_for(store) as client:
        response = client.put(
            "/api/preferences",
            json={"language": "tr"},
            headers={"Host": f"127.0.0.1:{PORT}"},
        )
    assert response.status_code == 401
    assert not store.exists()


def test_preferences_never_change_the_cli_configuration(store: Path) -> None:
    """Interface choices live in their own file and must not touch config.json."""
    config = Config(default_language="ar", default_output_formats=("srt", "vtt"))
    with client_for(store, config) as client:
        client.put("/api/preferences", json={"language": "tr"}, headers=HEADERS)
    assert config.default_language == "ar"
    assert config.default_output_formats == ("srt", "vtt")
    assert "web-preferences" in store.name
