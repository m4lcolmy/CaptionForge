"""CaptionForge command-line entry point."""

from app.interfaces.cli import app


def main() -> None:
    """Run the CaptionForge command-line application."""
    app()


if __name__ == "__main__":
    main()
