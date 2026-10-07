"""The page's colours and type, restated for Qt so the window looks the same.

``app/interfaces/web/static/app.css`` is the source of truth. Every token here
has the same name and value, and the stylesheet below follows the page's rules
section by section, with rem converted at 16 px. Qt style sheets have no
``letter-spacing``, ``text-transform``, ``opacity`` or ``:has()``, so the few
rules that need them are carried out by the widgets instead.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase, QGuiApplication, QPalette
from PySide6.QtWidgets import QApplication

# One analogous green ramp, and the ink that sits on it. Neither changes
# between light and dark.
LIME = "#A8F03C"
SPRING = "#34E27A"
EMERALD = "#00D68F"
TEAL = "#00C2AE"
ON_VIVID = "#052018"

RADIUS = 10
BASE_PIXELS = 15

# The page asks for weights 550 and 650, which Chrome draws from the variable
# face. Qt snaps them up to the next named weight, so they are written here as
# 500 and 600, which is what they look like in the page.


@dataclass(frozen=True)
class Palette:
    """The tokens app.css redefines for a dark colour scheme."""

    name: str
    bg: str
    surface: str
    sunk: str
    ink: str
    ink_2: str
    muted: str
    line: str
    bad: str
    bad_bg: str


LIGHT = Palette(
    name="light",
    bg="#F4FBF5",
    surface="#FFFFFF",
    sunk="#EAF7EC",
    ink="#06251A",
    ink_2="#2F5B47",
    muted="#6F9685",
    line="#D5EDDB",
    bad="#C8352F",
    bad_bg="#FDEDEC",
)

DARK = Palette(
    name="dark",
    bg="#04110D",
    surface="#091C15",
    sunk="#0D2820",
    ink="#E2FBEE",
    ink_2="#A9D8C0",
    muted="#6E9C86",
    line="#163427",
    bad="#FF8177",
    bad_bg="#2A100E",
)


def palette_for(scheme: Qt.ColorScheme) -> Palette:
    """Pick the tokens for a colour scheme; an unknown one reads as light."""
    return DARK if scheme == Qt.ColorScheme.Dark else LIGHT


def mono_family() -> str:
    """The system's monospace family, which is what ``ui-monospace`` resolves to."""
    return QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont).family()


def font(
    pixels: int,
    *,
    weight: int = 400,
    mono: bool = False,
    spacing: float = 0.0,
) -> QFont:
    """A font at a CSS pixel size, for the widgets that paint their own text."""
    chosen = QFont(mono_family()) if mono else QFont(QApplication.font())
    chosen.setPixelSize(pixels)
    chosen.setWeight(QFont.Weight(weight))
    if spacing:
        chosen.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    return chosen


class Theme(QObject):
    """The palette in use, applied to the whole application at once.

    Most of the window is drawn by the style sheet. The chips, the progress
    track and the rich-text labels paint colours themselves; they read
    :attr:`palette` while painting and listen to :attr:`changed` for the rest.
    """

    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.palette = LIGHT
        # The check and radio marks are images in a style sheet, and a style
        # sheet can only point at files. They live for as long as the app does.
        self._images = tempfile.TemporaryDirectory(prefix="captionforge-theme-")

    def install(self, application: QApplication) -> None:
        """Style the application now, and again whenever the system flips scheme."""
        application.setStyle("Fusion")
        base = QFont(application.font())
        base.setPixelSize(BASE_PIXELS)
        application.setFont(base)
        hints = QGuiApplication.styleHints()
        hints.colorSchemeChanged.connect(lambda scheme: self.apply(palette_for(scheme)))
        self.apply(palette_for(hints.colorScheme()))

    def apply(self, palette: Palette) -> None:
        """Switch every widget to one palette and ask the painted ones to redraw."""
        application = QApplication.instance()
        if not isinstance(application, QApplication):
            return
        self.palette = palette
        application.setPalette(qt_palette(palette))
        application.setStyleSheet(stylesheet(palette, self._indicators(palette)))
        self.changed.emit()
        for widget in application.allWidgets():
            widget.update()

    def _indicators(self, palette: Palette) -> dict[str, str]:
        """Write this palette's check and radio marks, and return their paths."""
        folder = Path(self._images.name)
        paths: dict[str, str] = {}
        for key, svg in indicator_svgs(palette).items():
            path = folder / f"{key}-{palette.name}.svg"
            if not path.exists():
                path.write_text(svg, encoding="utf-8")
            paths[key] = path.as_posix()
        return paths


THEME: Theme | None = None


def theme() -> Theme:
    """The one theme the application shares, made on first use."""
    global THEME
    if THEME is None:
        THEME = Theme()
    return THEME


def current() -> Palette:
    """The palette in use right now."""
    return theme().palette


def qt_palette(palette: Palette) -> QPalette:
    """Colour the parts of Qt the style sheet does not reach, such as selections."""
    colors = QPalette()
    role = QPalette.ColorRole
    for target, value in (
        (role.Window, palette.bg),
        (role.Base, palette.surface),
        (role.AlternateBase, palette.sunk),
        (role.Button, palette.surface),
        (role.WindowText, palette.ink),
        (role.Text, palette.ink),
        (role.ButtonText, palette.ink),
        (role.PlaceholderText, palette.muted),
        (role.Highlight, EMERALD),
        (role.HighlightedText, ON_VIVID),
        (role.Link, EMERALD),
        (role.LinkVisited, EMERALD),
        (role.ToolTipBase, palette.surface),
        (role.ToolTipText, palette.ink),
    ):
        colors.setColor(target, QColor(value))
    return colors


def shade(color: str, factor: float) -> str:
    """Brighten or darken a colour, as CSS ``filter: brightness()`` does."""
    source = QColor(color)
    return QColor(
        min(255, round(source.red() * factor)),
        min(255, round(source.green() * factor)),
        min(255, round(source.blue() * factor)),
    ).name()


def ramp(*stops: tuple[float, str], factor: float = 1.0, lean: float = 0.18) -> str:
    """A left-to-right gradient that leans a little downwards, like ``100deg``."""
    points = ", ".join(
        f"stop:{position} {shade(color, factor)}" for position, color in stops
    )
    return f"qlineargradient(x1:0, y1:0, x2:1, y2:{lean}, {points})"


def indicator_svgs(palette: Palette) -> dict[str, str]:
    """Check and radio marks in the page's accent colour, as Chrome draws them."""
    box = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 15 15" '
        'width="15" height="15">{}</svg>'
    )
    return {
        "check-off": box.format(
            f'<rect x="0.75" y="0.75" width="13.5" height="13.5" rx="2.5" '
            f'fill="{palette.surface}" stroke="{palette.muted}" stroke-width="1.5"/>'
        ),
        "check-on": box.format(
            f'<rect width="15" height="15" rx="3" fill="{EMERALD}"/>'
            f'<path d="M3.6 7.8 6.2 10.3 11.4 4.9" fill="none" stroke="{ON_VIVID}" '
            'stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/>'
        ),
        "radio-off": box.format(
            f'<circle cx="7.5" cy="7.5" r="6.75" fill="{palette.surface}" '
            f'stroke="{palette.muted}" stroke-width="1.5"/>'
        ),
        "radio-on": box.format(
            f'<circle cx="7.5" cy="7.5" r="6.5" fill="{palette.surface}" '
            f'stroke="{EMERALD}" stroke-width="2"/>'
            f'<circle cx="7.5" cy="7.5" r="3.6" fill="{EMERALD}"/>'
        ),
        # The page draws its dropdown chevron from two gradients in --muted.
        "chevron": (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 6" '
            'width="10" height="6"><path d="M1 1 5 5 9 1" fill="none" '
            f'stroke="{palette.muted}" stroke-width="1.6" '
            'stroke-linecap="round" stroke-linejoin="round"/></svg>'
        ),
    }


def stylesheet(palette: Palette, marks: dict[str, str]) -> str:
    """The page's style sheet, rule for rule, in Qt's dialect."""
    p = palette
    mono = mono_family()
    go = ramp((0, LIME), (0.45, SPRING), (1, TEAL))
    go_hover = ramp((0, LIME), (0.45, SPRING), (1, TEAL), factor=1.06)
    go_down = ramp((0, LIME), (0.45, SPRING), (1, TEAL), factor=0.96)
    return f"""
/* ---- shell ---- */

QMainWindow, QWidget#page {{ background: {p.bg}; }}
QWidget {{ color: {p.ink}; }}
QScrollArea {{ background: {p.bg}; border: none; }}
QScrollArea > QWidget > QWidget {{ background: {p.bg}; }}

QToolTip {{
  background: {p.surface}; color: {p.ink};
  border: 1px solid {p.line}; border-radius: 6px; padding: 4px 8px;
}}

QScrollBar:vertical {{ background: transparent; width: 11px; margin: 0; }}
QScrollBar::handle:vertical {{
  background: {p.line}; border-radius: 4px; min-height: 36px; margin: 2px;
}}
QScrollBar::handle:vertical:hover {{ background: {p.muted}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}

/* ---- top bar ---- */

QFrame#bar {{ background: {p.bg}; border: none; border-bottom: 1px solid {p.line}; }}
QLabel#brand {{ font-size: 15px; font-weight: 600; }}
QLabel#meta {{ font-family: "{mono}"; font-size: 11px; color: {p.muted}; }}

/* ---- text ---- */

QLabel {{ background: transparent; }}
QLabel[role="label"] {{ color: {p.muted}; }}
QLabel[role="note"] {{ color: {p.muted}; font-size: 14px; }}
QLabel[role="help"] {{ color: {p.muted}; font-size: 13px; }}
QLabel[role="title"] {{ font-size: 17px; font-weight: 600; }}
QLabel#key-saved-text {{ font-family: "{mono}"; font-size: 13px; }}

/* ---- inputs ---- */

QLineEdit {{
  font-size: 15px; color: {p.ink}; background: {p.surface};
  border: 1px solid {p.line}; border-radius: {RADIUS}px; padding: 10px 13px;
  selection-background-color: {EMERALD}; selection-color: {ON_VIVID};
}}
QLineEdit:focus {{ border: 2px solid {EMERALD}; padding: 9px 12px; }}
QLineEdit[dropping="true"] {{
  border: 1px dashed {EMERALD}; padding: 10px 13px; background: {p.sunk};
}}
QLineEdit#deepgram-key {{ font-family: "{mono}"; }}

QComboBox {{
  font-size: 15px; color: {p.ink}; background: {p.surface};
  border: 1px solid {p.line}; border-radius: {RADIUS}px;
  padding: 10px 38px 10px 13px;
}}
QComboBox:hover {{ border-color: {EMERALD}; }}
QComboBox:focus {{ border: 2px solid {EMERALD}; padding: 9px 37px 9px 12px; }}
QComboBox::drop-down {{
  subcontrol-origin: padding; subcontrol-position: center right;
  width: 38px; border: none; background: transparent;
}}
QComboBox::down-arrow {{ image: url("{marks['chevron']}"); width: 10px; height: 6px; }}
QComboBox QAbstractItemView {{
  background: {p.surface}; color: {p.ink}; border: 1px solid {p.line};
  border-radius: 6px; padding: 4px; outline: none;
  selection-background-color: {p.sunk}; selection-color: {p.ink};
}}

/* ---- buttons ---- */

QPushButton {{ font-size: 15px; border-radius: {RADIUS}px; }}

QPushButton[variant="ghost"] {{
  background: {p.surface}; color: {p.ink}; border: 1px solid {p.line};
  padding: 10px 17px; font-weight: 500;
}}
QPushButton[variant="ghost"]:enabled:hover,
QPushButton[variant="ghost"]:focus {{ border-color: {EMERALD}; color: {EMERALD}; }}

QPushButton[variant="go"] {{
  border: none; background: {go}; color: {ON_VIVID};
  font-weight: 700; font-size: 16px; padding: 14px 24px;
}}
QPushButton[variant="go"]:enabled:hover {{ background: {go_hover}; }}
QPushButton[variant="go"]:enabled:pressed {{ background: {go_down}; }}
QPushButton[variant="go"]:focus {{ border: 2px solid {p.ink}; padding: 12px 22px; }}

QPushButton[variant="quiet"] {{
  background: transparent; border: 1px solid {p.line}; color: {p.ink_2};
  padding: 6px 14px; font-size: 14px;
}}
QPushButton[variant="quiet"]:enabled:hover,
QPushButton[variant="quiet"]:focus {{ border-color: {p.bad}; color: {p.bad}; }}
QPushButton[variant="quiet"][size="small"] {{ padding: 3px 10px; font-size: 13px; }}

QPushButton[variant="summary"] {{
  background: transparent; border: none; color: {p.muted};
  font-size: 13px; padding: 0; text-align: left;
}}
QPushButton[variant="summary"]:hover,
QPushButton[variant="summary"]:focus {{ color: {EMERALD}; }}

/* ---- controls ---- */

QFrame#controls {{
  background: {p.surface}; border: 1px solid {p.line}; border-radius: {RADIUS}px;
}}

QCheckBox, QRadioButton {{ background: transparent; spacing: 9px; }}
QFrame[role="card"] QCheckBox, QFrame[role="card"] QRadioButton {{ spacing: 0; }}
QCheckBox {{ color: {p.ink_2}; font-size: 14px; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 15px; height: 15px; }}
QCheckBox::indicator {{ image: url("{marks['check-off']}"); }}
QCheckBox::indicator:checked {{ image: url("{marks['check-on']}"); }}
QRadioButton::indicator {{ image: url("{marks['radio-off']}"); }}
QRadioButton::indicator:checked {{ image: url("{marks['radio-on']}"); }}

/* ---- radio groups ---- */

QFrame[role="card"] {{
  background: transparent; border: 1px solid {p.line}; border-radius: {RADIUS}px;
}}
QFrame[role="card"]:hover {{ border-color: {EMERALD}; }}
QFrame[role="card"][checked="true"] {{ border-color: {EMERALD}; background: {p.sunk}; }}
QFrame[role="card"]:disabled {{ border-color: {p.line}; }}

/* ---- results ---- */

QPushButton[role="file"] {{
  background: {p.surface}; border: 1px solid {p.line}; border-radius: {RADIUS}px;
  padding: 0; text-align: left;
}}
QPushButton[role="file"]:hover, QPushButton[role="file"]:focus {{
  border-color: {EMERALD};
}}
QLabel[role="file-name"] {{ font-family: "{mono}"; font-size: 14px; color: {p.ink}; }}
QLabel[role="file-size"] {{ font-family: "{mono}"; font-size: 12px; color: {p.muted}; }}
QLabel[role="file-arrow"] {{ color: {EMERALD}; font-weight: 700; font-size: 14px; }}

/* ---- progress ---- */

QLabel#stage {{ font-size: 15px; }}
QLabel#pct {{ font-family: "{mono}"; font-size: 15px; color: {p.muted}; }}

/* ---- alert ---- */

QFrame#alert {{
  background: {p.bad_bg}; border: 1px solid {p.bad}; border-radius: {RADIUS}px;
}}
QLabel#alert-title {{ color: {p.bad}; }}
QLabel#alert-body {{ color: {p.ink}; font-size: 14px; }}
"""
