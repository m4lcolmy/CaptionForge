"""The page's components as Qt widgets: chips, cards, the track, the rows.

Each class is one piece of the page's visual vocabulary and nothing more. The
window composes them; none of them knows what CaptionForge does.
"""

from __future__ import annotations

import html
import unicodedata
from collections.abc import Callable

from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QEvent,
    QPoint,
    QPropertyAnimation,
    QRect,
    QRectF,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontInfo,
    QFontMetrics,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QPixmap,
    QResizeEvent,
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QBoxLayout,
    QCheckBox,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLayoutItem,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QWidgetItem,
)

from app.interfaces.desktop import theme
from app.interfaces.desktop.theme import EMERALD, LIME, ON_VIVID, SPRING, TEAL

# The page's ``button[disabled] { opacity: 0.45 }`` and ``.radio`` at 0.5.
DISABLED_OPACITY = 0.45
DISABLED_CARD_OPACITY = 0.5


def repolish(widget: QWidget) -> None:
    """Re-read the style sheet after a property its selectors test has changed."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def dim_when_disabled(widget: QWidget, opacity: float = DISABLED_OPACITY) -> None:
    """Fade a widget exactly as far as the page fades its disabled twin."""
    if widget.isEnabled():
        widget.setGraphicsEffect(None)  # type: ignore[arg-type]
        return
    effect = QGraphicsOpacityEffect(widget)
    effect.setOpacity(opacity)
    widget.setGraphicsEffect(effect)


def column(
    spacing: int, margins: tuple[int, int, int, int] = (0, 0, 0, 0)
) -> QVBoxLayout:
    """A vertical flex box: ``flex-direction: column`` with a ``gap``."""
    layout = QVBoxLayout()
    layout.setSpacing(spacing)
    layout.setContentsMargins(*margins)
    return layout


def row(spacing: int, margins: tuple[int, int, int, int] = (0, 0, 0, 0)) -> QHBoxLayout:
    """A horizontal flex box with a ``gap``."""
    layout = QHBoxLayout()
    layout.setSpacing(spacing)
    layout.setContentsMargins(*margins)
    return layout


def reads_right_to_left(text: str) -> bool:
    """Decide direction from the first strong character, as ``dir="auto"`` does."""
    for character in text:
        kind = unicodedata.bidirectional(character)
        if kind in {"R", "AL"}:
            return True
        if kind == "L":
            return False
    return False


# ``body { line-height: 1.55 }``. Qt's own spacing is tighter, and wrapped
# notes set that tight read as a different page.
LINE_HEIGHT = 1.55


def paragraph(markup: str, pixels: float, *, rtl: bool = False, align: str = "") -> str:
    """Wrap rich text so each line is 1.55 times ``pixels`` tall, as in CSS.

    The height is a minimum, not a fixed value: a fallback face such as Noto
    Sans Arabic needs taller lines than Latin text, and a fixed height would
    cut its descenders off. The direction is stated outright because Qt would
    otherwise guess it from the first letter, where the page inherits LTR.
    """
    height = round(LINE_HEIGHT * pixels)
    direction = "rtl" if rtl else "ltr"
    alignment = f' align="{align}"' if align else ""
    return (
        f'<div dir="{direction}"{alignment} style="line-height:{height}px; '
        f'-qt-line-height-type:minimum">{markup}</div>'
    )


class Label(QLabel):
    """A line of text with one of the page's text roles: note, help, title.

    It takes plain text, like ``textContent``, and sets it at the page's line
    height; :meth:`plain` returns what was set. :meth:`set_markup` is for the
    one note that carries a link.
    """

    def __init__(
        self, text: str = "", role: str = "note", *, wrap: bool = True
    ) -> None:
        super().__init__()
        self.setProperty("role", role)
        self.setWordWrap(wrap)
        self.setTextFormat(Qt.TextFormat.RichText)
        self._plain = ""
        self._markup = ""
        self._auto_direction = False
        self._rendered = ""
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt's own name
        """Show plain text; nothing in it is read as markup."""
        self.set_markup(html.escape(text), text)

    def set_markup(self, markup: str, plain: str = "") -> None:
        """Show rich text the caller has already escaped."""
        self._plain = plain
        self._markup = markup
        self._render()

    def plain(self) -> str:
        """The text as it was given."""
        return self._plain

    def follow_direction(self) -> None:
        """Read the direction from the text itself, as ``dir="auto"`` does."""
        self._auto_direction = True
        self._render()

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802 - Qt's own name
        """The style sheet sets the font late; the spacing follows it."""
        super().changeEvent(event)
        if event.type() == QEvent.Type.FontChange:
            self._render()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        """Whether a right-to-left title fits on one line depends on the width."""
        super().resizeEvent(event)
        if self._auto_direction:
            self._render()

    def _render(self) -> None:
        if not self._markup:
            rendered = ""
        else:
            rtl = self._auto_direction and reads_right_to_left(self._plain)
            align = ""
            if rtl and self._fits_one_line():
                # The page's title box is only as wide as its text, so a
                # one-line RTL title still starts beside the thumbnail.
                align = "left"
            rendered = paragraph(
                self._markup, QFontInfo(self.font()).pixelSize(), rtl=rtl, align=align
            )
        if rendered != self._rendered:
            self._rendered = rendered
            super().setText(rendered)

    def _fits_one_line(self) -> bool:
        width = self.contentsRect().width()
        return width > 0 and self.fontMetrics().horizontalAdvance(self._plain) <= width


class SectionLabel(QLabel):
    """The small uppercase heading every section starts with (``.label``)."""

    def __init__(self, text: str = "") -> None:
        super().__init__()
        self.setProperty("role", "label")
        # 0.7rem, weight 600, 0.09em tracking, uppercase.
        self.setFont(theme.font(11, weight=600, spacing=1.0))
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt's own name
        """Show the heading in capitals without changing what callers pass in."""
        super().setText(text.upper())


class ElidedLabel(QLabel):
    """A single line that ends in an ellipsis instead of widening its parent."""

    def __init__(self) -> None:
        super().__init__()
        self._full = ""
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def set_full_text(self, text: str) -> None:
        """Remember the whole text; the visible part follows the width."""
        self._full = text
        self.setToolTip(text)
        self._elide()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        """Re-elide whenever the space available changes."""
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        metrics = self.fontMetrics()
        super().setText(
            metrics.elidedText(self._full, Qt.TextElideMode.ElideRight, self.width())
        )


class Mark(QWidget):
    """The 13 px gradient square beside the name in the top bar."""

    def __init__(self) -> None:
        super().__init__()
        self.setFixedSize(13, 13)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt's own name
        """``linear-gradient(135deg, lime, emerald 55%, teal)`` with 3 px corners."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, QColor(LIME))
        gradient.setColorAt(0.55, QColor(EMERALD))
        gradient.setColorAt(1, QColor(TEAL))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawRoundedRect(QRectF(self.rect()), 3, 3)


class FlowLayout(QLayout):
    """Children in a row that wraps onto new lines, like ``flex-wrap: wrap``."""

    def __init__(self, spacing: int = 6) -> None:
        super().__init__()
        self._items: list[QLayoutItem] = []
        self._gap = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802 - Qt's own name
        """Append one item at the end of the flow."""
        self._items.append(item)

    def count(self) -> int:
        """How many items the flow holds."""
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:  # noqa: N802
        """The item at a position, or None past the end."""
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QLayoutItem | None:  # noqa: N802
        """Remove and return the item at a position."""
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def clear(self) -> None:
        """Remove and delete every widget in the flow."""
        while self._items:
            item = self._items.pop()
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()
        self.invalidate()

    def widgets(self) -> list[QWidget]:
        """The widgets in the flow, in order."""
        return [widget for item in self._items if (widget := item.widget()) is not None]

    def expandingDirections(self) -> Qt.Orientation:  # noqa: N802
        """A flow never asks for more room than its lines need."""
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        """Its height depends on how many lines the width allows."""
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        """The height of the lines at a given width."""
        return self._arrange(QRect(0, 0, width, 0), move=False)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        """Place every item, wrapping at the right edge."""
        super().setGeometry(rect)
        self._arrange(rect, move=True)

    def sizeHint(self) -> QSize:  # noqa: N802
        """The natural size is the smallest one."""
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802
        """Wide enough for the widest item."""
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(
            margins.left() + margins.right(), margins.top() + margins.bottom()
        )

    def _arrange(self, rect: QRect, *, move: bool) -> int:
        x, y, line = rect.x(), rect.y(), 0
        for item in self._items:
            widget = item.widget()
            if widget is not None and widget.isHidden():
                continue
            hint = item.sizeHint()
            if x + hint.width() > rect.right() + 1 and line > 0:
                x = rect.x()
                y += line + self._gap
                line = 0
            if move:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + self._gap
            line = max(line, hint.height())
        return y + line - rect.y()


class Chips(QWidget):
    """A ``ul.chips`` row: chips that wrap, 0.4rem apart."""

    def __init__(self) -> None:
        super().__init__()
        self.flow = FlowLayout(spacing=6)
        self.setLayout(self.flow)
        policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def add(self, widget: QWidget) -> None:
        """Append a chip, or a note, to the row."""
        self.flow.addItem(QWidgetItem(widget))
        widget.setParent(self)
        widget.show()
        self.flow.invalidate()
        self.updateGeometry()

    def clear(self) -> None:
        """Empty the row."""
        self.flow.clear()
        self.updateGeometry()

    def chips(self) -> list[Chip]:
        """The chips in the row, in order."""
        return [widget for widget in self.flow.widgets() if isinstance(widget, Chip)]


class Chip(QAbstractButton):
    """A pill with a monospace label and an optional muted ``.kind`` beside it.

    ``interactive=False`` is the page's ``span.chip``: it looks the same and
    does nothing. A picked chip wears the spring-to-teal ramp, and it stays
    vivid while disabled so a running row says which one is running.
    """

    PADDING_X = 11
    PADDING_Y = 5
    GAP = 6

    def __init__(self, text: str, kind: str = "", *, interactive: bool = True) -> None:
        super().__init__()
        self.setText(text)
        self._kind = kind
        self._picked = False
        self._interactive = interactive
        self.setAccessibleName(f"{text} {kind}".strip())
        self.setFocusPolicy(
            Qt.FocusPolicy.TabFocus if interactive else Qt.FocusPolicy.NoFocus
        )
        if not interactive:
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)
        self._sync_cursor()

    @property
    def kind(self) -> str:
        """The muted second part of the label."""
        return self._kind

    @property
    def picked(self) -> bool:
        """Whether the chip wears the gradient."""
        return self._picked

    def set_picked(self, picked: bool) -> None:
        """Pick or unpick the chip, which also changes how it reads when pressed."""
        self._picked = picked
        self.setAccessibleDescription("selected" if picked else "")
        self._sync_cursor()
        self.update()

    def _fonts(self) -> tuple[QFont, QFont]:
        weight = 600 if self._picked else 400
        return (
            theme.font(13, weight=weight, mono=True),
            theme.font(12, weight=weight, mono=True),
        )

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt's own name
        """Label, kind, padding and border, at the page's 1.55 line height."""
        # Measured bold, so picking a chip never makes the row reflow.
        main, small = (theme.font(13, weight=600, mono=True), theme.font(12, mono=True))
        width = QFontMetrics(main).horizontalAdvance(self.text())
        if self._kind:
            width += self.GAP + QFontMetrics(small).horizontalAdvance(self._kind)
        height = round(13 * 1.55)
        return QSize(width + 2 * self.PADDING_X + 2, height + 2 * self.PADDING_Y + 2)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        """A chip never squeezes below its label."""
        return self.sizeHint()

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        """Keep the cursor honest about whether a click would do anything."""
        super().changeEvent(event)
        self._sync_cursor()

    def _sync_cursor(self) -> None:
        if not self._interactive:
            return
        if self.isEnabled():
            shape = Qt.CursorShape.PointingHandCursor
        elif self._picked:
            shape = Qt.CursorShape.BusyCursor
        else:
            shape = Qt.CursorShape.ForbiddenCursor
        self.setCursor(shape)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        """Draw the pill, then the label, then the kind."""
        palette = theme.current()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self.isEnabled() and not self._picked:
            painter.setOpacity(DISABLED_OPACITY)
        bounds = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = bounds.height() / 2
        hovered = (
            self._interactive
            and self.isEnabled()
            and not self._picked
            and (self.underMouse() or self.hasFocus())
        )
        if self._picked:
            gradient = QLinearGradient(bounds.topLeft(), bounds.bottomRight())
            gradient.setColorAt(0, QColor(SPRING))
            gradient.setColorAt(1, QColor(TEAL))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(gradient))
            text_color = QColor(ON_VIVID)
            kind_color = QColor(ON_VIVID)
            kind_color.setAlphaF(0.75)
        else:
            painter.setPen(QPen(QColor(EMERALD if hovered else palette.line), 1))
            painter.setBrush(QColor(palette.surface))
            text_color = QColor(EMERALD if hovered else palette.ink_2)
            kind_color = QColor(palette.muted)
        painter.drawRoundedRect(bounds, radius, radius)

        main, small = self._fonts()
        painter.setFont(main)
        painter.setPen(text_color)
        text_width = painter.fontMetrics().horizontalAdvance(self.text())
        inner = QRectF(bounds).adjusted(self.PADDING_X, 0, -self.PADDING_X, 0)
        centre = Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        painter.drawText(inner, centre, self.text())
        if self._kind:
            painter.setFont(small)
            painter.setPen(kind_color)
            painter.drawText(
                inner.adjusted(text_width + self.GAP, 0, 0, 0), centre, self._kind
            )


class ProgressTrack(QWidget):
    """The 7 px rounded track and its gradient fill, easing between values."""

    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(7)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._fraction = 0.0
        # ``transition: width 420ms ease``.
        self._animation = QPropertyAnimation(self, b"fraction", self)
        self._animation.setDuration(420)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    def _get_fraction(self) -> float:
        return self._fraction

    def _set_fraction(self, value: float) -> None:
        self._fraction = max(0.0, min(1.0, value))
        self.update()

    fraction = Property(float, _get_fraction, _set_fraction)

    def set_percent(self, percent: float, *, animate: bool = True) -> None:
        """Move the fill to a percentage, sliding unless told to jump."""
        target = max(0.0, min(100.0, percent)) / 100
        self._animation.stop()
        if not animate or target < self._fraction:
            self._set_fraction(target)
            return
        self._animation.setStartValue(self._fraction)
        self._animation.setEndValue(target)
        self._animation.start()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        """Track in ``--sunk``; fill in the lime, spring, teal ramp."""
        palette = theme.current()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bounds = QRectF(self.rect())
        radius = bounds.height() / 2
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(palette.sunk))
        painter.drawRoundedRect(bounds, radius, radius)
        width = bounds.width() * self._fraction
        if width <= 0:
            return
        fill = QRectF(0, 0, width, bounds.height())
        gradient = QLinearGradient(fill.topLeft(), fill.topRight())
        gradient.setColorAt(0, QColor(LIME))
        gradient.setColorAt(0.4, QColor(SPRING))
        gradient.setColorAt(1, QColor(TEAL))
        clip = QPainterPath()
        clip.addRoundedRect(bounds, radius, radius)
        painter.setClipPath(clip)
        painter.setBrush(QBrush(gradient))
        painter.drawRoundedRect(fill, radius, radius)


class Thumbnail(QWidget):
    """A 160 by 90 picture with rounded corners, cropped to fill (``cover``)."""

    def __init__(self) -> None:
        super().__init__()
        self.setFixedSize(160, 90)
        self._pixmap: QPixmap | None = None

    def set_pixmap(self, pixmap: QPixmap | None) -> None:
        """Show a picture, or only the sunk placeholder while there is none."""
        self._pixmap = pixmap
        self.update()

    def has_picture(self) -> bool:
        """Whether a picture has arrived."""
        return self._pixmap is not None and not self._pixmap.isNull()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        """Placeholder first, then the picture clipped to the same corners."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        bounds = QRectF(self.rect())
        clip = QPainterPath()
        clip.addRoundedRect(bounds, theme.RADIUS, theme.RADIUS)
        painter.setClipPath(clip)
        painter.fillRect(bounds, QColor(theme.current().sunk))
        if not self.has_picture():
            return
        assert self._pixmap is not None
        ratio = self.devicePixelRatioF()
        scaled = self._pixmap.scaled(
            round(self.width() * ratio),
            round(self.height() * ratio),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        scaled.setDevicePixelRatio(ratio)
        size = scaled.deviceIndependentSize()
        left = (self.width() - size.width()) / 2
        top = (self.height() - size.height()) / 2
        painter.drawPixmap(QPoint(round(left), round(top)), scaled)


class CardText(QLabel):
    """A card's bold name and muted note, flowing together like ``b`` and ``em``."""

    def __init__(self, name: str, note: str) -> None:
        super().__init__()
        self._name = name
        self._note = note
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        theme.theme().changed.connect(self._render)
        self._render()

    def set_note(self, note: str) -> None:
        """Replace the muted part, as the device card does when CUDA is absent."""
        self._note = note
        self._render()

    def _render(self) -> None:
        palette = theme.current()
        name = (
            f'<span style="font-weight:600; font-size:14px; color:{palette.ink}">'
            f"{html.escape(self._name)}</span>"
        )
        note = (
            f'&nbsp; <span style="font-size:13px; color:{palette.muted}">'
            f"{html.escape(self._note)}</span>"
            if self._note
            else ""
        )
        # The line box is the 15 px body's, whichever span is tallest.
        self.setText(paragraph(name + note, 15))


class OptionCard(QFrame):
    """A ``label.radio``: a bordered row holding a radio or a checkbox and text.

    The whole card is the click target, and it fills with ``--sunk`` while its
    input is checked, which is what ``.radio:has(input:checked)`` does.
    """

    def __init__(
        self, indicator: QRadioButton | QCheckBox, name: str, note: str
    ) -> None:
        super().__init__()
        self.setProperty("role", "card")
        self.setProperty("checked", "false")
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.indicator = indicator
        self.indicator.setText("")
        self.indicator.setCursor(Qt.CursorShape.PointingHandCursor)
        self.text = CardText(name, note)
        # The input takes exactly its 15 px, so the gap is the page's 0.6rem,
        # and sits 0.22rem low, level with the first line of text.
        self.indicator.setFixedSize(15, 15)
        mark = column(0, (0, 4, 0, 0))
        mark.addWidget(self.indicator)
        mark.addStretch(1)
        layout = row(10, (10, 7, 10, 7))
        layout.addLayout(mark)
        layout.addWidget(self.text, 1)
        self.setLayout(layout)
        self.indicator.toggled.connect(self._sync)
        self._sync()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """A click anywhere on the card is a click on its input."""
        if event.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            if isinstance(self.indicator, QRadioButton):
                self.indicator.setChecked(True)
            else:
                self.indicator.toggle()
            self.indicator.setFocus(Qt.FocusReason.MouseFocusReason)
        super().mouseReleaseEvent(event)

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        """Fade a disabled card to half strength, and say so with the cursor."""
        super().changeEvent(event)
        if event.type() == QEvent.Type.EnabledChange:
            dim_when_disabled(self, DISABLED_CARD_OPACITY)
            shape = (
                Qt.CursorShape.PointingHandCursor
                if self.isEnabled()
                else Qt.CursorShape.ForbiddenCursor
            )
            self.setCursor(shape)
            self.indicator.setCursor(shape)

    def _sync(self) -> None:
        self.setProperty("checked", "true" if self.indicator.isChecked() else "false")
        repolish(self)


class Button(QPushButton):
    """One of the page's three buttons: ``ghost``, ``go`` or ``quiet``."""

    def __init__(self, text: str, variant: str) -> None:
        super().__init__(text)
        self.setProperty("variant", variant)
        # Focus from Tab only, so a clicked button does not stay outlined:
        # that is the difference between :focus and :focus-visible.
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802 - Qt's own name
        """``button[disabled] { cursor: not-allowed; opacity: 0.45 }``."""
        super().changeEvent(event)
        if event.type() == QEvent.Type.EnabledChange:
            dim_when_disabled(self)
            self.setCursor(
                Qt.CursorShape.PointingHandCursor
                if self.isEnabled()
                else Qt.CursorShape.ForbiddenCursor
            )


class FileRow(QPushButton):
    """One produced file: its name, its size, and a mark that says it opens."""

    def __init__(self, name: str, size: str) -> None:
        super().__init__()
        self.setProperty("role", "file")
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Open with this computer's usual app")
        self.setAccessibleName(f"Open {name}")
        layout = row(12, (14, 11, 14, 11))
        self.name_label = QLabel(name)
        self.name_label.setProperty("role", "file-name")
        size_label = QLabel(size)
        size_label.setProperty("role", "file-size")
        arrow = QLabel("↗")
        arrow.setProperty("role", "file-arrow")
        for label in (self.name_label, size_label, arrow):
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.name_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        layout.addWidget(self.name_label, 1)
        layout.addWidget(size_label)
        layout.addWidget(arrow)
        self.setLayout(layout)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt's own name
        """As tall as its labels need; as wide as the column it sits in."""
        layout = self.layout()
        return layout.sizeHint() if layout is not None else super().sizeHint()

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        """Never shorter than one line of text."""
        return QSize(0, self.sizeHint().height())


class Disclosure(QWidget):
    """``<details><summary>``: a quiet toggle and the panel it opens."""

    toggled = Signal(bool)

    def __init__(self, summary: str, body: QWidget) -> None:
        super().__init__()
        self._summary = summary
        self.button = QPushButton()
        self.button.setProperty("variant", "summary")
        self.button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.button.clicked.connect(lambda: self.set_open(not self.is_open()))
        self.body = body
        self._layout = column(0)
        self._layout.addWidget(self.button, 0, Qt.AlignmentFlag.AlignLeft)
        self._layout.addWidget(body)
        self.setLayout(self._layout)
        self.set_open(False)

    def is_open(self) -> bool:
        """Whether the panel is showing."""
        return not self.body.isHidden()

    def set_open(self, opened: bool) -> None:
        """Open or close the panel, turning the triangle to match."""
        self.body.setVisible(opened)
        self.button.setText(f"{'▼' if opened else '▶'}  {self._summary}")
        # ``.options[open] summary { margin-bottom: .85rem }``.
        self._layout.setSpacing(14 if opened else 0)
        self.toggled.emit(opened)


def clear_layout(layout: QBoxLayout) -> None:
    """Remove and delete every widget a box layout holds."""
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget() if item is not None else None
        if widget is not None:
            widget.hide()
            widget.deleteLater()


def on_click(widget: QAbstractButton, action: Callable[[], None]) -> None:
    """Connect a click without passing Qt's ``checked`` flag along."""
    widget.clicked.connect(lambda _checked=False: action())
