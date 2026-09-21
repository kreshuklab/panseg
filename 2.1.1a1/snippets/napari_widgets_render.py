import base64

from napari.qt import get_stylesheet
from napari.resources._icons import _theme_path
from qtpy import QtCore, QtGui

css_style = """
    <style>
        .light-mode-message,
        .dark-mode-message {
            display: none;
        }

        /* Show dark mode message only in dark mode */
        [data-md-color-scheme="slate"] .dark-mode-message {
            display: block;
        }

        /* Show dark mode message only in dark mode */
        [data-md-color-scheme="default"] .light-mode-message {
            display: block;
        }
    </style>
    """

QtCore.QDir.addSearchPath("theme_dark", str(_theme_path("dark")))
QtCore.QDir.addSearchPath("theme_light", str(_theme_path("light")))


def get_widget_title(widget) -> str:
    return f"<h2>Widget: {widget.call_button.text}</h2>"


def get_parameters_tooltips(widget) -> str:
    widget.show()
    doc_str = "<ul>"
    for key in widget._param_options.keys():
        _widget = widget.__getattr__(key)
        if _widget.visible:
            doc_str += f"<li><b>{_widget.label}</b>: {_widget.tooltip}</li>\n"
    doc_str += "</ul>"
    widget.close()
    return doc_str


def get_doc_string(widget) -> str:
    return widget._function.__doc__


def _render_widget(widget, theme="dark"):
    widget.show()
    widget.native.setFixedWidth(550)
    widget.native.setStyleSheet(get_stylesheet(theme))
    geometry = widget.native.geometry()
    pixmap = QtGui.QPixmap(geometry.size())
    widget.native.render(pixmap)
    qimage = pixmap.toImage()
    widget.close()
    buffer = QtCore.QBuffer()
    buffer.open(QtCore.QBuffer.ReadWrite)
    qimage.save(buffer, "PNG")
    base64_image = base64.b64encode(buffer.data()).decode("utf-8")
    buffer.close()

    return f'<img src="data:image/png;base64,{base64_image}" alt="Screenshot">'


def get_widget_images(widget) -> str:
    light_image = _render_widget(widget, theme="light")
    dark_image = _render_widget(widget, theme="dark")
    return f"""
        <p class="light-mode-message">{light_image}</p>
        <p class="dark-mode-message">{dark_image}</p>
        """


def _format_widget(name="", images="", doc_str="", param_doc_str="") -> str:
    return f"""
        {css_style}
        <body>
            {name}
            {images}
            {doc_str}
            {param_doc_str}
        </body>
        """


def render_widget(widget, skip_name=True, skip_doc=False) -> str:
    widget_name = get_widget_title(widget) if not skip_name else ""
    widget_images = get_widget_images(widget)
    widget_doc = get_doc_string(widget) if not skip_doc else ""
    widget_par_doc = get_parameters_tooltips(widget)
    return _format_widget(widget_name, widget_images, widget_doc, widget_par_doc)
