import sys

sys.path.append("docs/snippets")

from napari_widgets_render import render_widget

from panseg.viewer_napari.widgets.input import Input_Tab

input_tab = Input_Tab()


html = render_widget(input_tab.widget_open_file)
print(html)
